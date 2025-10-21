from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset08281 import save_emb


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class FlashMultiHeadAttention(nn.Module):
    """一个高效的多头注意力实现，优先使用Flash Attention"""
    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float = 0.0, use_flash: bool = True):
        super().__init__()
        assert hidden_units % num_heads == 0
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate
        self.use_flash = use_flash

        self.q_proj = nn.Linear(hidden_units, hidden_units)
        self.k_proj = nn.Linear(hidden_units, hidden_units)
        self.v_proj = nn.Linear(hidden_units, hidden_units)
        self.out_proj = nn.Linear(hidden_units, hidden_units)

    def forward(self, query, key, value, attn_mask=None):
        B, L, H = query.shape
        Q = self.q_proj(query).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention") and self.use_flash:
            mask = (~attn_mask).unsqueeze(1) if attn_mask is not None else None
            out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, dropout_p=self.dropout_rate if self.training else 0.0)
        else:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
            if attn_mask is not None:
                scores = scores.masked_fill((~attn_mask).unsqueeze(1), float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout_rate, training=self.training)
            out = torch.matmul(attn, V)
        
        out = out.transpose(1, 2).contiguous().view(B, L, H)
        return self.out_proj(out)

class PointWiseFeedForward(nn.Module):
    """Transformer中的前馈网络层"""
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units * 4, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv1d(hidden_units * 4, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.gelu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        return outputs.transpose(-1, -2)

class ThreeTowerModel(nn.Module):
    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):
        super().__init__()
        self.dev = args.device
        self.hidden_units = args.hidden_units
        self.temp = getattr(args, "temp", 0.029)
        
        # --- 1. 初始化所有特征的Embedding层 ---
        self._init_feature_embeddings(item_num, feat_statistics, feat_types, args)

        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        
        # --- 2. 定义各个模块 ---
        # 物品/用户特征融合时使用的归一化层
        self.feature_fusion_norm = nn.LayerNorm(self.hidden_units)
        
        # 用户行为塔 (Transformer)
        self.attention_layers = nn.ModuleList()
        for _ in range(args.num_blocks):
            self.attention_layers.append(nn.ModuleList([
                nn.LayerNorm(self.hidden_units, eps=1e-8),
                FlashMultiHeadAttention(self.hidden_units, args.num_heads, args.dropout_rate),
                nn.LayerNorm(self.hidden_units, eps=1e-8),
                PointWiseFeedForward(self.hidden_units, args.dropout_rate)
            ]))
        self.behavior_last_layernorm = nn.LayerNorm(self.hidden_units, eps=1e-8)
        
        # 用户静态塔 (MLP)
        self.user_tower_mlp = nn.Sequential(
            nn.Linear(self.hidden_units, self.hidden_units * 2),
            nn.GELU(),
            nn.Dropout(p=args.dropout_rate),
            nn.Linear(self.hidden_units * 2, self.hidden_units)
        )
        self.user_tower_norm = nn.LayerNorm(self.hidden_units)

        # 用户静态塔和行为塔融合时使用的归一化层
        self.fusion_norm = nn.LayerNorm(self.hidden_units)

    def _init_feature_embeddings(self, item_num, feat_statistics, feat_types, args):
        # 物品ID嵌入
        self.item_emb = nn.Embedding(item_num + 1, self.hidden_units, padding_idx=0)
        
        # 行为类型和位置嵌入 (用于行为塔)
        self.action_emb = nn.Embedding(args.action_type_vocab_size, self.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen, self.hidden_units)

        # 物品稀疏特征嵌入
        self.ITEM_SPARSE_FEAT_KEYS = list(feat_types['item_sparse'])
        self.item_sparse_emb = nn.ModuleDict()
        for fid in self.ITEM_SPARSE_FEAT_KEYS:
            # feat_statistics['f'][fid] 包含了该特征的词典
            vocab_size = len(feat_statistics.get(fid, {}))
            self.item_sparse_emb[fid] = nn.Embedding(vocab_size + 1, self.hidden_units, padding_idx=0)
        
        # 用户稀疏特征嵌入
        self.USER_SPARSE_FEAT_KEYS = list(feat_types['user_sparse'])
        self.user_sparse_emb = nn.ModuleDict()
        for fid in self.USER_SPARSE_FEAT_KEYS:
            vocab_size = len(feat_statistics.get(fid, {}))
            self.user_sparse_emb[fid] = nn.Embedding(vocab_size + 1, self.hidden_units, padding_idx=0)
            
        # 用户数组特征嵌入
        self.USER_ARRAY_FEAT_KEYS = list(feat_types.get('user_array', []))
        self.user_array_emb = nn.ModuleDict()
        for fid in self.USER_ARRAY_FEAT_KEYS:
            vocab_size = len(feat_statistics.get(fid, {}))
            self.user_array_emb[fid] = nn.Embedding(vocab_size + 1, self.hidden_units, padding_idx=0)

        # (可选) 多模态特征的线性转换层
        EMB_SHAPE_DICT = {"81": 32} # 示例
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types.get('item_emb', []) if k in EMB_SHAPE_DICT}
        self.emb_transform = nn.ModuleDict()
        for fid, dim in self.ITEM_EMB_FEAT.items():
            self.emb_transform[fid] = nn.Linear(dim, self.hidden_units)

    # --- TOWER 1: 物品塔 ---
    def encode_item(self, item_ids, sparse_feats, mm_batch=None):
        all_embs = [self.item_emb(item_ids)]
        
        # 聚合稀疏特征
        for i, fid in enumerate(self.ITEM_SPARSE_FEAT_KEYS):
            all_embs.append(self.item_sparse_emb[fid](sparse_feats[..., i]))
        
        # (可选) 聚合多模态特征
        if mm_batch:
            for fid, tensor in mm_batch.items():
                all_embs.append(self.emb_transform[fid](tensor))
        
        combined_emb = torch.stack(all_embs, dim=0).sum(dim=0)
        return self.feature_fusion_norm(combined_emb)

    # --- TOWER 2: 用户行为塔 ---
    def encode_sequence_behavior(self, seq_ids, action_seqs, seq_sparse, seq_mm=None):
        # 行为序列中的每个物品，都通过物品塔得到其基础嵌入
        # 为提高效率，将 (B, L) 展平为 (B*L) 送入物品塔，再还原
        B, L = seq_ids.shape
        seq_item_embs = self.encode_item(
            seq_ids.view(-1),
            seq_sparse.view(-1, len(self.ITEM_SPARSE_FEAT_KEYS)),
            {fid: tensor.view(-1, tensor.shape[-1]) for fid, tensor in seq_mm.items()} if seq_mm and seq_mm.items() else None
        ).view(B, L, -1)

        # 加入行为类型和位置信息
        seq_embs = seq_item_embs
        seq_embs += self.action_emb(action_seqs)
        positions = torch.arange(seq_ids.shape[1], device=self.dev).unsqueeze(0).expand_as(seq_ids)
        seq_embs += self.pos_emb(positions)
        seq_embs = self.emb_dropout(seq_embs)
        
        # 通过Transformer层
        padding_mask = (seq_ids != 0).unsqueeze(1)
        causal_mask = torch.tril(torch.ones((L, L), dtype=torch.bool, device=self.dev))
        attn_mask = padding_mask & causal_mask

        for norm1, attn, norm2, ffn in self.attention_layers:
            seq_embs = seq_embs + F.dropout(attn(norm1(seq_embs), norm1(seq_embs), norm1(seq_embs), attn_mask=attn_mask), p=self.emb_dropout.p, training=self.training)
            seq_embs = seq_embs + F.dropout(ffn(norm2(seq_embs)), p=self.emb_dropout.p, training=self.training)
            
        return self.behavior_last_layernorm(seq_embs)
    
    # --- TOWER 3: 用户静态塔 ---
    def encode_user_static(self, user_sparse_feats, user_array_feats):
        all_user_feature_embs = []

        # 1. 处理稀疏特征
        for i, fid in enumerate(self.USER_SPARSE_FEAT_KEYS):
            all_user_feature_embs.append(self.user_sparse_emb[fid](user_sparse_feats[..., i]))
        
        # 2. 处理数组特征 (Embedding + Average Pooling)
        for fid in self.USER_ARRAY_FEAT_KEYS:
            array_input = user_array_feats[fid] # (B, max_len)
            mask = (array_input != 0).float().unsqueeze(-1) # (B, max_len, 1)
            
            embedded_array = self.user_array_emb[fid](array_input) # (B, max_len, H)
            embedded_array = embedded_array * mask
            
            sum_pooled_emb = torch.sum(embedded_array, dim=1) # (B, H)
            effective_lengths = mask.sum(dim=1).clamp(min=1e-9) # (B, 1)
            avg_pooled_emb = sum_pooled_emb / effective_lengths
            all_user_feature_embs.append(avg_pooled_emb)
        
        # 3. 融合所有用户特征的嵌入
        combined_emb = torch.stack(all_user_feature_embs, dim=0).sum(dim=0)
        processed_emb = self.user_tower_mlp(combined_emb)
        return self.user_tower_norm(processed_emb)

    # --- 模型主流程 ---
    def forward(self, seq_ids, pos_ids, action_seqs, 
                user_sparse, user_arrays,
                seq_sparse, pos_sparse, 
                seq_mm, pos_mm, full_histories, writer=None):
        
        # 1. 行为塔编码 -> 动态用户兴趣
        seq_embs = self.encode_sequence_behavior(seq_ids, action_seqs, seq_sparse, seq_mm)
        
        # 2. 静态塔编码 -> 用户静态画像
        user_static_embs = self.encode_user_static(user_sparse, user_arrays)
        
        # 3. 融合: 行为序列表示 + 静态画像表示
        final_user_embs = self.fusion_norm(seq_embs + user_static_embs.unsqueeze(1))
        
        # 4. 物品塔编码 -> 正负样本物品表示
        B, L = pos_ids.shape
        pos_item_embs = self.encode_item(
            pos_ids.view(-1),
            pos_sparse.view(-1, len(self.ITEM_SPARSE_FEAT_KEYS)),
            {fid: tensor.view(-1, tensor.shape[-1]) for fid, tensor in pos_mm.items()} if pos_mm and pos_mm.items() else None
        ).view(B, L, -1)
        
        neg_item_embs = pos_item_embs.view(-1, self.hidden_units)

        # 5. 计算损失
        loss_mask = (pos_ids != 0)
        return self.compute_infonce_loss(final_user_embs, pos_item_embs, neg_item_embs, loss_mask, full_histories, pos_ids)

    def compute_infonce_loss(self, user_embs, pos_item_embs, neg_item_embs, loss_mask, user_full_histories, all_pos_ids):
        user_norm = F.normalize(user_embs, p=2, dim=-1)
        pos_norm = F.normalize(pos_item_embs, p=2, dim=-1)
        neg_norm = F.normalize(neg_item_embs, p=2, dim=-1)

        pos_logits = (user_norm * pos_norm).sum(dim=-1, keepdim=True)
        neg_logits = torch.matmul(user_norm, neg_norm.transpose(0, 1))

        B, L, _ = user_embs.shape
        
        # In-batch 负采样时的 mask，防止正样本被当作负样本
        pos_to_seq = torch.arange(B, device=self.dev).repeat_interleave(L)
        query_seq_ids = torch.arange(B, device=self.dev).unsqueeze(1).expand(B, L)
        same_seq_mask = (query_seq_ids.unsqueeze(-1) == pos_to_seq.unsqueeze(0).unsqueeze(0))
        
        # In-batch 负采样时的 mask，防止用户历史点击过的物品被当作负样本
        neg_candidate_ids = all_pos_ids.view(-1)
        user_history_mask = torch.zeros(B, B * L, dtype=torch.bool, device=self.dev)
        for b in range(B):
            history_tensor = user_full_histories[b]
            if history_tensor.numel() > 0:
                user_history_mask[b] = torch.isin(neg_candidate_ids, history_tensor.to(self.dev))
        
        user_history_mask = user_history_mask.unsqueeze(1).expand(B, L, B * L)

        final_mask = same_seq_mask | user_history_mask
        neg_logits.masked_fill_(final_mask, -1e9)
        
        logits = torch.cat([pos_logits, neg_logits], dim=-1)[loss_mask] / self.temp
        labels = torch.zeros(logits.size(0), device=self.dev, dtype=torch.long)
        loss = F.cross_entropy(logits, labels)
        
        # 记录一些监控指标
        with torch.no_grad():
            self.last_pos_sim = pos_logits.mean().item()
            self.last_neg_sim = neg_logits[neg_logits > -1e8].mean().item()
            k = min(10, logits.size(1))
            if k > 0:
                self.last_acc10 = (torch.topk(logits, k=k, dim=-1)[1] == labels.unsqueeze(-1)).any(1).float().mean().item()
            else:
                self.last_acc10 = 0.0
            
        return loss


    # ... (predict, save_item_emb, 和其他辅助/绘图函数保持不变)
    def predict(self, seq_ids, sparse_feats, mm_batch, action_seqs=None, topk=None, chunk_size=None):
        seq_embs = self.encode_sequence(seq_ids, sparse_feats, mm_batch, action_seqs)
        user_vector = seq_embs[:, -1, :]
        all_item_emb = self.item_emb.weight
        if topk is None:
            return torch.matmul(user_vector, all_item_emb.T)
        if chunk_size is None: chunk_size = all_item_emb.size(0)
        topk_scores, topk_indices = [], []
        for start in range(0, all_item_emb.size(0), chunk_size):
            end = min(start + chunk_size, all_item_emb.size(0))
            scores = torch.matmul(user_vector, all_item_emb[start:end].T)
            vals, idxs = torch.topk(scores, k=min(topk, scores.size(1)), dim=-1)
            topk_scores.append(vals)
            topk_indices.append(idxs + start)
        topk_scores = torch.cat(topk_scores, dim=-1)
        topk_indices = torch.cat(topk_indices, dim=-1)
        final_vals, final_idxs = torch.topk(topk_scores, k=topk, dim=-1)
        final_indices = torch.gather(topk_indices, 1, final_idxs)
        return final_vals, final_indices

    def save_item_emb(self, item_ids, retrieval_ids, sparse_feats, mm_batch, save_path, batch_size=1024):
        all_embs = []
        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))
            batch_sparse_feats = sparse_feats[start_idx:end_idx]
            batch_mm_batch = {fid: mm_tensor[start_idx:end_idx] for fid, mm_tensor in mm_batch.items()}
            # 注意: 保存 item emb 时不使用 action_type
            batch_emb = self.encode_sequence(
                torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0),
                batch_sparse_feats.unsqueeze(0),
                {fid: tensor.unsqueeze(0) for fid, tensor in batch_mm_batch.items()}
            ).squeeze(0)
            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))
