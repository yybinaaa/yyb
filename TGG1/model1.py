import torch
import torch.nn as nn
import torch.nn.functional as F

# ... (所有子模块: RMSNorm, LazyCrossAttention, etc. 保持完全不变) ...

# (此处省略未改变的子模块代码，请保留您文件中的原始内容)
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

class LazyCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim, n_heads, n_kv_heads = config.hidden_dim, config.n_heads, config.n_kv_heads
        assert n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = hidden_dim // n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.q_proj = nn.Linear(hidden_dim, n_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, hidden_dim, bias=False)
    def forward(self, x: torch.Tensor, context_kv: torch.Tensor) -> torch.Tensor:
        bs, seq_len, _ = x.shape
        context_len = context_kv.shape[1]
        q = self.q_proj(x).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        kv = context_kv.view(bs, context_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        k, v = kv, kv
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.o_proj(output)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim, n_heads, n_kv_heads = config.hidden_dim, config.n_heads, config.n_kv_heads
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = hidden_dim // n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.q_proj = nn.Linear(hidden_dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, hidden_dim, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, seq_len, _ = x.shape
        q = self.q_proj(x).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bs, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bs, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.o_proj(output)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class LazyDecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cross_attn_norm = RMSNorm(config.hidden_dim)
        self.cross_attention = LazyCrossAttention(config)
        self.self_attn_norm = RMSNorm(config.hidden_dim)
        self.self_attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_dim)
        self.feed_forward = FeedForward(dim=config.hidden_dim, hidden_dim=config.ffn_hidden_dim)
    def forward(self, x: torch.Tensor, context_kv: torch.Tensor) -> torch.Tensor:
        h = x + self.cross_attention(self.cross_attn_norm(x), context_kv)
        h2 = h + self.self_attention(self.self_attn_norm(h))
        out = h2 + self.feed_forward(self.ffn_norm(h2))
        return out

class ContextProcessor(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.embedding_tables = nn.ModuleDict()
        
        self.user_sparse_keys = list(dataset.USER_SPARSE_FEAT.keys())
        # [核心修正] 我们不再需要 item_sparse_keys，因为RL阶段简化了历史
        # self.item_sparse_keys = list(dataset.ITEM_SPARSE_FEAT.keys())

        # 创建 Embedding 表
        self.embedding_tables['item_id'] = nn.Embedding(dataset.itemnum + 1, self.hidden_dim, padding_idx=0)
        for name, vocab_size in dataset.USER_SPARSE_FEAT.items():
            self.embedding_tables[name] = nn.Embedding(vocab_size + 1, self.hidden_dim, padding_idx=0)
        # [核心修正] 暂时移除历史物品稀疏特征的Embedding表，因为我们不用它了
        # for name, vocab_size in dataset.ITEM_SPARSE_FEAT.items():
        #     self.embedding_tables[name] = nn.Embedding(vocab_size + 1, self.hidden_dim, padding_idx=0)
        
        kv_dim = config.n_kv_heads * (config.hidden_dim // config.n_heads)
        self.kv_proj = nn.Linear(self.hidden_dim, kv_dim)
        self.norm = RMSNorm(kv_dim)

    def forward(self, context_features: dict) -> torch.Tensor:
        all_context_embs = []
        
        # 1. 处理用户稀疏特征 (使用固定顺序)
        user_sparse = context_features['user_sparse']
        for i, name in enumerate(self.user_sparse_keys):
            emb = self.embedding_tables[name](user_sparse[:, i]).unsqueeze(1)
            all_context_embs.append(emb)
        
        # 2. 处理历史行为序列 (只用 item_id)
        history_item_ids = context_features['history_item_ids']
        history_embs = self.embedding_tables['item_id'](history_item_ids)
        
        # [核心修正] 移除融合历史物品稀疏特征的逻辑
        # history_item_sparse = context_features['history_item_sparse']
        # for i, name in enumerate(self.item_sparse_keys):
        #     history_embs += self.embedding_tables[name](history_item_sparse[:, :, i])
        
        all_context_embs.append(history_embs)
        
        # 3. 拼接
        context_tensor = torch.cat(all_context_embs, dim=1)
        
        projected_context = self.kv_proj(context_tensor)
        
        return self.norm(projected_context)

class ModelConfig:
    hidden_dim: int = 256
    n_heads: int = 8
    n_kv_heads: int = 4
    n_layers: int = 4
    ffn_hidden_dim: int = 512

class OneRecV2Model(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.context_processor = ContextProcessor(config, dataset)
        
        # [核心升级] 使用来自 Dataset 的新 VOCAB_SIZE
        self.target_emb = nn.Embedding(dataset.VOCAB_SIZE, config.hidden_dim, padding_idx=dataset.PAD_TOKEN_ID)
        
        self.decoder_blocks = nn.ModuleList([LazyDecoderBlock(config) for _ in range(config.n_layers)])
        self.output_norm = RMSNorm(config.hidden_dim)
        
        # [核心升级] 输出层的维度也更新为新的 VOCAB_SIZE
        self.output_proj = nn.Linear(config.hidden_dim, dataset.VOCAB_SIZE, bias=False)

    def forward(self, context_features: dict, target_ids: torch.Tensor):
        context_kv = self.context_processor(context_features)
        h = self.target_emb(target_ids)
        for layer in self.decoder_blocks:
            h = layer(h, context_kv)
        h = self.output_norm(h)
        logits = self.output_proj(h)
        return logits
    #@torch.no_grad()
    def get_action_prob(self, context_features: dict, action_ids: torch.Tensor):
        """
        计算在给定上下文下，采取某个动作(生成某个语义ID序列)的对数概率。
        
        Args:
            context_features: 上下文特征字典。
            action_ids: 动作(目标物品)的语义ID序列, shape: [bs, seq_len]。
            
        Returns:
            log_probs: 每个样本的动作对数概率, shape: [bs]。
        """
        # 构造 decoder 输入: [BOS, s1, s2]
        bos_token = torch.full((action_ids.shape[0], 1), self.target_emb.padding_idx + 1, # a placeholder for BOS
                               dtype=torch.long, device=action_ids.device)
        decoder_input_ids = torch.cat([bos_token, action_ids[:, :-1]], dim=1)
        
        # 获取模型的预测 logits
        logits = self.forward(context_features, decoder_input_ids)
        
        # 计算每个位置的 log_softmax 概率
        log_probs_dist = F.log_softmax(logits, dim=-1)
        
        # 从概率分布中收集真实 action_ids 对应的概率
        # 使用 gather 方法高效实现
        # action_ids.unsqueeze(-1) -> [bs, seq_len, 1]
        action_log_probs = torch.gather(log_probs_dist, 2, action_ids.unsqueeze(-1)).squeeze(-1)
        
        # 忽略 padding 位置的概率 (如果标签是 padding_idx)
        pad_mask = (action_ids != self.target_emb.padding_idx)
        action_log_probs = action_log_probs * pad_mask
        
        # 将一个序列中所有 token 的对数概率相加，得到整个序列的对数概率
        return torch.sum(action_log_probs, dim=1)