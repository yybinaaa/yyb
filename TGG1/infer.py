import argparse
import json
import os
import struct
from pathlib import Path
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset08281 import MyTestDataset, save_emb
from model08281 import BaselineModel

# --- 1. 候选物品数据集 (已修改，增加时间特征) ---

class CandidateItemDataset(Dataset):
    """
    专门用于加载和预处理候选物品（predict_set.jsonl）的数据集。
    """
    def __init__(self, data_dir, test_dataset_ref):
        self.candidate_path = Path(data_dir, 'predict_set.jsonl')
        self.lines = self.candidate_path.read_text().strip().split('\n')
        
        self.indexer_i = test_dataset_ref.indexer['i']
        self.feature_types = test_dataset_ref.feature_types
        self.feature_default_value = test_dataset_ref.feature_default_value
        self.mm_emb_dict = test_dataset_ref.mm_emb_dict
        self.ITEM_SPARSE_FEAT = test_dataset_ref.ITEM_SPARSE_FEAT
        self.ITEM_EMB_FEAT = test_dataset_ref.ITEM_EMB_FEAT
        self.indexer_i_rev = test_dataset_ref.indexer_i_rev

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = json.loads(self.lines[idx])
        
        creative_id = line['creative_id']
        retrieval_id = line['retrieval_id']
        item_id = self.indexer_i.get(creative_id, 0)
        
        feat = self.fill_missing_feat(line.get('features', {}), item_id)
        
        sparse_feats = torch.tensor([feat[str(fid)] for fid in self.ITEM_SPARSE_FEAT], dtype=torch.long)
        
        mm_feats = {}
        for fid, dim in self.ITEM_EMB_FEAT.items():
            mm_np = np.array(feat.get(str(fid), np.zeros(dim)), dtype=np.float32)
            mm_feats[str(fid)] = torch.from_numpy(mm_np)

        # [核心修改] 为单个候选物品创建一个虚拟的时间特征张量
        # 形状为 [sequence_length, num_time_features]，这里是 [1, 4]
        # 使用全零表示这是一个独立的物品，没有时间上下文。
        time_feats = torch.zeros((1, 4), dtype=torch.long)
        
        # 将物品视为长度为1的序列
        seq_ids = torch.tensor([item_id], dtype=torch.long)
        
        return seq_ids, sparse_feats.unsqueeze(0), {k: v.unsqueeze(0) for k, v in mm_feats.items()}, time_feats, retrieval_id

    def fill_missing_feat(self, feat, item_id):
        if feat is None: feat = {}
        filled_feat = feat.copy()
        
        all_feat_ids = []
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)
        
        missing_fields = set(all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:
            if feat_id in self.feature_default_value:
                filled_feat[feat_id] = self.feature_default_value[feat_id]

        for feat_id in self.feature_types['item_emb']:
            if item_id != 0 and self.indexer_i_rev.get(item_id) in self.mm_emb_dict.get(feat_id, {}):
                filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]

        return filled_feat

def candidate_collate_fn(batch):
    """为CandidateItemDataset定制的collate_fn (已修改)"""
    seq_ids, sparse_feats_list, mm_feats_list, time_feats_list, retrieval_ids = zip(*batch)
    
    seq_ids = torch.cat(seq_ids, dim=0) # 使用 cat 因为 seq_ids 已经是 [1] 的形状
    sparse_feats = torch.cat(sparse_feats_list, dim=0)
    time_feats = torch.cat(time_feats_list, dim=0)
    
    mm_batch = {}
    if mm_feats_list and mm_feats_list[0]:
        for fid in mm_feats_list[0].keys():
            mm_batch[fid] = torch.cat([x[fid] for x in mm_feats_list], dim=0)
            
    return seq_ids, sparse_feats, mm_batch, time_feats, list(retrieval_ids)


# --- 2. 辅助函数 (保持不变) ---

def get_ckpt_path():
    ckpt_path_env = os.environ.get("MODEL_OUTPUT_PATH")
    if not ckpt_path_env:
        raise ValueError("环境变量 MODEL_OUTPUT_PATH 未设置")
    
    pt_files = list(Path(ckpt_path_env).glob("model.pt")) # 假设模型都叫 model.pt
    if not pt_files:
        raise FileNotFoundError(f"在目录 {ckpt_path_env} 中未找到任何 model.pt 文件")
    
    latest_file = max(pt_files, key=lambda p: p.parent.stat().st_mtime) # 按目录时间排序
    print(f"找到并使用最新的模型文件: {latest_file}")
    return latest_file

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=512, type=int)
    # 模型的参数定义需要和训练时严格一致
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--hidden_units', default=64, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=2, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str)
    # 添加训练时引入的新参数，确保模型结构一致
    parser.add_argument('--temp', default=0.8, type=float)
    args = parser.parse_args()
    return args

def read_result_ids(file_path):
    with open(file_path, 'rb') as f:
        num_points_query, query_ann_top_k = struct.unpack('II', f.read(8))
        print(f"从检索结果读取: {num_points_query} 个查询, 每个返回 Top {query_ann_top_k} 个结果")
        num_result_ids = num_points_query * query_ann_top_k
        result_ids = np.fromfile(f, dtype=np.uint64, count=num_result_ids)
        return result_ids.reshape((num_points_query, query_ann_top_k))

# --- 3. 主推理流程 ---

def main():
    args = get_args()
    data_path = os.environ.get('EVAL_DATA_PATH', 'data')
    result_path = Path(os.environ.get('EVAL_RESULT_PATH', 'results'))
    result_path.mkdir(parents=True, exist_ok=True)

    # --- Step 1: 加载模型和数据集 ---
    print("## 1. 加载模型和数据 ##")
    test_dataset = MyTestDataset(data_path, args)
    # [核心修改] 使用 MyTestDataset 自带的 collate_fn，它能正确处理数据批处理
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=MyTestDataset.collate_fn
    )
    
    candidate_dataset = CandidateItemDataset(data_path, test_dataset)
    candidate_loader = DataLoader(
        candidate_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0, 
        collate_fn=candidate_collate_fn
    )

    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    
    ckpt_path = get_ckpt_path()
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(args.device)))
    model.eval() # 切换到评估模式

    # --- Step 2: 生成用户Query向量 (已重写) ---
    print("\n## 2. 生成用户Query向量 ##")
    all_query_embs = []
    all_user_ids = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating User Query Embeddings"):
            # MyTestDataset.collate_fn 返回的是元组，直接解包
            seqs, token_types, seq_feats_list, time_feats, user_ids = batch
            
            seqs = seqs.to(args.device)
            time_feats = time_feats.to(args.device)

            # 将 list of feature dicts 转换为 tensor dict
            sparse_feats = torch.tensor(
                [[feat[str(fid)] for fid in model.ITEM_SPARSE_FEAT.keys()] for feat in seq_feats_list],
                dtype=torch.long
            ).to(args.device)

            mm_batch = {}
            if model.ITEM_EMB_FEAT:
                for fid, dim in model.ITEM_EMB_FEAT.items():
                    mm_np = np.array([f.get(str(fid), np.zeros(dim)) for f in seq_feats_list], dtype=np.float32)
                    mm_batch[str(fid)] = torch.from_numpy(mm_np).to(args.device)
            
            # [核心修改] 调用 predict 时传入 time_feats
            embs = model.predict(seqs, sparse_feats, mm_batch, time_feats)
            embs = F.normalize(embs, p=2, dim=1)

            all_query_embs.append(embs.cpu().numpy())
            all_user_ids.extend(user_ids)

    query_embs = np.concatenate(all_query_embs, axis=0)
    save_emb(query_embs, result_path / 'query.fbin')
    print(f"Query向量生成完毕，形状: {query_embs.shape}")

    # --- Step 3: 生成物品Candidate向量 (已重写) ---
    print("\n## 3. 生成物品Candidate向量 ##")
    all_candidate_embs = []
    all_retrieval_ids = []
    with torch.no_grad():
        for batch in tqdm(candidate_loader, desc="Generating Candidate Embeddings"):
            seq_ids, sparse_feats, mm_batch, time_feats, retrieval_ids = batch
            
            seq_ids = seq_ids.to(args.device)
            sparse_feats = sparse_feats.to(args.device)
            mm_batch = {k: v.to(args.device) for k, v in mm_batch.items()}
            time_feats = time_feats.to(args.device)

            # [核心修改] 调用 predict 时传入虚拟的 time_feats
            # 模型内部的 encode_sequence 会处理长度为1的序列
            embs = model.predict(seq_ids, sparse_feats, mm_batch, time_feats)
            embs = F.normalize(embs, p=2, dim=1)

            all_candidate_embs.append(embs.cpu().numpy())
            all_retrieval_ids.extend(retrieval_ids)

    candidate_embs = np.concatenate(all_candidate_embs, axis=0)
    save_emb(candidate_embs, result_path / 'embedding.fbin')
    save_emb(np.array(all_retrieval_ids, dtype=np.uint64).reshape(-1, 1), result_path / 'id.u64bin')
    print(f"Candidate向量生成完毕，形状: {candidate_embs.shape}")

    # --- Step 4 & 5: ANN检索和结果处理 (保持不变) ---
    print("\n## 4. 执行ANN检索 ##")
    faiss_cmd = [
        "/workspace/faiss-based-ann/faiss_demo",
        "--dataset_vector_file_path", str(result_path / "embedding.fbin"),
        "--dataset_id_file_path", str(result_path / "id.u64bin"),
        "--query_vector_file_path", str(result_path / "query.fbin"),
        "--result_id_file_path", str(result_path / "result_ids.u64bin"),
        "--query_ann_top_k", "10",
        "--faiss_metric_type", "1"
    ]
    print(f"执行命令: {' '.join(faiss_cmd)}")
    subprocess.run(faiss_cmd, check=True)

    print("\n## 5. 处理检索结果 ##")
    retrieve_id2creative_id = {
        line['retrieval_id']: line['creative_id'] 
        for line in [json.loads(l) for l in Path(data_path, 'predict_set.jsonl').read_text().strip().split('\n')]
    }
    top10_retrieved_ids = read_result_ids(result_path / "result_ids.u64bin")
    
    final_results = []
    for user_id, retrieved_ids in zip(all_user_ids, top10_retrieved_ids):
        creative_ids = [retrieve_id2creative_id.get(int(rid), 0) for rid in retrieved_ids]
        final_results.append({
            "user_id": user_id,
            "creative_id_list": creative_ids
        })

    print("推理流程完成！")
    return final_results

if __name__ == '__main__':
    required_envs = ['EVAL_DATA_PATH', 'MODEL_OUTPUT_PATH', 'EVAL_RESULT_PATH']
    for env in required_envs:
        if env not in os.environ:
            if env == 'EVAL_DATA_PATH': os.environ[env] = 'data'
            elif env == 'MODEL_OUTPUT_PATH': os.environ[env] = 'checkpoints'
            elif env == 'EVAL_RESULT_PATH': os.environ[env] = 'results'
            print(f"警告：环境变量 {env} 未设置，使用默认值: {os.environ[env]}")

    main()