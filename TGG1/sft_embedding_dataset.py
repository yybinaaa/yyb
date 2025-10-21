# file: sft_embedding_dataset.py
import torch
import numpy as np
import random
from pathlib import Path
import json
import pickle

class SFTEmbeddingDataset(torch.utils.data.Dataset):
    """
    一个极简化的数据集，只用于 SFT 预训练阶段，目标是 item_reid。
    它不读取 semantic_id_map.json。
    """
    def __init__(self, data_dir, args):
        self.data_dir = Path(data_dir)
        self.maxlen = args.maxlen
        with open(self.data_dir / 'indexer.pkl', 'rb') as f:
            indexer = pickle.load(f)
            self.itemnum = len(indexer['i'])
            self.indexer = indexer
        
        self.BOS_TOKEN_ID = self.itemnum + 1
        self.PAD_TOKEN_ID = 0
        self.VOCAB_SIZE = self.itemnum + 2 # 使用 item_reid 作为词汇表

        # 加载必要的文件
        self.data_file = open(self.data_dir / "seq.jsonl", 'rb')
        with open(self.data_dir / 'seq_offsets.pkl', 'rb') as f:
            self.seq_offsets = pickle.load(f)
        
    def _load_user_data(self, uid):
        # ... (和之前一样)
        try:
            self.data_file.seek(self.seq_offsets[uid])
            return json.loads(self.data_file.readline())
        except Exception:
            return []

    def __len__(self):
        return len(self.seq_offsets)
        
    def __getitem__(self, uid):
        user_sequence = self._load_user_data(uid)
        item_interactions = [rec for rec in user_sequence if rec and len(rec) > 1 and rec[1] is not None]
        if len(item_interactions) < 2: return None

        split_point = random.randint(1, len(item_interactions) - 1)
        history_records = item_interactions[:split_point]
        target_record = item_interactions[split_point]
        
        # 只准备最基本的上下文：历史 item_id
        history_item_ids = np.full(self.maxlen, self.PAD_TOKEN_ID, dtype=np.int64)
        actual_history = history_records[-self.maxlen:]
        start_idx = self.maxlen - len(actual_history)
        for i, record in enumerate(actual_history):
            history_item_ids[start_idx + i] = record[1] if record[1] else 0

        # 目标是 item_reid
        target_item_id = target_record[1] if target_record[1] else 0
        decoder_input_ids = np.array([self.BOS_TOKEN_ID], dtype=np.int64)
        decoder_labels = np.array([target_item_id], dtype=np.int64)
        
        # 返回一个简化的、不含稀疏特征的上下文
        return {
            "history_item_ids": history_item_ids,
            "decoder_input_ids": decoder_input_ids,
            "decoder_labels": decoder_labels
        }

    @staticmethod
    def collate_fn(batch):
        # ... (和之前一样)
        batch = [b for b in batch if b is not None]
        if not batch: return None
        elem = batch[0]
        batch_dict = {key: [d[key] for d in batch] for key in elem}
        return {key: torch.from_numpy(np.stack(values)) for key, values in batch_dict.items()}