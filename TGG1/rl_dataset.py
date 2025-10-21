import torch
import numpy as np
import random
from pathlib import Path
import json
import pickle

# 我们需要复用之前 Dataset 的一些功能
from dataset_v2 import OneRecV2Dataset

class RLDataset(OneRecV2Dataset):
    """
    为 OneRec-V2 的强化学习阶段准备数据。
    """
    def __init__(self, data_dir, args, codebook_size=8192):
        # 复用父类的初始化
        super().__init__(data_dir, args, codebook_size)
        print("RL Dataset Initialized.")

    def __getitem__(self, uid):
        user_sequence = self._load_user_data(uid)
        
        # [核心区别] 这里的交互记录需要包含 action 类型
        # 我们假设 record[4] 是 action_type, record[1] 是 item_id
        # 1: impression, 2: click, 3: add_to_cart, 4: purchase, -1: negative
        item_interactions = [rec for rec in user_sequence if rec[1] is not None and rec[1] != 0]

        if len(item_interactions) < 2:
            return None

        # 随机选择一个交互作为 (state, action, reward_info) 样本
        split_point = random.randint(1, len(item_interactions) - 1)
        history_records = item_interactions[:split_point]
        target_record = item_interactions[split_point]

        # 1. 准备 State (Observation), 与之前完全相同
        u_feat_record = user_sequence[0][2]
        user_sparse_context = []
        if isinstance(u_feat_record, dict):
            for feat_name in self.USER_SPARSE_FEAT.keys():
                user_sparse_context.append(u_feat_record.get(feat_name, 0))
        else:
            user_sparse_context = [0] * len(self.USER_SPARSE_FEAT)

        history_item_ids = np.full(self.maxlen, self.PAD_TOKEN_ID, dtype=np.int64)
        history_item_sparse = np.full((self.maxlen, len(self.ITEM_SPARSE_FEAT)), self.PAD_TOKEN_ID, dtype=np.int64)
        actual_history = history_records[-self.maxlen:]
        start_idx = self.maxlen - len(actual_history)
        for i, record in enumerate(actual_history):
            item_id = self._validate_id(record[1], self.itemnum + 1)
            history_item_ids[start_idx + i] = item_id
            i_feat_record = record[3]
            if isinstance(i_feat_record, dict):
                for j, feat_name in enumerate(self.ITEM_SPARSE_FEAT.keys()):
                    history_item_sparse[start_idx + i, j] = i_feat_record.get(feat_name, 0)

        # 2. 准备 Action
        target_item_id = self._validate_id(target_record[1], self.itemnum + 1)
        action_semantic_ids = self.semantic_map.get(target_item_id, [self.PAD_TOKEN_ID] * 3)

        # 3. 准备 Reward Info
        # [核心] 我们只传递 action_type, 具体的奖励值在主循环中计算
        action_type = target_record[4] if len(target_record) > 4 else 1 # 默认为 impression
        if action_type is None:
           action_type = 1

        return {
            # State / Observation
            "user_sparse": np.array(user_sparse_context, dtype=np.int64),
            "history_item_ids": history_item_ids,
            "history_item_sparse": history_item_sparse,
            # Action
            "action_ids": np.array(action_semantic_ids, dtype=np.int64),
            # Reward Info
            "action_type": np.array(action_type, dtype=np.int64)
        }

    @staticmethod
    def collate_fn(batch):
        # Collate fn 与之前基本一致
        batch = [b for b in batch if b is not None]
        if not batch: return None
        elem = batch[0]
        batch_dict = {key: [d[key] for d in batch] for key in elem}
        return {key: torch.from_numpy(np.stack(values)) for key, values in batch_dict.items()}