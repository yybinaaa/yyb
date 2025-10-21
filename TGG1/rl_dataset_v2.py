import torch
import numpy as np
import random
from dataset_v2 import OneRecV2Dataset

class PreferenceDataset(OneRecV2Dataset):
    """
    [全新] 为基于偏好的强化学习(类DPO)准备数据。
    生成 (上下文, 正样本, 负样本) 三元组。
    """
    def __init__(self, data_dir, args, codebook_size=8192):
        super().__init__(data_dir, args, codebook_size)
        
        # 创建一个所有物品 reid 的集合，方便负采样
        self.all_item_reids = set(range(1, self.itemnum + 1))
        
        # 预处理，为每个用户创建一个交互过的物品集合
        print("Preprocessing user interaction history for negative sampling...")
        self.user_interacted_items = {}
        for uid in range(len(self.seq_offsets)):
            user_sequence = self._load_user_data(uid)
            interacted_items = {
                rec[1] for rec in user_sequence if rec and len(rec) > 1 and rec[1] is not None
            }
            self.user_interacted_items[uid] = interacted_items
        
        print("RL Preference Dataset Initialized.")

    def __getitem__(self, uid):
        user_sequence = self._load_user_data(uid)
        item_interactions = [rec for rec in user_sequence if rec[1] is not None and rec[1] != 0]

        if len(item_interactions) < 2:
            return None

        # [核心逻辑] 最后一次交互是正样本，之前的是上下文
        history_records = item_interactions[:-1]
        positive_record = item_interactions[-1]

        # 1. 准备 State (上下文)
        # 这部分逻辑与监督学习时完全一致
        u_feat_record = user_sequence[0][2]
        user_sparse_context = []
        if isinstance(u_feat_record, dict):
            for feat_name in self.USER_SPARSE_FEAT.keys():
                user_sparse_context.append(u_feat_record.get(feat_name, 0))
        else:
            user_sparse_context = [0] * len(self.USER_SPARSE_FEAT)

        history_item_ids = np.full(self.maxlen, self.PAD_TOKEN_ID, dtype=np.int64)
        actual_history = history_records[-self.maxlen:]
        start_idx = self.maxlen - len(actual_history)
        for i, record in enumerate(actual_history):
            item_id = self._validate_id(record[1], self.itemnum + 1)
            history_item_ids[start_idx + i] = item_id
            # (为了简化，我们暂时只用 item_id 作为历史，省略历史物品的稀疏特征)
        
        # 2. 准备正样本 (Positive Action)
        positive_item_id = self._validate_id(positive_record[1], self.itemnum + 1)
        positive_semantic_ids = self.semantic_map.get(positive_item_id, [self.PAD_TOKEN_ID] * 3)
        
        # 3. 准备负样本 (Negative Action)
        user_seen_items = self.user_interacted_items[uid]
        possible_negatives = list(self.all_item_reids - user_seen_items)
        if not possible_negatives: # 如果用户看完了所有物品...
            negative_item_id = random.randint(1, self.itemnum) # 随机选一个
        else:
            negative_item_id = random.choice(possible_negatives)
        
        negative_semantic_ids = self.semantic_map.get(negative_item_id, [self.PAD_TOKEN_ID] * 3)

        return {
            "user_sparse": np.array(user_sparse_context, dtype=np.int64),
            "history_item_ids": history_item_ids,
            "positive_action_ids": np.array(positive_semantic_ids, dtype=np.int64),
            "negative_action_ids": np.array(negative_semantic_ids, dtype=np.int64)
        }
        
    @staticmethod
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if not batch: return None
        elem = batch[0]
        batch_dict = {key: [d[key] for d in batch] for key in elem}
        return {key: torch.from_numpy(np.stack(values)) for key, values in batch_dict.items()}