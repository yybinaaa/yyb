import torch
import numpy as np
import random
from pathlib import Path
import json
import pickle

class OneRecV2Dataset(torch.utils.data.Dataset):
    """
    [最终升级版] 为 OneRec-V2 模型准备多-token语义ID的生成式训练样本。
    """
    def __init__(self, data_dir, args, codebook_size=8192): # [升级] 接收 codebook_size
        super().__init__()
        self.data_dir = Path(data_dir)
        self.maxlen = args.maxlen
        self.data_path_str = str(self.data_dir / "seq.jsonl")

        print("Loading indexer...")
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
            self.indexer = indexer
            self.feat_max_id = {fid: len(vocab) for fid, vocab in self.indexer['f'].items()}
        
        # [升级] 特殊 Token ID 现在基于 codebook_size
        self.CODEBOOK_SIZE = codebook_size
        self.BOS_TOKEN_ID = self.CODEBOOK_SIZE
        self.PAD_TOKEN_ID = 0 # 假设 codebook id 从 1 开始, 0 作为 padding
        self.VOCAB_SIZE = self.CODEBOOK_SIZE + 1 # [BOS]
        
        # [升级] 加载我们生成的语义ID映射文件
        print("Loading semantic ID map...")
        with open('semantic_id_map.json', 'r') as f:
            # json加载后key是字符串，需要转回int
            self.semantic_map = {int(k): v for k, v in json.load(f).items()}

        print("Loading feature info...")
        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()
        self.USER_SPARSE_FEAT = {k: self.feat_statistics[k] for k in self.feature_types['user_sparse']}
        self.ITEM_SPARSE_FEAT = {k: self.feat_statistics[k] for k in self.feature_types['item_sparse']}
        
        print("Loading data offsets...")
        with open(self.data_dir / 'seq_offsets.pkl', 'rb') as f:
            self.seq_offsets = pickle.load(f)
        
        self.data_file = None
        print(f"Dataset initialized. Total users: {len(self.seq_offsets)}")
        
    # _init_feat_info, _load_user_data, _validate_id 方法保持不变
    def _init_feat_info(self):
        feat_default_value, feat_statistics, feat_types = {}, {}, {}
        feat_types['user_sparse'] = ['103', '104', '105', '109']
        feat_types['item_sparse'] = ['100', '117', '111', '118', '101', '102', '119', '120', '114', '112', '121', '115', '122', '116']
        for feat_id in feat_types['user_sparse'] + feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            if feat_id in self.indexer['f']:
                feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
            else:
                feat_statistics[feat_id] = 1
        return feat_default_value, feat_types, feat_statistics

    def _init_worker_state(self):
        self.data_file = open(self.data_path_str, 'rb')

    def _load_user_data(self, uid):
        if self.data_file is None:
            self._init_worker_state()
        try:
            self.data_file.seek(self.seq_offsets[uid])
            line = self.data_file.readline()
            return json.loads(line)
        except Exception:
            return []

    def _validate_id(self, value, max_valid_id):
        if not isinstance(value, int) or value >= max_valid_id:
            return 0
        return value

    def __len__(self):
        return len(self.seq_offsets)

    def __getitem__(self, uid):
        # 上下文处理部分保持不变
        user_sequence = self._load_user_data(uid)
        item_interactions = [rec for rec in user_sequence if rec[1] is not None and rec[1] != 0]
        if len(item_interactions) < 2:
            return None

        split_point = random.randint(1, len(item_interactions) - 1)
        history_records = item_interactions[:split_point]
        target_record = item_interactions[split_point]

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
        
        # [核心升级] 准备多-token的目标序列
        target_item_id = self._validate_id(target_record[1], self.itemnum + 1)
        
        # 从 map 中查找语义ID，如果找不到，用一个默认的 padding 序列
        semantic_ids = self.semantic_map.get(target_item_id, [self.PAD_TOKEN_ID] * 3)
        
        # 构建 decoder 的输入和标签
        # 例如: semantic_ids = [s1, s2, s3]
        # decoder_input_ids -> [BOS, s1, s2]
        # decoder_labels    -> [s1,  s2, s3]
        decoder_input_ids = np.array([self.BOS_TOKEN_ID] + semantic_ids[:-1], dtype=np.int64)
        decoder_labels = np.array(semantic_ids, dtype=np.int64)

        return {
            "user_sparse": np.array(user_sparse_context, dtype=np.int64),
            "history_item_ids": history_item_ids,
            "history_item_sparse": history_item_sparse,
            "decoder_input_ids": decoder_input_ids,
            "decoder_labels": decoder_labels
        }

    # collate_fn 保持不变
    @staticmethod
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if not batch: return None
        elem = batch[0]
        batch_dict = {key: [d[key] for d in batch] for key in elem}
        return {key: torch.from_numpy(np.stack(values)) for key, values in batch_dict.items()}