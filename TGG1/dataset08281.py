import json
import pickle
import struct
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# def save_emb(emb, save_path):
#     """
#     将Embedding保存为二进制文件
#     """
#     num_points, num_dimensions = emb.shape
#     print(f'正在保存 {save_path}')
#     with open(Path(save_path), 'wb') as f:
#         f.write(struct.pack('II', num_points, num_dimensions))
#         emb.tofile(f)


# def load_mm_emb(mm_path, feat_ids):
#     """
#     加载多模态特征Embedding
#     """
#     SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
#     mm_emb_dict = {}
#     for feat_id in tqdm(feat_ids, desc='加载多模态特征'):
#         shape = SHAPE_DICT[feat_id]
#         emb_dict = {}
#         if feat_id == '81':
#             with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
#                 emb_dict = pickle.load(f)
#         else:
#             try:
#                 base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
#                 for json_file in base_path.glob('*.json'):
#                     with open(json_file, 'r', encoding='utf-8') as file:
#                         for line in file:
#                             data = json.loads(line.strip())
#                             emb = np.array(data['emb'], dtype=np.float32) if isinstance(data['emb'], list) else data['emb']
#                             emb_dict[data['anonymous_cid']] = emb
#             except Exception as e:
#                 print(f"转换错误: {e}")
#         mm_emb_dict[feat_id] = emb_dict
#         print(f'已加载 #{feat_id} 多模态特征')
#     return mm_emb_dict




# class MyDataset(torch.utils.data.Dataset):
#     # 新的 __init__ 方法
#     def __init__(self, data_dir, args):
#         super().__init__()
#         self.data_dir = Path(data_dir)
#         self.maxlen = args.maxlen
#         self.mm_emb_ids = args.mm_emb_id

#         # <<< 核心修改 1: 加载 preprocess.py 生成的三个核心文件 >>>
#         print("💾 正在加载预处理文件 (indexer, user_features, item_features)...")
#         with open(self.data_dir / 'indexer.pkl', 'rb') as f:
#             self.indexer = pickle.load(f)
#         with open(self.data_dir / 'user_features.pkl', 'rb') as f:
#             self.user_features = pickle.load(f)
#         with open(self.data_dir / 'item_features.pkl', 'rb') as f:
#             self.item_features = pickle.load(f)
#         print("✅ 预处理文件加载完毕!")

#         # <<< 核心修改 2: 直接从新的 indexer 获取所有元信息 >>>
#         self.usernum = len(self.indexer['u'])
#         self.itemnum = len(self.indexer['i'])
#         # 创建一个从 物品原始ID -> 物品整数索引 的反向映射，方便后面查找
#         self.item_original_id_to_idx = self.indexer['i']

#         # <<< 核心修改 3: 硬编码特征键，并从 indexer 获取词典大小 >>>
#         # 这里的顺序必须和 preprocess.py 中定义的完全一致
#         self.USER_SPARSE_FEAT_KEYS = ['103', '104', '105', '109']
#         self.ITEM_SPARSE_FEAT_KEYS = ['100', '117', '111', '118', '101', '102', '119', '120', '114', '112', '121', '115', '122', '116']
        
#         # 这个信息将传递给模型，告诉模型每个特征的 Embedding 层应该建多大
#         self.feat_statistics = {
#             feat_id: len(feat_map) for feat_id, feat_map in self.indexer['f'].items()
#         }
#         self.feature_types = {
#             'user_sparse': self.USER_SPARSE_FEAT_KEYS,
#             'item_sparse': self.ITEM_SPARSE_FEAT_KEYS,
#             'item_emb': self.mm_emb_ids,
#         }

#         # <<< 核心修改 4: 多模态特征部分保持不变 >>>
#         EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
#         self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in self.feature_types['item_emb']}

#         # 这个方法加载原始序列文件，保持不变
#         self._load_data_and_offsets()

#     def _load_data_and_offsets(self):
#         self.data_file = open(self.data_dir / "seq.jsonl", 'rb')
#         with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
#             self.seq_offsets = pickle.load(f)

#     def _load_user_data(self, uid):
#         self.data_file.seek(self.seq_offsets[uid])
#         line = self.data_file.readline()
#         try:
#             return json.loads(line)
#         except (json.JSONDecodeError, Exception):
#             return []

#     def __len__(self):
#         return len(self.seq_offsets)

#         # 新的 __getitem__ 方法
#     def __getitem__(self, uid):
#         # uid 是 dataloader 传过来的用户整数索引
        
#         # 1. <<< 获取用户静态特征 (极快) >>>
#         # 直接从内存中的字典里，用整数索引 uid 拿到预处理好的特征
#         user_static_feat_indexed = self.user_features.get(uid, {})
        
#         # 2. <<< 读取原始序列数据 >>>
#         # 注意：_load_user_data 需要 uid 的原始 ID，我们需要一个反向映射
#         # 为了简化，我们假设 dataloader 的 index 和我们的 uid 是一致的。
#         # 如果不一致，你需要建立一个 uid_idx -> original_uid 的映射。
#         # 暂时我们先假设传进来的 uid 就是文件中的用户行号索引。
#         user_sequence = self._load_user_data(uid)
        
#         item_interactions = []
#         user_full_history_items = set()
#         for u, i, user_feat, item_feat, action_type, ts in user_sequence:
#             if i and item_feat:
#                 # 我们只需要物品的原始ID和行为类型
#                 item_interactions.append({'id': i, 'action': action_type})
#                 # 同时，将物品的整数索引加入历史记录，用于负采样排除
#                 item_idx = self.item_original_id_to_idx.get(i)
#                 if item_idx is not None:
#                     user_full_history_items.add(item_idx)

#         full_history_tensor = torch.tensor(list(user_full_history_items), dtype=torch.long)
        
#         # 3. <<< 准备空的 Numpy 数组用于填充 >>>
#         # 物品ID序列 (存的是整数索引)
#         seq_ids = np.zeros(self.maxlen, dtype=np.int32)
#         pos_ids = np.zeros(self.maxlen, dtype=np.int32)
#         # 行为序列
#         action_seq = np.zeros(self.maxlen, dtype=np.int32)
#         # 物品稀疏特征序列 (存的是特征值的整数索引)
#         seq_sparse_feat = np.zeros((self.maxlen, len(self.ITEM_SPARSE_FEAT_KEYS)), dtype=np.int32)
#         pos_sparse_feat = np.zeros((self.maxlen, len(self.ITEM_SPARSE_FEAT_KEYS)), dtype=np.int32)
#         # 物品多模态特征序列 (存的是原始 embedding)
#         seq_mm_feat = {fid: np.zeros((self.maxlen, dim), dtype=np.float32) for fid, dim in self.ITEM_EMB_FEAT.items()}
#         pos_mm_feat = {fid: np.zeros((self.maxlen, dim), dtype=np.float32) for fid, dim in self.ITEM_EMB_FEAT.items()}

#         if not item_interactions:
#             # 如果用户没有交互，直接返回空的和静态特征
#             return self._package_tensors(seq_ids, pos_ids, action_seq, user_static_feat_indexed, 
#                                         seq_sparse_feat, pos_sparse_feat, seq_mm_feat, pos_mm_feat, 
#                                         full_history_tensor)

#         history = item_interactions[:-1]
#         targets = item_interactions[1:]
        
#         start_idx = max(0, len(history) - self.maxlen)
#         history = history[start_idx:]
#         targets = targets[start_idx:]
        
#         end_idx = len(history)
        
#         # 4. <<< 循环填充序列 (核心逻辑) >>>
#         for i in range(end_idx):
#             # --- 处理历史序列 ---
#             hist_item = history[i]
#             # 获取物品的整数索引
#             hist_item_idx = self.item_original_id_to_idx.get(hist_item['id'])
#             if hist_item_idx is not None:
#                 seq_ids[i] = hist_item_idx
#                 action_seq[i] = hist_item['action'] if hist_item['action'] is not None else 0
                
#                 # 从内存中获取该物品的预处理特征 (极快)
#                 item_features_indexed = self.item_features.get(hist_item_idx, {})
#                 # 填充稀疏特征
#                 for feat_idx, feat_key in enumerate(self.ITEM_SPARSE_FEAT_KEYS):
#                     seq_sparse_feat[i, feat_idx] = item_features_indexed.get(feat_key, 0)
#                 # 填充多模态特征
#                 for fid, dim in self.ITEM_EMB_FEAT.items():
#                     v = item_features_indexed.get(fid)
#                     if v is not None:
#                         a = np.asarray(v, dtype=np.float32).flatten()
#                         actual_len = min(len(a), dim)
#                         seq_mm_feat[fid][i, :actual_len] = a[:actual_len]

#             # --- 处理目标序列 (逻辑同上) ---
#             target_item = targets[i]
#             target_item_idx = self.item_original_id_to_idx.get(target_item['id'])
#             if target_item_idx is not None:
#                 pos_ids[i] = target_item_idx
                
#                 item_features_indexed = self.item_features.get(target_item_idx, {})
#                 for feat_idx, feat_key in enumerate(self.ITEM_SPARSE_FEAT_KEYS):
#                     pos_sparse_feat[i, feat_idx] = item_features_indexed.get(feat_key, 0)
#                 for fid, dim in self.ITEM_EMB_FEAT.items():
#                     v = item_features_indexed.get(fid)
#                     if v is not None:
#                         a = np.asarray(v, dtype=np.float32).flatten()
#                         actual_len = min(len(a), dim)
#                         pos_mm_feat[fid][i, :actual_len] = a[:actual_len]
                
#         return self._package_tensors(seq_ids, pos_ids, action_seq, user_static_feat_indexed,
#                                     seq_sparse_feat, pos_sparse_feat, seq_mm_feat, pos_mm_feat,
#                                     full_history_tensor)
#     # 新增一个辅助方法，用于将 Numpy 数组打包成 Torch 张量
#     def _package_tensors(self, seq_ids, pos_ids, action_seq, user_static_feat_indexed,
#                         seq_sparse_feat, pos_sparse_feat, seq_mm_feat, pos_mm_feat,
#                         full_history_tensor):
        
#         # 用户静态特征
#         user_sparse_tensor = torch.tensor(
#             [user_static_feat_indexed.get(k, 0) for k in self.USER_SPARSE_FEAT_KEYS], 
#             dtype=torch.long
#         )

#         # 序列相关特征
#         seq_ids_tensor = torch.tensor(seq_ids, dtype=torch.long)
#         pos_ids_tensor = torch.tensor(pos_ids, dtype=torch.long)
#         action_seq_tensor = torch.tensor(action_seq, dtype=torch.long)
#         seq_sparse_tensor = torch.tensor(seq_sparse_feat, dtype=torch.long)
#         pos_sparse_tensor = torch.tensor(pos_sparse_feat, dtype=torch.long)
        
#         seq_mm_tensors = {fid: torch.from_numpy(arr) for fid, arr in seq_mm_feat.items()}
#         pos_mm_tensors = {fid: torch.from_numpy(arr) for fid, arr in pos_mm_feat.items()}

#         return (seq_ids_tensor, pos_ids_tensor, action_seq_tensor, 
#                 user_sparse_tensor, seq_sparse_tensor, pos_sparse_tensor, 
#                 seq_mm_tensors, pos_mm_tensors, full_history_tensor)
#     # def _features_to_tensors(self, seq_ids, pos_ids, action_seq, seq_feat_list, pos_feat_list):
#     #     seq_ids_tensor = torch.tensor(seq_ids, dtype=torch.long)
#     #     pos_ids_tensor = torch.tensor(pos_ids, dtype=torch.long)
#     #     action_seq_tensor = torch.tensor(action_seq, dtype=torch.long)
#     #     seq_sparse_tensor = torch.tensor([[feat.get(str(fid), 0) for fid in self.ITEM_SPARSE_FEAT_KEYS] for feat in seq_feat_list], dtype=torch.long)
#     #     pos_sparse_tensor = torch.tensor([[feat.get(str(fid), 0) for fid in self.ITEM_SPARSE_FEAT_KEYS] for feat in pos_feat_list], dtype=torch.long)
#     #     seq_mm_tensors, pos_mm_tensors = {}, {}
#     #     for fid, dim in self.ITEM_EMB_FEAT.items():
#     #         seq_mm_tensors[fid] = torch.from_numpy(self._extract_mm_feat(seq_feat_list, fid, dim))
#     #         pos_mm_tensors[fid] = torch.from_numpy(self._extract_mm_feat(pos_feat_list, fid, dim))
#     #     return seq_ids_tensor, pos_ids_tensor, action_seq_tensor, seq_sparse_tensor, pos_sparse_tensor, seq_mm_tensors, pos_mm_tensors
    
#     def _extract_mm_feat(self, feat_list, fid, dim):
#         mm_np = np.zeros((len(feat_list), dim), dtype=np.float32)
#         for t, f in enumerate(feat_list):
#             v = f.get(str(fid))
#             if v is not None:
#                 a = np.asarray(v, dtype=np.float32).flatten()
#                 actual_len = min(len(a), dim)
#                 mm_np[t, :actual_len] = a[:actual_len]
#         return mm_np

#     def _init_feat_info(self):
#         feat_default_value = {}
#         feat_statistics = {}
#         feat_types = {
#             'user_sparse': ['103', '104', '105', '109'],
#             'item_sparse': ['100', '117', '111', '118', '101', '102', '119', '120', '114', '112', '121', '115', '122', '116'],
#             'item_emb': self.mm_emb_ids
#         }
#         all_sparse_ids = set(feat_types['user_sparse']) | set(feat_types['item_sparse'])
#         for feat_id in all_sparse_ids:
#             feat_default_value[feat_id] = 0
#             feat_statistics[feat_id] = len(self.indexer['f'].get(feat_id, {}))
#         return feat_default_value, feat_types, feat_statistics

#     def fill_missing_feat(self, feat, item_id):
#         if feat is None: feat = {}
#         filled_feat = feat.copy()
#         for k in self.ITEM_SPARSE_FEAT_KEYS:
#             if k not in filled_feat:
#                 filled_feat[k] = self.feature_default_value.get(k, 0)
#         return filled_feat

#     @staticmethod
#     # def collate_fn(batch):
#     #     (seq_ids, pos_ids, action_seqs, 
#     #      seq_sparse, pos_sparse, 
#     #      seq_mm, pos_mm, full_histories) = zip(*batch)

#     #     seq_ids, pos_ids, action_seqs, seq_sparse, pos_sparse = \
#     #         torch.stack(seq_ids), torch.stack(pos_ids), torch.stack(action_seqs), \
#     #         torch.stack(seq_sparse), torch.stack(pos_sparse)

#     #     seq_mm_batch, pos_mm_batch = {}, {}
#     #     if seq_mm[0]:
#     #         for fid in seq_mm[0].keys():
#     #             seq_mm_batch[fid] = torch.stack([x[fid] for x in seq_mm])
#     #             pos_mm_batch[fid] = torch.stack([x[fid] for x in pos_mm])
        
#     #     return (seq_ids, pos_ids, action_seqs, 
#     #             seq_sparse, pos_sparse, 
#     #             seq_mm_batch, pos_mm_batch,
#     #             list(full_histories))
#     # 新的 __getitem__ 方法
#     def __getitem__(self, uid):
#         # uid 是 dataloader 传过来的用户整数索引
        
#         # 1. <<< 获取用户静态特征 (极快) >>>
#         # 直接从内存中的字典里，用整数索引 uid 拿到预处理好的特征
#         user_static_feat_indexed = self.user_features.get(uid, {})
        
#         # 2. <<< 读取原始序列数据 >>>
#         # 注意：_load_user_data 需要 uid 的原始 ID，我们需要一个反向映射
#         # 为了简化，我们假设 dataloader 的 index 和我们的 uid 是一致的。
#         # 如果不一致，你需要建立一个 uid_idx -> original_uid 的映射。
#         # 暂时我们先假设传进来的 uid 就是文件中的用户行号索引。
#         user_sequence = self._load_user_data(uid)
        
#         item_interactions = []
#         user_full_history_items = set()
#         for u, i, user_feat, item_feat, action_type, ts in user_sequence:
#             if i and item_feat:
#                 # 我们只需要物品的原始ID和行为类型
#                 item_interactions.append({'id': i, 'action': action_type})
#                 # 同时，将物品的整数索引加入历史记录，用于负采样排除
#                 item_idx = self.item_original_id_to_idx.get(i)
#                 if item_idx is not None:
#                     user_full_history_items.add(item_idx)

#         full_history_tensor = torch.tensor(list(user_full_history_items), dtype=torch.long)
        
#         # 3. <<< 准备空的 Numpy 数组用于填充 >>>
#         # 物品ID序列 (存的是整数索引)
#         seq_ids = np.zeros(self.maxlen, dtype=np.int32)
#         pos_ids = np.zeros(self.maxlen, dtype=np.int32)
#         # 行为序列
#         action_seq = np.zeros(self.maxlen, dtype=np.int32)
#         # 物品稀疏特征序列 (存的是特征值的整数索引)
#         seq_sparse_feat = np.zeros((self.maxlen, len(self.ITEM_SPARSE_FEAT_KEYS)), dtype=np.int32)
#         pos_sparse_feat = np.zeros((self.maxlen, len(self.ITEM_SPARSE_FEAT_KEYS)), dtype=np.int32)
#         # 物品多模态特征序列 (存的是原始 embedding)
#         seq_mm_feat = {fid: np.zeros((self.maxlen, dim), dtype=np.float32) for fid, dim in self.ITEM_EMB_FEAT.items()}
#         pos_mm_feat = {fid: np.zeros((self.maxlen, dim), dtype=np.float32) for fid, dim in self.ITEM_EMB_FEAT.items()}

#         if not item_interactions:
#             # 如果用户没有交互，直接返回空的和静态特征
#             return self._package_tensors(seq_ids, pos_ids, action_seq, user_static_feat_indexed, 
#                                         seq_sparse_feat, pos_sparse_feat, seq_mm_feat, pos_mm_feat, 
#                                         full_history_tensor)

#         history = item_interactions[:-1]
#         targets = item_interactions[1:]
        
#         start_idx = max(0, len(history) - self.maxlen)
#         history = history[start_idx:]
#         targets = targets[start_idx:]
        
#         end_idx = len(history)
        
#         # 4. <<< 循环填充序列 (核心逻辑) >>>
#         for i in range(end_idx):
#             # --- 处理历史序列 ---
#             hist_item = history[i]
#             # 获取物品的整数索引
#             hist_item_idx = self.item_original_id_to_idx.get(hist_item['id'])
#             if hist_item_idx is not None:
#                 seq_ids[i] = hist_item_idx
#                 action_seq[i] = hist_item['action'] if hist_item['action'] is not None else 0
                
#                 # 从内存中获取该物品的预处理特征 (极快)
#                 item_features_indexed = self.item_features.get(hist_item_idx, {})
#                 # 填充稀疏特征
#                 for feat_idx, feat_key in enumerate(self.ITEM_SPARSE_FEAT_KEYS):
#                     seq_sparse_feat[i, feat_idx] = item_features_indexed.get(feat_key, 0)
#                 # 填充多模态特征
#                 for fid, dim in self.ITEM_EMB_FEAT.items():
#                     v = item_features_indexed.get(fid)
#                     if v is not None:
#                         a = np.asarray(v, dtype=np.float32).flatten()
#                         actual_len = min(len(a), dim)
#                         seq_mm_feat[fid][i, :actual_len] = a[:actual_len]

#             # --- 处理目标序列 (逻辑同上) ---
#             target_item = targets[i]
#             target_item_idx = self.item_original_id_to_idx.get(target_item['id'])
#             if target_item_idx is not None:
#                 pos_ids[i] = target_item_idx
                
#                 item_features_indexed = self.item_features.get(target_item_idx, {})
#                 for feat_idx, feat_key in enumerate(self.ITEM_SPARSE_FEAT_KEYS):
#                     pos_sparse_feat[i, feat_idx] = item_features_indexed.get(feat_key, 0)
#                 for fid, dim in self.ITEM_EMB_FEAT.items():
#                     v = item_features_indexed.get(fid)
#                     if v is not None:
#                         a = np.asarray(v, dtype=np.float32).flatten()
#                         actual_len = min(len(a), dim)
#                         pos_mm_feat[fid][i, :actual_len] = a[:actual_len]
                
#         return self._package_tensors(seq_ids, pos_ids, action_seq, user_static_feat_indexed,
#                                     seq_sparse_feat, pos_sparse_feat, seq_mm_feat, pos_mm_feat,
#                                     full_history_tensor)

# class MyTestDataset(MyDataset):
#     """
#     测试数据集，用于生成预测结果。
#     """
#     def _load_data_and_offsets(self):
#         """
#         加载预测用的用户序列数据和偏移量。
#         """
#         predict_file = self.data_dir / "predict_seq.jsonl"
#         offsets_file = self.data_dir / 'predict_seq_offsets.pkl'
        
#         if not predict_file.exists() or not offsets_file.exists():
#             raise FileNotFoundError(f"预测文件 {predict_file} 或 {offsets_file} 不存在。")
            
#         self.data_file = open(predict_file, 'rb')
#         with open(offsets_file, 'rb') as f:
#             self.seq_offsets = pickle.load(f)

#     # ⭐ [核心修改] 为测试集重写 __getitem__ 方法
#     def __getitem__(self, uid):
#         user_sequence = self._load_user_data(uid)
        
#         item_interactions = []
#         user_full_history_items = set()
#         for u, i, user_feat, item_feat, action_type, ts in user_sequence:
#             if i and item_feat:
#                 item_interactions.append({'id': i, 'feat': item_feat, 'action': action_type})
#                 user_full_history_items.add(i)

#         full_history_tensor = torch.tensor(list(user_full_history_items), dtype=torch.long)
        
#         seq_ids = np.zeros(self.maxlen, dtype=np.int32)
#         action_seq = np.zeros(self.maxlen, dtype=np.int32)
#         seq_feat = [self.feature_default_value.copy() for _ in range(self.maxlen)]
        
#         # 测试时没有目标，所以 pos 序列用 0 填充
#         pos_ids = np.zeros(self.maxlen, dtype=np.int32)
#         pos_feat = [self.feature_default_value.copy() for _ in range(self.maxlen)]

#         if not item_interactions:
#             tensors = self._features_to_tensors(seq_ids, pos_ids, action_seq, seq_feat, pos_feat)
#             return (*tensors, full_history_tensor)

#         # 测试逻辑: history 是完整的用户交互序列
#         history = item_interactions
        
#         start_idx = max(0, len(history) - self.maxlen)
#         history = history[start_idx:]
        
#         end_idx = len(history)
        
#         for i in range(end_idx):
#             hist_item = history[i]
#             seq_ids[i] = hist_item['id']
#             seq_feat[i] = self.fill_missing_feat(hist_item['feat'], hist_item['id'])
#             action_seq[i] = hist_item['action'] if hist_item['action'] is not None else 0
            
#         tensors = self._features_to_tensors(seq_ids, pos_ids, action_seq, seq_feat, pos_feat)
#         return (*tensors, full_history_tensor)

#     @staticmethod
#     def collate_fn(batch):
#         """
#         为测试集定制的 collate_fn。
#         它负责将 __getitem__ 的输出打包成与训练时格式一致的元组，
#         以便验证和推理代码可以统一处理。
#         """
#         # 解包 __getitem__ 返回的所有元素
#         (seq_ids, pos_ids, action_seqs, 
#          seq_sparse, pos_sparse, 
#          seq_mm, pos_mm, full_histories) = zip(*batch)

#         # 堆叠基本张量
#         seq_ids = torch.stack(seq_ids, dim=0)
#         pos_ids = torch.stack(pos_ids, dim=0) # pos_ids 在测试时作为 ground truth 或占位符
#         action_seqs = torch.stack(action_seqs, dim=0)
#         seq_sparse = torch.stack(seq_sparse, dim=0)
#         pos_sparse = torch.stack(pos_sparse, dim=0)

#         # 堆叠多模态特征字典
#         seq_mm_batch, pos_mm_batch = {}, {}
#         if seq_mm and seq_mm[0]:
#             for fid in seq_mm[0].keys():
#                 seq_mm_batch[fid] = torch.stack([x[fid] for x in seq_mm])
#                 pos_mm_batch[fid] = torch.stack([x[fid] for x in pos_mm])
        
#         # 返回所有打包好的数据
#         return (seq_ids, pos_ids, action_seqs, 
#                 seq_sparse, pos_sparse, 
#                 seq_mm_batch, pos_mm_batch,
#                 list(full_histories))



    



def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids):
    """
    加载多模态特征Embedding

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        if feat_id != '81':
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                for json_file in base_path.glob('*.json'):
                    with open(json_file, 'r', encoding='utf-8') as file:
                        for line in file:
                            data_dict_origin = json.loads(line.strip())
                            insert_emb = data_dict_origin['emb']
                            if isinstance(insert_emb, list):
                                insert_emb = np.array(insert_emb, dtype=np.float32)
                            data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                            emb_dict.update(data_dict)
            except Exception as e:
                print(f"transfer error: {e}")
        if feat_id == '81':
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
    return mm_emb_dict
import json
import pickle
import numpy as np
import torch
from pathlib import Path

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, args):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id # 多模态特征ID

        # --- 1. 加载所有预处理和官方文件 ---
        print("💾 正在加载官方和预处理文件...")
        with open(self.data_dir / 'indexer.pkl', 'rb') as f:
            self.indexer = pickle.load(f)
        with open(self.data_dir / 'user_features.pkl', 'rb') as f:
            self.user_features = pickle.load(f)
        with open(self.data_dir / 'item_features.pkl', 'rb') as f:
            self.item_features = pickle.load(f)
        with open(self.data_dir / 'seq_offsets.pkl', 'rb') as f:
            self.seq_offsets = pickle.load(f)
        print("✅ 文件加载完毕!")
        
        # 打开序列文件以备读取
        self.data_file = open(self.data_dir / "seq.jsonl", 'r', encoding='utf-8')

        # --- 2. 设置元信息和特征键 ---
        self.usernum = len(self.indexer['u'])
        self.itemnum = len(self.indexer['i'])
        # DataLoader 的索引是从0开始的，我们需要一个映射到官方的 user re-id
        # 我们假设 seq_offsets.pkl 的顺序对应了 user re-id 从1到N
        # 如果不是，需要根据 indexer['u'] 创建更复杂的映射
        self.dataloader_idx_to_user_reid = list(range(1, len(self.seq_offsets) + 1))


        self.USER_SPARSE_KEYS = ['103', '104', '105', '109']
        self.USER_ARRAY_KEYS = ['205', '207']
        self.MAX_ARRAY_LENGTH = 10
        self.ITEM_SPARSE_KEYS = ['100', '117', '111', '118', '101', '102', '119', '120', '114', '112', '121', '115', '122', '116']
        
        self.feat_statistics = {
            feat_id: len(feat_map) for feat_id, feat_map in self.indexer['f'].items()
        }
        self.feature_types = {
            'user_sparse': self.USER_SPARSE_KEYS,
            'user_array': self.USER_ARRAY_KEYS,
            'item_sparse': self.ITEM_SPARSE_KEYS,
            'item_emb': self.mm_emb_ids,
        }
        # (多模态相关的定义，如果适用的话)
        EMB_SHAPE_DICT = {"81": 32} # 假设只有81是多模态
        self.ITEM_EMB_FEAT_DIMS = {k: EMB_SHAPE_DICT[k] for k in self.mm_emb_ids if k in EMB_SHAPE_DICT}

    def _load_user_data(self, dataloader_idx):
        offset = self.seq_offsets[dataloader_idx]
        self.data_file.seek(offset)
        line = self.data_file.readline()
        return json.loads(line)

    def __len__(self):
        return len(self.seq_offsets)

    def __getitem__(self, dataloader_idx):
        user_reid = self.dataloader_idx_to_user_reid[dataloader_idx]
        
        # 1. 获取用户静态特征 (从预处理文件中)
        user_static_feat = self.user_features.get(user_reid, {})

        # 2. 读取用户行为序列
        user_sequence = self._load_user_data(dataloader_idx)
        
        item_interactions = []
        user_full_history_items = set()
        for record in user_sequence:
            # 只处理物品交互记录
            if record[1] is not None:
                item_id = record[1]
                action_type = record[4]
                item_interactions.append({'id': item_id, 'action': action_type})
                user_full_history_items.add(item_id)

        full_history_tensor = torch.tensor(list(user_full_history_items), dtype=torch.long)
        
        # 3. 准备空的 Numpy 数组
        seq_ids = np.zeros(self.maxlen, dtype=np.int32)
        pos_ids = np.zeros(self.maxlen, dtype=np.int32)
        action_seq = np.zeros(self.maxlen, dtype=np.int32)
        seq_sparse_feat = np.zeros((self.maxlen, len(self.ITEM_SPARSE_KEYS)), dtype=np.int32)
        pos_sparse_feat = np.zeros((self.maxlen, len(self.ITEM_SPARSE_KEYS)), dtype=np.int32)
        # (处理多模态，如果需要)
        # seq_mm_feat = ...

        # 4. 填充序列
        history = item_interactions[:-1]
        targets = item_interactions[1:]
        start_idx = max(0, len(history) - self.maxlen)
        history = history[start_idx:]
        targets = targets[start_idx:]
        end_idx = len(history)
        
        for i in range(end_idx):
            # 处理历史序列
            hist_item_id = history[i]['id']
            seq_ids[i] = hist_item_id
            action_seq[i] = history[i]['action'] if history[i]['action'] is not None else 0
            
            item_features = self.item_features.get(hist_item_id, {})
            for feat_idx, feat_key in enumerate(self.ITEM_SPARSE_KEYS):
                # 值已经是 re-id，直接使用
                seq_sparse_feat[i, feat_idx] = item_features.get(feat_key, 0)

            # 处理目标序列
            target_item_id = targets[i]['id']
            pos_ids[i] = target_item_id
            item_features = self.item_features.get(target_item_id, {})
            for feat_idx, feat_key in enumerate(self.ITEM_SPARSE_KEYS):
                pos_sparse_feat[i, feat_idx] = item_features.get(feat_key, 0)
        
        # 5. 将所有数据打包成张量
        return self._package_tensors(seq_ids, pos_ids, action_seq, user_static_feat,
                                     seq_sparse_feat, pos_sparse_feat, full_history_tensor)

    def _package_tensors(self, seq_ids, pos_ids, action_seq, user_static_feat,
                         seq_sparse_feat, pos_sparse_feat, full_history_tensor):
        
        user_sparse_tensor = torch.tensor([user_static_feat.get(k, 0) for k in self.USER_SPARSE_KEYS], dtype=torch.long)
        user_array_tensors = {
            k: torch.tensor(user_static_feat.get(k, [0] * self.MAX_ARRAY_LENGTH), dtype=torch.long)
            for k in self.USER_ARRAY_KEYS
        }

        seq_ids_tensor = torch.tensor(seq_ids, dtype=torch.long)
        pos_ids_tensor = torch.tensor(pos_ids, dtype=torch.long)
        action_seq_tensor = torch.tensor(action_seq, dtype=torch.long)
        seq_sparse_tensor = torch.tensor(seq_sparse_feat, dtype=torch.long)
        pos_sparse_tensor = torch.tensor(pos_sparse_feat, dtype=torch.long)
        
        # (多模态特征打包，如果需要)
        seq_mm_tensors, pos_mm_tensors = {}, {}

        return (seq_ids_tensor, pos_ids_tensor, action_seq_tensor, 
                user_sparse_tensor, user_array_tensors,
                seq_sparse_tensor, pos_sparse_tensor, 
                seq_mm_tensors, pos_mm_tensors, full_history_tensor)

    @staticmethod
    def collate_fn(batch):
        # 这个 collate_fn 与我们之前的版本一致，可以正确处理
        (seq_ids, pos_ids, action_seqs, 
         user_sparse, user_arrays,
         seq_sparse, pos_sparse, 
         seq_mm, pos_mm, full_histories) = zip(*batch)

        seq_ids, pos_ids, action_seqs, user_sparse, seq_sparse, pos_sparse = \
            torch.stack(seq_ids), torch.stack(pos_ids), torch.stack(action_seqs), \
            torch.stack(user_sparse), torch.stack(seq_sparse), torch.stack(pos_sparse)

        user_arrays_batch = {}
        if user_arrays and user_arrays[0]:
            for fid in user_arrays[0].keys():
                user_arrays_batch[fid] = torch.stack([x[fid] for x in user_arrays])
        
        # (多模态 collate 逻辑)
        seq_mm_batch, pos_mm_batch = {}, {}

        return (seq_ids, pos_ids, action_seqs, 
                user_sparse, user_arrays_batch,
                seq_sparse, pos_sparse, 
                seq_mm_batch, pos_mm_batch,
                list(full_histories))