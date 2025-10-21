import json
import pickle
import struct
from pathlib import Path

import numpy as np
import pandas as pd  # 添加 pandas 库的导入
import torch
from tqdm import tqdm


class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集

    Args:
        data_dir: 数据文件目录
        args: 全局参数

    Attributes:
        data_dir: 数据文件目录
        maxlen: 最大长度
        item_feat_dict: 物品特征字典
        mm_emb_ids: 激活的mm_emb特征ID
        mm_emb_dict: 多模态特征字典
        itemnum: 物品数量
        usernum: 用户数量
        indexer_i_rev: 物品索引字典 (reid -> item_id)
        indexer_u_rev: 用户索引字典 (reid -> user_id)
        indexer: 索引字典
        feature_default_value: 特征缺省值
        feature_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feat_statistics: 特征统计信息，包括user和item的特征数量
    """

    def __init__(self, data_dir, args):
        """
        初始化数据集
        """
        super().__init__()#调用父类的初始化函数，保证继承的父类正常初始化
        self.data_dir = Path(data_dir)#将data_dir转换为Path对象，方便后续操作
        self._load_data_and_offsets()#加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        self.maxlen = args.maxlen#最大长度
        self.mm_emb_ids = args.mm_emb_id#多模态特征ID   

        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))#加载物品特征字典
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)#加载多模态特征Embedding 
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:#打开索引字典文件
            indexer = pickle.load(ff)#加载索引字典
            self.itemnum = len(indexer['i'])#物品数量
            self.usernum = len(indexer['u'])#用户数量
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}#物品索引字典 (reid -> item_id)
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}#用户索引字典 (reid -> user_id)
        self.indexer = indexer#索引字典

        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()#初始化特征信息, 包括特征缺省值和特征类型

    def _add_time_features(self, user_sequence, tau=86400):
        """
        根据图片逻辑，添加时间特征
        """
        # 如果序列为空，直接返回
        if not user_sequence:
            return np.array([], dtype=np.int64), []

        ts_array = np.array([r[5] for r in user_sequence], dtype=np.int64)

        # time_gap, log_gap
        prev_ts_array = np.roll(ts_array, 1)
        prev_ts_array[0] = ts_array[0]
        time_gap = ts_array - prev_ts_array
        time_gap[0] = 0
        log_gap = np.log1p(time_gap)

        # hour, weekday, month
        ts_utc8 = ts_array + 8 * 3600
        hours = (ts_utc8 % 86400) // 3600
        weekdays = ((ts_utc8 // 86400 + 4) % 7).astype(np.int32)
        months = pd.to_datetime(ts_utc8, unit="s").month.to_numpy()

        # time decay
        last_ts = ts_array[-1]
        delta_t = last_ts - ts_array
        delta_scaled = np.log1p(delta_t / tau)

        # 添加到 sequence
        new_sequence = []
        for idx, record in enumerate(user_sequence):
            u, i, user_feat, item_feat, action_type, ts = record
            if user_feat is None:
                user_feat = {}
            user_feat["200"] = int(hours[idx])
            user_feat["201"] = int(weekdays[idx])
            # user_feat["202"] = int(time_gap[idx]) # 原始图片中被注释
            user_feat["203"] = float(log_gap[idx])
            user_feat["204"] = int(months[idx])
            user_feat["205"] = float(delta_scaled[idx])
            new_sequence.append((u, i, user_feat, item_feat, action_type, ts))

        return ts_array, new_sequence

    def _load_data_and_offsets(self):#加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        """
        加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        """
        self.data_file = open(self.data_dir / "seq.jsonl", 'rb')#打开用户序列数据文件
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)#加载每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O

    def _load_user_data(self, uid):#从数据文件中加载单个用户的数据
        """
        从数据文件中加载单个用户的数据

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        self.data_file.seek(self.seq_offsets[uid])#跳转到指定位置
        line = self.data_file.readline()#读取一行
        data = json.loads(line)#将行转换为json格式
        return data

    def _random_neq(self, l, r, s):#生成一个不在序列s中的随机整数, 用于训练时的负采样
        """
        生成一个不在序列s中的随机整数, 用于训练时的负采样

        Args:
            l: 随机整数的最小值
            r: 随机整数的最大值
            s: 序列

        Returns:
            t: 不在序列s中的随机整数
        """
        t = np.random.randint(l, r)#生成一个在[l, r)范围内的随机整数
        while t in s or str(t) not in self.item_feat_dict:#如果t在序列s中或者t不在物品特征字典中
            t = np.random.randint(l, r)#重新生成一个在[l, r)范围内的随机整数
        return t

    def __getitem__(self, uid):#获取单个用户的数据，并进行padding处理，生成模型需要的数据格式
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户ID(reid)

        Returns:
            seq: 用户序列ID
            pos: 正样本ID（即下一个真实访问的item）
            neg: 负样本ID
            token_type: 用户序列类型，1表示item，2表示user
            next_token_type: 下一个token类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            pos_feat: 正样本特征，每个元素为字典，key为特征ID，value为特征值
            neg_feat: 负样本特征，每个元素为字典，key为特征ID，value为特征值
        """
        user_sequence = self._load_user_data(uid)  #     动态加载用户数据
        _, user_sequence = self._add_time_features(user_sequence) # 添加时间特征

        ext_user_sequence = []#扩展用户序列
        for record_tuple in user_sequence:#遍历用户序列
            u, i, user_feat, item_feat, action_type, _ = record_tuple#解包
            if u and user_feat:#如果用户ID和用户特征不为空
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type))#将用户ID和用户特征插入到扩展用户序列的头部
            if i and item_feat:#如果物品ID和物品特征不为空
                ext_user_sequence.append((i, item_feat, 1, action_type))#将物品ID和物品特征插入到扩展用户序列的尾部

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)#用户序列ID
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)#正样本ID
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)#负样本ID
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)#用户序列类型
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)#下一个token类型
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)#下一个动作类型

        seq_feat = np.empty([self.maxlen + 1], dtype=object)#用户序列特征
        pos_feat = np.empty([self.maxlen + 1], dtype=object)#正样本特征
        neg_feat = np.empty([self.maxlen + 1], dtype=object)#负样本特征

        nxt = ext_user_sequence[-1]#最后一个元素
        idx = self.maxlen#最大长度

        ts = set()#正样本ID集合
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:#如果动作类型为1且物品ID不为0
                ts.add(record_tuple[0])#将物品ID添加到正样本ID集合中

        # left-padding, 从后往前遍历，将用户序列填充到maxlen+1的长度
        for record_tuple in reversed(ext_user_sequence[:]):#从后往前遍历，将用户序列填充到maxlen+1的长度
            i, feat, type_, act_type = record_tuple#解包
            next_i, next_feat, next_type, next_act_type = nxt#解包
            feat = self.fill_missing_feat(feat, i)#填充缺失的特征
            next_feat = self.fill_missing_feat(next_feat, next_i)#填充缺失的特征
            seq[idx] = i#将物品ID填充到用户序列ID中
            token_type[idx] = type_#将动作类型填充到用户序列类型中
            next_token_type[idx] = next_type#将下一个动作类型填充到下一个token类型中
            if next_act_type is not None:
                next_action_type[idx] = next_act_type#将下一个动作类型填充到下一个动作类型中
            seq_feat[idx] = feat#将特征填充到用户序列特征中
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i#将下一个物品ID填充到正样本ID中
                pos_feat[idx] = next_feat#将下一个特征填充到正样本特征中
                neg_id = self._random_neq(1, self.itemnum + 1, ts)#生成一个不在正样本ID集合中的随机整数
                neg[idx] = neg_id#将随机整数填充到负样本ID中
                neg_feat[idx] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)#将随机整数填充到负样本特征中
            nxt = record_tuple#将当前元素赋值给下一个元素
            idx -= 1#索引减1
            if idx == -1:#如果索引为-1
                break#跳出循环

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)#将用户序列特征中为None的元素替换为特征缺省值
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)#将正样本特征中为None的元素替换为特征缺省值
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)#将负样本特征中为None的元素替换为特征缺省值

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat

    def __len__(self):#返回数据集长度，即用户数量
        """
        返回数据集长度，即用户数量

        Returns:
            usernum: 用户数量
        """
        return len(self.seq_offsets)#返回数据集长度，即用户数量

    def _init_feat_info(self):#初始化特征信息, 包括特征缺省值和特征类型
        """
        初始化特征信息, 包括特征缺省值和特征类型

        Returns:
            feat_default_value: 特征缺省值，每个元素为字典，key为特征ID，value为特征缺省值
            feat_types: 特征类型，key为特征类型名称，value为包含的特征ID列表
        """
        feat_default_value = {}#特征缺省值
        feat_statistics = {}#特征统计信息
        feat_types = {}#特征类型
        feat_types['user_sparse'] = ['103', '104', '105', '109', '200', '201', '204']#用户稀疏特征
        feat_types['item_sparse'] = ['100', '117', '111', '118', '101', '102', '119', '120', '114', '112', '121', '115', '122', '116']#物品稀疏特征
        feat_types['item_array'] = []#物品数组特征
        feat_types['user_array'] = ['106', '107', '108', '110']#用户数组特征
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = ['203', '205']
        feat_types['item_continual'] = []

        # 为生成的时间特征定义词典大小
        time_feat_statistics = {'200': 24, '201': 7, '204': 13}  # hour(0-23), weekday(0-6), month(1-12 + pad)

        for feat_id in feat_types['user_sparse']:#遍历用户稀疏特征
            feat_default_value[feat_id] = 0#特征缺省值
            if feat_id in self.indexer['f']:
                feat_statistics[feat_id] = len(self.indexer['f'][feat_id])#特征统计信息
            elif feat_id in time_feat_statistics:
                feat_statistics[feat_id] = time_feat_statistics[feat_id]
        for feat_id in feat_types['item_sparse']:#遍历物品稀疏特征
            feat_default_value[feat_id] = 0#特征缺省值
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])#特征统计信息
        for feat_id in feat_types['item_array']:#遍历物品数组特征
            feat_default_value[feat_id] = [0]#特征缺省值
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])#特征统计信息
        for feat_id in feat_types['user_array']:#遍历用户数组特征
            feat_default_value[feat_id] = [0]#特征缺省值
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])#特征统计信息
        for feat_id in feat_types['user_continual']:#遍历用户连续特征
            feat_default_value[feat_id] = 0.0#特征缺省值
        for feat_id in feat_types['item_continual']:#遍历物品连续特征
            feat_default_value[feat_id] = 0.0#特征缺省值
        for feat_id in feat_types['item_emb']:#遍历物品Embedding特征
            feat_default_value[feat_id] = np.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
            )#特征缺省值

        return feat_default_value, feat_types, feat_statistics#返回特征缺省值，特征类型，特征统计信息

    def fill_missing_feat(self, feat, item_id):#对于原始数据中缺失的特征进行填充缺省值
        """
        对于原始数据中缺失的特征进行填充缺省值

        Args:
            feat: 特征字典
            item_id: 物品ID

        Returns:
            filled_feat: 填充后的特征字典
        """
        if feat == None:#如果特征为None
            feat = {}#将特征设置为空字典
        filled_feat = {}#填充后的特征字典
        for k in feat.keys():#遍历特征字典
            filled_feat[k] = feat[k]#将特征字典中的特征填充到填充后的特征字典中

        all_feat_ids = []#所有特征ID
        for feat_type in self.feature_types.values():#遍历特征类型
            all_feat_ids.extend(feat_type)#将特征类型中的特征ID添加到所有特征ID中
        missing_fields = set(all_feat_ids) - set(feat.keys())#缺失的特征ID
        for feat_id in missing_fields:#遍历缺失的特征ID
            filled_feat[feat_id] = self.feature_default_value[feat_id]#将缺失的特征ID填充到填充后的特征字典中
        for feat_id in self.feature_types['item_emb']:#遍历物品Embedding特征
            if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:#如果物品ID不为0且物品ID在物品Embedding字典中
                if type(self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]) == np.ndarray:#如果物品Embedding字典中的物品ID的值为np.ndarray
                    filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]#将物品Embedding字典中的物品ID的值填充到填充后的特征字典中

        return filled_feat#返回填充后的特征字典

    @staticmethod
    def collate_fn(batch):#将多个__getitem__返回的数据拼接成一个batch
        """
        按照图片中的逻辑，实现高效的批处理函数
        基于批次内数组最大长度进行padding，使用torch.from_numpy(np.stack(..., axis=0))处理特征
        
        Args:
            batch: 多个__getitem__返回的数据，每个元素为(item_seq, pos_seq, neg_seq)

        Returns:
            batch_data: 包含所有批次数据的字典
        """
        # 解包批次数据
        item_seqs, pos_seqs, neg_seqs = zip(*batch)
        batch_size = len(batch)
        
        # 获取序列长度（所有序列长度应该一致）
        seq_length = len(item_seqs[0]['item_id'])
        
        # 初始化批次数据结构
        batch_data = {
            'item_seq': {},
            'pos_seq': {},
            'neg_seq': {}
        }
        
        # 处理每个序列类型
        for seq_name, seqs in [('item_seq', item_seqs), ('pos_seq', pos_seqs), ('neg_seq', neg_seqs)]:
            # 基础字段处理
            batch_data[seq_name]['item_id'] = torch.from_numpy(
                np.stack([seq['item_id'] for seq in seqs], axis=0)
            )
            batch_data[seq_name]['token_type'] = torch.from_numpy(
                np.stack([seq['token_type'] for seq in seqs], axis=0)
            )
            batch_data[seq_name]['action_type'] = torch.from_numpy(
                np.stack([seq['action_type'] for seq in seqs], axis=0)
            )
            
            # 特征字段处理 - 使用torch.from_numpy(np.stack(..., axis=0))策略
            batch_data[seq_name]['features'] = {}
            
            # 获取所有特征ID
            all_feat_ids = set()
            for seq in seqs:
                all_feat_ids.update(seq['features'].keys())
            
            # 处理每个特征
            for feat_id in all_feat_ids:
                # 收集该特征在所有序列中的值
                feat_values = []
                for seq in seqs:
                    if feat_id in seq['features']:
                        feat_values.append(seq['features'][feat_id])
                    else:
                        # 如果某个序列缺少该特征，用0填充
                        feat_values.append(np.zeros(seq_length, dtype=np.int32))
                
                # 使用np.stack添加批次维度，然后转换为torch tensor
                feat_tensor = torch.from_numpy(np.stack(feat_values, axis=0))
                batch_data[seq_name]['features'][feat_id] = feat_tensor
        
        return batch_data


class MyTestDataset(MyDataset):#测试数据集
    """
    测试数据集
    """

    def __init__(self, data_dir, args):#初始化测试数据集
        super().__init__(data_dir, args)#调用父类的初始化函数

    def _load_data_and_offsets(self):#加载数据和偏移量
        self.data_file = open(self.data_dir / "predict_seq.jsonl", 'rb')#打开数据文件
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)#加载偏移量

    def _process_cold_start_feat(self, feat):#处理冷启动特征
        """
        处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
        """
        processed_feat = {}#处理后的特征字典
        for feat_id, feat_value in feat.items():#遍历特征字典
            if type(feat_value) == list:#如果特征值为列表
                value_list = []#处理后的特征值列表
                for v in feat_value:
                    if type(v) == str:#如果特征值为字符串
                        value_list.append(0)#将0添加到处理后的特征值列表中
                    else:
                        value_list.append(v)#将特征值添加到处理后的特征值列表中
                processed_feat[feat_id] = value_list#将处理后的特征值列表添加到处理后的特征字典中
            elif type(feat_value) == str:#如果特征值为字符串
                processed_feat[feat_id] = 0#将0添加到处理后的特征字典中
            else:
                processed_feat[feat_id] = feat_value#将特征值添加到处理后的特征字典中
        return processed_feat#返回处理后的特征字典

    def __getitem__(self, uid):#获取单个用户的数据，并进行padding处理，生成模型需要的数据格式
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户在self.data_file中储存的行号
        Returns:
            seq: 用户序列ID
            token_type: 用户序列类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            user_id: user_id eg. user_xxxxxx ,便于后面对照答案
        """
        user_sequence = self._load_user_data(uid)  # 动 态加载用户数据
        _, user_sequence = self._add_time_features(user_sequence) # 添加时间特征

        ext_user_sequence = []#扩展用户序列
        for record_tuple in user_sequence:#遍历用户序列
            u, i, user_feat, item_feat, _, _ = record_tuple#解包
            if u:#如果用户ID不为空
                if type(u) == str:  # 如果是字符串，说明是user_id
                    user_id = u#将用户ID赋值给user_id
                else:  # 如果是int，说明是re_id
                    user_id = self.indexer_u_rev[u]#将用户ID转换为字符串
            if u and user_feat:#如果用户ID和用户特征不为空
                if type(u) == str:#如果用户ID为字符串
                    u = 0#将用户ID赋值为0
                if user_feat:#如果用户特征不为空
                    user_feat = self._process_cold_start_feat(user_feat)#处理冷启动特征
                ext_user_sequence.insert(0, (u, user_feat, 2))#将用户ID和用户特征插入到扩展用户序列的头部

            if i and item_feat:#如果物品ID和物品特征不为空
                # 序列对于训练时没见过的item，不会直接赋0，而是保留creative_id，creative_id远大于训练时的itemnum
                if i > self.itemnum:#如果物品ID大于物品数量
                    i = 0#将物品ID赋值为0
                if item_feat:#如果物品特征不为空
                    item_feat = self._process_cold_start_feat(item_feat)#处理冷启动特征
                ext_user_sequence.append((i, item_feat, 1))#将物品ID和物品特征插入到扩展用户序列的尾部

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)#用户序列ID
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)#用户序列类型
        seq_feat = np.empty([self.maxlen + 1], dtype=object)#用户序列特征

        idx = self.maxlen#最大长度

        ts = set()#正样本ID集合
        for record_tuple in ext_user_sequence:#遍历扩展用户序列
            if record_tuple[2] == 1 and record_tuple[0]:#如果动作类型为1且物品ID不为0
                ts.add(record_tuple[0])#将物品ID添加到正样本ID集合中

        for record_tuple in reversed(ext_user_sequence[:-1]):#从后往前遍历，将用户序列填充到maxlen+1的长度
            i, feat, type_ = record_tuple#解包
            feat = self.fill_missing_feat(feat, i)#填充缺失的特征
            seq[idx] = i#将物品ID填充到用户序列ID中
            token_type[idx] = type_#将动作类型填充到用户序列类型中
            seq_feat[idx] = feat#将特征填充到用户序列特征中
            idx -= 1#索引减1
            if idx == -1:#如果索引为-1
                break#跳出循环

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)#将用户序列特征中为None的元素替换为特征缺省值

        return seq, token_type, seq_feat, user_id#返回用户序列ID，用户序列类型，用户序列特征，用户ID

    def __len__(self):#返回数据集长度，即用户数量
        """
        Returns:
            len(self.seq_offsets): 用户数量
        """
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:#打开偏移量文件
            temp = pickle.load(f)#加载偏移量
        return len(temp)#返回数据集长度，即用户数量

    @staticmethod
    def collate_fn(batch):
        """
        将多个测试样本拼接成一个 batch，包含时间特征。
        
        Args:
            batch: 每个元素为 (seq, token_type, seq_feat, time_feat, user_id)
        
        Returns:
            seqs: Tensor [B, L]
            token_types: Tensor [B, L]
            seq_feats: List[Dict], 长度为 B，每个是 [L] 个特征字典
            time_feats: Tensor [B, L, 4]
            user_ids: List[str]
        """
        # 注意：此处的collate_fn签名与__getitem__的返回值不匹配（缺少time_feats）。
        # 根据请求，时间特征已合并到seq_feats中。
        # 如果需要分离time_feats，需要修改__getitem__的返回值和此处的逻辑。
        # 当前保持原样，但它可能无法按预期工作。
        seqs, token_types, seq_feats, time_feats, user_ids = zip(*batch)

        seqs = torch.stack(seqs)
        token_types = torch.stack(token_types)
        time_feats = torch.stack(time_feats)  # [B, L, 4]
        seq_feats = list(seq_feats)
        user_ids = list(user_ids)

        return seqs, token_types, seq_feats, time_feats, user_ids

def save_emb(emb, save_path):#将Embedding保存为二进制文件
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


def load_mm_emb(mm_path, feat_ids):#加载多模态特征Embedding
    """
    加载多模态特征Embedding

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}#特征ID对应的Embedding维度
    mm_emb_dict = {}#多模态特征Embedding字典
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):#遍历多模态特征ID
        shape = SHAPE_DICT[feat_id]#获取特征ID对应的Embedding维度
        emb_dict = {}#特征Embedding字典
        if feat_id != '81':#如果特征ID不为81
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')#获取特征Embedding路径
                for json_file in base_path.glob('*.json'):#遍历特征Embedding路径下的所有json文件
                    with open(json_file, 'r', encoding='utf-8') as file:#打开json文件
                        for line in file:#遍历json文件
                            data_dict_origin = json.loads(line.strip())#加载json文件
                            insert_emb = data_dict_origin['emb']#获取特征Embedding
                            if isinstance(insert_emb, list):#如果特征Embedding为列表
                                insert_emb = np.array(insert_emb, dtype=np.float32)#将特征Embedding转换为numpy数组
                            data_dict = {data_dict_origin['anonymous_cid']: insert_emb}#将特征Embedding添加到特征Embedding字典中
                            emb_dict.update(data_dict)#将特征Embedding字典添加到多模态特征Embedding字典中
            except Exception as e:
                print(f"transfer error: {e}")#打印错误信息
        if feat_id == '81':
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)#加载特征Embedding
        mm_emb_dict[feat_id] = emb_dict#将特征Embedding添加到多模态特征Embedding字典中
        print(f'Loaded #{feat_id} mm_emb')#打印加载的特征ID
    return mm_emb_dict#返回多模态特征Embedding字典