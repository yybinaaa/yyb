import json
import pickle
from pathlib import Path
from tqdm import tqdm
import os

def restructure_provided_data(data_dir: Path):
    """
    读取官方提供的数据文件，将其重组为我们为高效训练设计的结构。
    - 读取 item_feat_dict.json -> 创建 item_features.pkl
    - 读取 seq.jsonl -> 提取用户特征 -> 创建 user_features.pkl
    """
    print("🚀 开始重组官方数据以便高效加载...")

    # --- 参数定义 ---
    # 这些特征键列表需要您根据实际情况确认
    USER_SPARSE_KEYS = ['103', '104', '105', '109'] # 假设这些是用户稀疏特征
    USER_ARRAY_KEYS = ['205', '207'] # 假设这些是用户数组特征
    MAX_ARRAY_LENGTH_K = 10 # 最终每个用户的数组特征，我们选择 K 个

    # --- 文件路径 ---
    user_seq_file = data_dir / "seq.jsonl"
    item_feat_file = data_dir / "item_feat_dict.json"
    if not user_seq_file.exists() or not item_feat_file.exists():
        raise FileNotFoundError(f"请确保 {user_seq_file} 和 {item_feat_file} 都存在。")

    # --- 1. 处理物品特征 ---
    print("\n--- 步骤 1: 转换 item_feat_dict.json -> item_features.pkl ---")
    with open(item_feat_file, 'r', encoding='utf-8') as f:
        item_feat_dict = json.load(f)
    
    # 因为 key 已经是 re-id，我们只需要确保它们是整数类型
    item_features_data = {int(k): v for k, v in item_feat_dict.items()}

    with open(data_dir / 'item_features.pkl', 'wb') as f:
        pickle.dump(item_features_data, f)
    print(f"  - ✅ 已保存: {data_dir / 'item_features.pkl'} (共 {len(item_features_data)} 个物品)")

    # --- 2. 处理用户特征 ---
    print("\n--- 步骤 2: 从 seq.jsonl 提取用户特征 -> user_features.pkl ---")
    user_feature_data = {}
    with open(user_seq_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="扫描用户序列提取特征"):
            try:
                sequence = json.loads(line)
                if not sequence: continue
                
                current_user_id = -1
                temp_user_features = {}

                for record in sequence:
                    user_id, item_id, user_feat, _, _, _ = record
                    current_user_id = user_id
                    
                    # 如果是用户特征记录 (item_id is null)
                    if item_id is None and user_feat:
                        temp_user_features.update(user_feat)
                
                # 在处理完整个序列后，为该用户处理特征
                if current_user_id != -1:
                    processed_feats = {}
                    # 处理稀疏特征
                    for key in USER_SPARSE_KEYS:
                        processed_feats[key] = temp_user_features.get(key, 0) # 缺失值为0
                    
                    # 处理数组特征 (Top-K & Padding)
                    for key in USER_ARRAY_KEYS:
                        vals = temp_user_features.get(key, [])
                        if not isinstance(vals, list): vals = [vals]
                        
                        top_k_vals = vals[:MAX_ARRAY_LENGTH_K]
                        padded_vals = top_k_vals + [0] * (MAX_ARRAY_LENGTH_K - len(top_k_vals))
                        processed_feats[key] = padded_vals
                    
                    user_feature_data[current_user_id] = processed_feats

            except (json.JSONDecodeError, IndexError):
                continue

    with open(data_dir / 'user_features.pkl', 'wb') as f:
        pickle.dump(user_feature_data, f)
    print(f"  - ✅ 已保存: {data_dir / 'user_features.pkl'} (共 {len(user_feature_data)} 个用户)")
    print("\n🎉 数据重组完毕！")


if __name__ == '__main__':
    current_data_dir = Path(os.environ.get('TRAIN_DATA_PATH', 'data'))
    restructure_provided_data(current_data_dir)