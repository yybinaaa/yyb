import json
import pickle
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import os

def find_action_feature_id(data_dir: Path):
    """
    通过扫描 seq.jsonl 和 indexer.pkl 来自动推断
    哪个 feature_id 对应 action_type。
    """
    seq_file = data_dir / "seq.jsonl"
    indexer_file = data_dir / "indexer.pkl"

    if not seq_file.exists() or not indexer_file.exists():
        raise FileNotFoundError(f"请确保 {seq_file} 和 {indexer_file} 存在。")

    print("--- 步骤 1: 扫描 seq.jsonl 以收集所有出现过的 action_type re-id ---")
    
    action_type_values = set()
    with open(seq_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="扫描Action Types"):
            try:
                sequence = json.loads(line)
                for record in sequence:
                    # record[4] is action_type
                    action_val = record[4]
                    if action_val is not None:
                        action_type_values.add(action_val)
            except (json.JSONDecodeError, IndexError):
                continue
    
    if not action_type_values:
        print("❌ 错误: 在 seq.jsonl 中没有找到任何有效的 action_type 值。")
        return None

    print(f"✅ 扫描完成。发现的 Action Type re-id 值: {sorted(list(action_type_values))}")
    
    print("\n--- 步骤 2: 加载 indexer.pkl 并反向查找特征ID ---")
    with open(indexer_file, 'rb') as f:
        indexer = pickle.load(f)
        
    feature_dictionaries = indexer.get('f', {})
    if not feature_dictionaries:
        print("❌ 错误: indexer.pkl 中没有 'f' (特征) 字段。")
        return None

    candidate_features = []
    print("正在匹配词典...")
    for feature_id, value_map in feature_dictionaries.items():
        # value_map 是一个 dict, e.g., {'val_A': 1, 'val_B': 2}
        # 我们需要检查它的 values()
        indexed_values_in_dict = set(value_map.values())
        
        # 判断 action_type_values 是否是 indexed_values_in_dict 的子集
        if action_type_values.issubset(indexed_values_in_dict):
            # 计算匹配度：词典中额外元素的数量。越少越好。
            extra_elements = len(indexed_values_in_dict - action_type_values)
            candidate_features.append({
                'id': feature_id,
                'match_score': extra_elements, # 分数越低，匹配越好
                'vocab_size': len(indexed_values_in_dict)
            })
            print(f"  - 发现候选特征ID: '{feature_id}' (词典大小: {len(indexed_values_in_dict)}, 额外元素: {extra_elements})")

    if not candidate_features:
        print("\n❌ 未能找到任何特征的词典完全包含所有发现的 action_type 值。")
        print("这可能意味着 action_type 没有被包含在 indexer['f'] 中，或者数据有误。")
        return None

    # 按匹配分数（额外元素数量）排序，分数越低越靠前
    candidate_features.sort(key=lambda x: x['match_score'])
    
    best_match = candidate_features[0]
    
    print("\n--- 结论 ---")
    print(f"🎉 最有可能的 Action Feature ID 是: '{best_match['id']}'")
    print(f"   - 它的词典大小是 {best_match['vocab_size']}")
    print(f"   - 与实际action值相比，它只多了 {best_match['match_score']} 个额外元素 (最少)。")
    print("\n请将这个ID用于您的 main_three_tower.py 脚本中。")
    
    return best_match['id']


if __name__ == '__main__':
    data_path = Path(os.environ.get('TRAIN_DATA_PATH', 'data'))
    find_action_feature_id(data_path)