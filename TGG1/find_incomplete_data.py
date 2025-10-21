#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查找seq.jsonl文件中的数据残缺情况
"""

import json
import re

def find_incomplete_data(file_path, max_examples=10):
    """查找数据残缺的例子"""
    
    incomplete_examples = []
    total_lines = 0
    
    print(f"正在分析文件: {file_path}")
    print("查找数据残缺的例子...\n")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 1000 == 0:
                print(f"已处理 {line_num} 行...")
            
            try:
                # 解析JSON行
                data = json.loads(line)
                total_lines += 1
                
                # 检查每个记录是否完整
                for record_idx, record in enumerate(data):
                    if len(record) != 6:
                        # 记录长度不正确
                        incomplete_examples.append({
                            'line': line_num,
                            'record_idx': record_idx,
                            'record': record,
                            'length': len(record),
                            'issue': f'记录长度不正确，期望6个字段，实际{len(record)}个字段'
                        })
                        continue
                    
                    user_id, item_id, user_feat, item_feat, action_type, timestamp = record
                    
                    # 检查字段类型和完整性
                    issues = []
                    
                    # 检查用户ID
                    if user_id is None:
                        issues.append("用户ID为None")
                    
                    # 检查物品ID
                    if item_id is None:
                        issues.append("物品ID为None")
                    
                    # 检查用户特征
                    if user_feat is None:
                        issues.append("用户特征为None")
                    elif not isinstance(user_feat, dict):
                        issues.append(f"用户特征类型错误: {type(user_feat)}")
                    
                    # 检查物品特征
                    if item_feat is None:
                        issues.append("物品特征为None")
                    elif not isinstance(item_feat, dict):
                        issues.append(f"物品特征类型错误: {type(item_feat)}")
                    
                    # 检查动作类型
                    if action_type is None:
                        issues.append("动作类型为None")
                    elif not isinstance(action_type, int):
                        issues.append(f"动作类型类型错误: {type(action_type)}")
                    
                    # 检查时间戳
                    if timestamp is None:
                        issues.append("时间戳为None")
                    elif not isinstance(timestamp, int):
                        issues.append(f"时间戳类型错误: {type(timestamp)}")
                    
                    # 如果有问题，添加到例子中
                    if issues:
                        incomplete_examples.append({
                            'line': line_num,
                            'record_idx': record_idx,
                            'record': record,
                            'issues': issues
                        })
                    
                    # 如果找到足够的例子，停止
                    if len(incomplete_examples) >= max_examples:
                        break
                
                if len(incomplete_examples) >= max_examples:
                    break
                    
            except json.JSONDecodeError as e:
                print(f"第{line_num}行JSON解析错误: {e}")
                incomplete_examples.append({
                    'line': line_num,
                    'record_idx': -1,
                    'record': line.strip(),
                    'issues': [f'JSON解析错误: {e}']
                })
            except Exception as e:
                print(f"第{line_num}行处理错误: {e}")
    
    print(f"\n分析完成!")
    print(f"总行数: {total_lines}")
    print(f"找到残缺例子: {len(incomplete_examples)}")
    
    # 输出残缺例子
    print("\n=== 数据残缺例子 ===")
    for i, example in enumerate(incomplete_examples, 1):
        print(f"\n例子 {i}:")
        print(f"  行号: {example['line']}")
        print(f"  记录索引: {example['record_idx']}")
        print(f"  记录内容: {example['record']}")
        if 'issues' in example:
            print(f"  问题: {'; '.join(example['issues'])}")
        elif 'issue' in example:
            print(f"  问题: {example['issue']}")
        print("-" * 50)
    
    return incomplete_examples

if __name__ == "__main__":
    file_path = "data/seq.jsonl"
    find_incomplete_data(file_path, max_examples=15)

