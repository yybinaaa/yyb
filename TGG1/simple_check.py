#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单检查seq.jsonl文件中的数据残缺情况
"""

import json

def check_data_integrity(file_path, max_lines=100):
    """检查数据完整性"""
    
    print(f"检查文件: {file_path}")
    print("=" * 60)
    
    incomplete_count = 0
    total_records = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num > max_lines:
                break
                
            try:
                data = json.loads(line)
                
                # 检查每个记录
                for record_idx, record in enumerate(data):
                    total_records += 1
                    
                    # 检查记录长度
                    if len(record) != 6:
                        print(f"行{line_num}, 记录{record_idx}: 长度不正确 ({len(record)} != 6)")
                        print(f"  内容: {record}")
                        incomplete_count += 1
                        continue
                    
                    user_id, item_id, user_feat, item_feat, action_type, timestamp = record
                    
                    # 检查字段完整性
                    issues = []
                    
                    if user_id is None:
                        issues.append("用户ID为None")
                    if item_id is None:
                        issues.append("物品ID为None")
                    if user_feat is None:
                        issues.append("用户特征为None")
                    if item_feat is None:
                        issues.append("物品特征为None")
                    if action_type is None:
                        issues.append("动作类型为None")
                    if timestamp is None:
                        issues.append("时间戳为None")
                    
                    # 检查特征字段类型
                    if user_feat is not None and not isinstance(user_feat, dict):
                        issues.append(f"用户特征类型错误: {type(user_feat)}")
                    if item_feat is not None and not isinstance(item_feat, dict):
                        issues.append(f"物品特征类型错误: {type(item_feat)}")
                    
                    if issues:
                        print(f"行{line_num}, 记录{record_idx}: {'; '.join(issues)}")
                        print(f"  内容: {record}")
                        incomplete_count += 1
                
                if line_num % 20 == 0:
                    print(f"已检查 {line_num} 行...")
                    
            except json.JSONDecodeError as e:
                print(f"行{line_num}: JSON解析错误 - {e}")
                incomplete_count += 1
            except Exception as e:
                print(f"行{line_num}: 处理错误 - {e}")
                incomplete_count += 1
    
    print("\n" + "=" * 60)
    print(f"检查完成!")
    print(f"检查行数: {min(max_lines, line_num)}")
    print(f"总记录数: {total_records}")
    print(f"残缺记录数: {incomplete_count}")
    print(f"残缺比例: {incomplete_count/total_records*100:.2f}%" if total_records > 0 else "无记录")

if __name__ == "__main__":
    file_path = "data/seq.jsonl"
    check_data_integrity(file_path, max_lines=50)

