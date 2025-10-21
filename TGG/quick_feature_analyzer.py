#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速特征维度分析器

这是一个简化版本的特征维度分析脚本，用于快速生成物品和用户特征的维度信息。
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def load_feature_data(data_dir: str) -> Dict[str, Any]:
    """
    加载特征数据
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        包含特征数据的字典
    """
    data_dir = Path(data_dir)
    data = {}
    
    # 加载物品特征字典
    item_feat_path = data_dir / "item_feat_dict.json"
    if item_feat_path.exists():
        with open(item_feat_path, 'r', encoding='utf-8') as f:
            data['item_feat_dict'] = json.load(f)
        print(f"✓ 已加载物品特征字典，包含 {len(data['item_feat_dict'])} 个物品")
    else:
        print("⚠ 物品特征字典文件不存在")
        data['item_feat_dict'] = {}
    
    # 加载索引文件
    indexer_path = data_dir / "indexer.pkl"
    if indexer_path.exists():
        with open(indexer_path, 'rb') as f:
            data['indexer'] = pickle.load(f)
        print("✓ 已加载索引文件")
    else:
        print("⚠ 索引文件不存在")
        data['indexer'] = {}
    
    return data


def define_feature_types() -> Dict[str, List[str]]:
    """
    定义特征类型
    
    Returns:
        特征类型字典
    """
    return {
        'user_sparse': ['103', '104', '105', '109'],
        'item_sparse': ['100', '117', '111', '118', '101', '102', '119', '120', '114', '112', '121', '115', '122', '116'],
        'item_array': [],
        'user_array': ['106', '107', '108', '110'],
        'item_emb': ['81', '82', '83', '84', '85', '86'],  # 多模态特征ID
        'user_continual': [],
        'item_continual': []
    }


def analyze_feature_dimensions(data: Dict[str, Any], feature_types: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    分析特征维度
    
    Args:
        data: 特征数据
        feature_types: 特征类型定义
        
    Returns:
        分析结果字典
    """
    print("正在分析特征维度...")
    
    result = {
        'feature_counts': {},
        'sparse_feature_cardinalities': {},
        'embedding_dimensions': {},
        'array_feature_lengths': {},
        'feature_coverage': {}
    }
    
    # 统计各类型特征数量
    for feat_type, feat_ids in feature_types.items():
        result['feature_counts'][feat_type] = len(feat_ids)
    
    # 分析稀疏特征基数
    if 'indexer' in data and 'f' in data['indexer']:
        for feat_type, feat_ids in feature_types.items():
            if 'sparse' in feat_type:
                for feat_id in feat_ids:
                    if feat_id in data['indexer']['f']:
                        cardinality = len(data['indexer']['f'][feat_id])
                        result['sparse_feature_cardinalities'][feat_id] = cardinality
    
    # 定义Embedding特征维度
    embedding_shapes = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    for feat_id in feature_types['item_emb']:
        if feat_id in embedding_shapes:
            result['embedding_dimensions'][feat_id] = embedding_shapes[feat_id]
    
    # 分析数组特征长度
    if 'item_feat_dict' in data:
        for feat_type, feat_ids in feature_types.items():
            if 'array' in feat_type:
                for feat_id in feat_ids:
                    lengths = []
                    for item_id, item_feat in data['item_feat_dict'].items():
                        if feat_id in item_feat:
                            feat_value = item_feat[feat_id]
                            if isinstance(feat_value, list):
                                lengths.append(len(feat_value))
                    if lengths:
                        result['array_feature_lengths'][feat_id] = {
                            'min_length': min(lengths),
                            'max_length': max(lengths),
                            'avg_length': np.mean(lengths),
                            'std_length': np.std(lengths)
                        }
    
    # 分析特征覆盖率
    if 'item_feat_dict' in data:
        total_items = len(data['item_feat_dict'])
        for feat_type, feat_ids in feature_types.items():
            if feat_ids:
                coverage_stats = {}
                for feat_id in feat_ids:
                    covered_items = sum(1 for item_feat in data['item_feat_dict'].values() 
                                      if feat_id in item_feat)
                    coverage_rate = covered_items / total_items * 100
                    coverage_stats[feat_id] = coverage_rate
                result['feature_coverage'][feat_type] = coverage_stats
    
    print("✓ 特征维度分析完成")
    return result


def generate_dimension_summary(analysis_result: Dict[str, Any], feature_types: Dict[str, List[str]]) -> str:
    """
    生成特征维度摘要
    
    Args:
        analysis_result: 分析结果
        feature_types: 特征类型定义
        
    Returns:
        格式化的摘要字符串
    """
    summary = []
    summary.append("=" * 60)
    summary.append("特征维度摘要")
    summary.append("=" * 60)
    summary.append("")
    
    # 特征类型统计
    summary.append("1. 特征类型统计")
    summary.append("-" * 30)
    for feat_type, count in analysis_result['feature_counts'].items():
        summary.append(f"{feat_type:20}: {count:3d} 个特征")
    summary.append("")
    
    # 稀疏特征基数
    if analysis_result['sparse_feature_cardinalities']:
        summary.append("2. 稀疏特征基数")
        summary.append("-" * 30)
        for feat_id, cardinality in sorted(analysis_result['sparse_feature_cardinalities'].items()):
            category = "用户" if feat_id in feature_types['user_sparse'] else "物品"
            summary.append(f"特征 {feat_id:3s} ({category:2s}): {cardinality:8d} 个唯一值")
        summary.append("")
    
    # Embedding特征维度
    if analysis_result['embedding_dimensions']:
        summary.append("3. Embedding特征维度")
        summary.append("-" * 30)
        for feat_id, dim in analysis_result['embedding_dimensions'].items():
            summary.append(f"特征 {feat_id:3s}: {dim:4d} 维")
        summary.append("")
    
    # 数组特征长度
    if analysis_result['array_feature_lengths']:
        summary.append("4. 数组特征长度")
        summary.append("-" * 30)
        for feat_id, stats in analysis_result['array_feature_lengths'].items():
            category = "用户" if feat_id in feature_types['user_array'] else "物品"
            summary.append(f"特征 {feat_id:3s} ({category:2s}): 长度范围 [{stats['min_length']:2d}, {stats['max_length']:2d}], "
                         f"平均 {stats['avg_length']:5.2f}")
        summary.append("")
    
    # 特征覆盖率
    if analysis_result['feature_coverage']:
        summary.append("5. 特征覆盖率")
        summary.append("-" * 30)
        for feat_type, coverage_stats in analysis_result['feature_coverage'].items():
            if coverage_stats:
                coverage_str = " ".join([f"{feat_id}:{rate:5.1f}%" for feat_id, rate in coverage_stats.items()])
                summary.append(f"{feat_type:20}: {coverage_str}")
        summary.append("")
    
    summary.append("=" * 60)
    return "\n".join(summary)


def save_summary_to_file(summary: str, output_path: str = "feature_dimensions.txt"):
    """
    保存摘要到文件
    
    Args:
        summary: 摘要内容
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"✓ 特征维度摘要已保存到: {output_path}")


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python quick_feature_analyzer.py <数据目录路径> [输出文件路径]")
        print("示例: python quick_feature_analyzer.py ./data ./feature_dimensions.txt")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "feature_dimensions.txt"
    
    print(f"开始分析数据目录: {data_dir}")
    
    try:
        # 1. 加载数据
        data = load_feature_data(data_dir)
        
        # 2. 定义特征类型
        feature_types = define_feature_types()
        
        # 3. 分析特征维度
        analysis_result = analyze_feature_dimensions(data, feature_types)
        
        # 4. 生成摘要
        summary = generate_dimension_summary(analysis_result, feature_types)
        
        # 5. 保存结果
        save_summary_to_file(summary, output_path)
        
        # 6. 打印摘要
        print("\n" + summary)
        
        print(f"\n🎉 特征维度分析完成！结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
