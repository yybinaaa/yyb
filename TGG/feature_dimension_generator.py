#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
物品和用户特征维度生成器

这个脚本用于分析和生成物品、用户特征的维度信息，包括：
1. 特征类型统计
2. 特征维度分析
3. 特征分布统计
4. 生成特征维度报告
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import argparse


class FeatureDimensionGenerator:
    """
    特征维度生成器类
    """
    
    def __init__(self, data_dir: str):
        """
        初始化特征维度生成器
        
        Args:
            data_dir: 数据文件目录路径
        """
        self.data_dir = Path(data_dir)
        self.item_feat_dict = None
        self.indexer = None
        self.mm_emb_dict = None
        self.feature_types = None
        self.feature_statistics = {}
        
    def load_data(self):
        """加载必要的数据文件"""
        print("正在加载数据文件...")
        
        # 加载物品特征字典
        item_feat_path = self.data_dir / "item_feat_dict.json"
        if item_feat_path.exists():
            with open(item_feat_path, 'r', encoding='utf-8') as f:
                self.item_feat_dict = json.load(f)
            print(f"✓ 已加载物品特征字典，包含 {len(self.item_feat_dict)} 个物品")
        else:
            print("⚠ 物品特征字典文件不存在")
            
        # 加载索引文件
        indexer_path = self.data_dir / "indexer.pkl"
        if indexer_path.exists():
            with open(indexer_path, 'rb') as f:
                self.indexer = pickle.load(f)
            print(f"✓ 已加载索引文件")
        else:
            print("⚠ 索引文件不存在")
            
        # 加载多模态特征（如果存在）
        mm_emb_path = self.data_dir / "creative_emb"
        if mm_emb_path.exists():
            self.mm_emb_dict = self._load_mm_emb(mm_emb_path)
            print(f"✓ 已加载多模态特征")
        else:
            print("⚠ 多模态特征目录不存在")
            
        print("数据加载完成！")
        
    def _load_mm_emb(self, mm_path: Path) -> Dict:
        """加载多模态特征Embedding"""
        SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        mm_emb_dict = {}
        
        for feat_id, shape in SHAPE_DICT.items():
            try:
                if feat_id == '81':
                    emb_file = mm_path / f'emb_{feat_id}_{shape}.pkl'
                    if emb_file.exists():
                        with open(emb_file, 'rb') as f:
                            mm_emb_dict[feat_id] = pickle.load(f)
                else:
                    base_path = mm_path / f'emb_{feat_id}_{shape}'
                    if base_path.exists():
                        emb_dict = {}
                        for json_file in base_path.glob('*.json'):
                            with open(json_file, 'r', encoding='utf-8') as file:
                                for line in file:
                                    data_dict_origin = json.loads(line.strip())
                                    insert_emb = data_dict_origin['emb']
                                    if isinstance(insert_emb, list):
                                        insert_emb = np.array(insert_emb, dtype=np.float32)
                                    emb_dict[data_dict_origin['anonymous_cid']] = insert_emb
                        mm_emb_dict[feat_id] = emb_dict
            except Exception as e:
                print(f"加载特征 {feat_id} 时出错: {e}")
                
        return mm_emb_dict
    
    def define_feature_types(self):
        """定义特征类型"""
        self.feature_types = {
            'user_sparse': ['103', '104', '105', '109'],
            'item_sparse': ['100', '117', '111', '118', '101', '102', '119', '120', '114', '112', '121', '115', '122', '116'],
            'item_array': [],
            'user_array': ['106', '107', '108', '110'],
            'item_emb': list(self.mm_emb_dict.keys()) if self.mm_emb_dict else [],
            'user_continual': [],
            'item_continual': []
        }
        print("✓ 特征类型定义完成")
        
    def analyze_feature_dimensions(self) -> Dict[str, Any]:
        """分析特征维度"""
        print("正在分析特征维度...")
        
        analysis_result = {
            'feature_counts': {},
            'feature_dimensions': {},
            'feature_distributions': {},
            'sparse_feature_cardinalities': {},
            'embedding_dimensions': {},
            'array_feature_lengths': {}
        }
        
        # 分析稀疏特征基数
        if self.indexer and 'f' in self.indexer:
            for feat_type, feat_ids in self.feature_types.items():
                if 'sparse' in feat_type:
                    for feat_id in feat_ids:
                        if feat_id in self.indexer['f']:
                            cardinality = len(self.indexer['f'][feat_id])
                            analysis_result['sparse_feature_cardinalities'][feat_id] = cardinality
                            
        # 分析Embedding特征维度
        if self.mm_emb_dict:
            for feat_id, emb_dict in self.mm_emb_dict.items():
                if emb_dict:
                    # 获取第一个embedding的维度
                    first_emb = next(iter(emb_dict.values()))
                    if hasattr(first_emb, 'shape'):
                        analysis_result['embedding_dimensions'][feat_id] = first_emb.shape[0]
                    elif isinstance(first_emb, list):
                        analysis_result['embedding_dimensions'][feat_id] = len(first_emb)
                        
        # 分析数组特征长度
        if self.item_feat_dict:
            for feat_type, feat_ids in self.feature_types.items():
                if 'array' in feat_type:
                    for feat_id in feat_ids:
                        lengths = []
                        for item_id, item_feat in self.item_feat_dict.items():
                            if feat_id in item_feat:
                                feat_value = item_feat[feat_id]
                                if isinstance(feat_value, list):
                                    lengths.append(len(feat_value))
                        if lengths:
                            analysis_result['array_feature_lengths'][feat_id] = {
                                'min_length': min(lengths),
                                'max_length': max(lengths),
                                'avg_length': np.mean(lengths),
                                'std_length': np.std(lengths)
                            }
                            
        # 统计各类型特征数量
        for feat_type, feat_ids in self.feature_types.items():
            analysis_result['feature_counts'][feat_type] = len(feat_ids)
            
        print("✓ 特征维度分析完成")
        return analysis_result
    
    def generate_feature_report(self, analysis_result: Dict[str, Any]) -> str:
        """生成特征维度报告"""
        print("正在生成特征维度报告...")
        
        report = []
        report.append("=" * 80)
        report.append("物品和用户特征维度分析报告")
        report.append("=" * 80)
        report.append("")
        
        # 特征类型统计
        report.append("1. 特征类型统计")
        report.append("-" * 40)
        for feat_type, count in analysis_result['feature_counts'].items():
            report.append(f"{feat_type:20}: {count:3d} 个特征")
        report.append("")
        
        # 稀疏特征基数
        if analysis_result['sparse_feature_cardinalities']:
            report.append("2. 稀疏特征基数统计")
            report.append("-" * 40)
            for feat_id, cardinality in sorted(analysis_result['sparse_feature_cardinalities'].items()):
                report.append(f"特征 {feat_id:3s}: {cardinality:8d} 个唯一值")
            report.append("")
            
        # Embedding特征维度
        if analysis_result['embedding_dimensions']:
            report.append("3. Embedding特征维度")
            report.append("-" * 40)
            for feat_id, dim in analysis_result['embedding_dimensions'].items():
                report.append(f"特征 {feat_id:3s}: {dim:4d} 维")
            report.append("")
            
        # 数组特征长度统计
        if analysis_result['array_feature_lengths']:
            report.append("4. 数组特征长度统计")
            report.append("-" * 40)
            for feat_id, stats in analysis_result['array_feature_lengths'].items():
                report.append(f"特征 {feat_id:3s}: 长度范围 [{stats['min_length']:2d}, {stats['max_length']:2d}], "
                           f"平均 {stats['avg_length']:5.2f} ± {stats['std_length']:5.2f}")
            report.append("")
            
        # 特征分布统计
        if self.item_feat_dict:
            report.append("5. 特征覆盖率统计")
            report.append("-" * 40)
            total_items = len(self.item_feat_dict)
            for feat_type, feat_ids in self.feature_types.items():
                if feat_ids:
                    coverage_stats = []
                    for feat_id in feat_ids:
                        covered_items = sum(1 for item_feat in self.item_feat_dict.values() 
                                          if feat_id in item_feat)
                        coverage_rate = covered_items / total_items * 100
                        coverage_stats.append(f"{feat_id}:{coverage_rate:5.1f}%")
                    report.append(f"{feat_type:20}: {' '.join(coverage_stats)}")
            report.append("")
            
        report.append("=" * 80)
        report.append("报告生成完成")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_feature_report(self, report: str, output_path: str = "feature_dimension_report.txt"):
        """保存特征维度报告到文件"""
        output_file = Path(output_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✓ 特征维度报告已保存到: {output_file}")
        
    def generate_feature_summary_csv(self, analysis_result: Dict[str, Any], output_path: str = "feature_summary.csv"):
        """生成特征摘要CSV文件"""
        print("正在生成特征摘要CSV文件...")
        
        summary_data = []
        
        # 稀疏特征
        for feat_id, cardinality in analysis_result['sparse_feature_cardinalities'].items():
            summary_data.append({
                'feature_id': feat_id,
                'feature_type': 'sparse',
                'category': 'user' if feat_id in self.feature_types['user_sparse'] else 'item',
                'dimension': cardinality,
                'description': f'稀疏特征，基数: {cardinality}'
            })
            
        # Embedding特征
        for feat_id, dim in analysis_result['embedding_dimensions'].items():
            summary_data.append({
                'feature_id': feat_id,
                'feature_type': 'embedding',
                'category': 'item',
                'dimension': dim,
                'description': f'多模态Embedding，维度: {dim}'
            })
            
        # 数组特征
        for feat_id, stats in analysis_result['array_feature_lengths'].items():
            summary_data.append({
                'feature_id': feat_id,
                'feature_type': 'array',
                'category': 'user' if feat_id in self.feature_types['user_array'] else 'item',
                'dimension': int(stats['avg_length']),
                'description': f'数组特征，平均长度: {stats["avg_length"]:.1f}'
            })
            
        # 连续特征
        for feat_type in ['user_continual', 'item_continual']:
            for feat_id in self.feature_types[feat_type]:
                summary_data.append({
                    'feature_id': feat_id,
                    'feature_type': 'continual',
                    'category': 'user' if feat_type == 'user_continual' else 'item',
                    'dimension': 1,
                    'description': '连续数值特征'
                })
                
        # 转换为DataFrame并保存
        df = pd.DataFrame(summary_data)
        df = df.sort_values(['category', 'feature_type', 'feature_id'])
        
        output_file = Path(output_path)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✓ 特征摘要CSV文件已保存到: {output_file}")
        
        return df
    
    def visualize_feature_dimensions(self, analysis_result: Dict[str, Any], output_dir: str = "feature_visualizations"):
        """生成特征维度可视化图表"""
        print("正在生成特征维度可视化图表...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 特征类型分布饼图
        plt.figure(figsize=(10, 8))
        feat_counts = analysis_result['feature_counts']
        plt.pie(feat_counts.values(), labels=feat_counts.keys(), autopct='%1.1f%%', startangle=90)
        plt.title('特征类型分布')
        plt.savefig(output_dir / 'feature_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 稀疏特征基数柱状图
        if analysis_result['sparse_feature_cardinalities']:
            plt.figure(figsize=(12, 6))
            feat_ids = list(analysis_result['sparse_feature_cardinalities'].keys())
            cardinalities = list(analysis_result['sparse_feature_cardinalities'].values())
            
            colors = ['skyblue' if feat_id in self.feature_types['user_sparse'] else 'lightcoral' 
                     for feat_id in feat_ids]
            
            plt.bar(feat_ids, cardinalities, color=colors, alpha=0.7)
            plt.title('稀疏特征基数分布')
            plt.xlabel('特征ID')
            plt.ylabel('基数')
            plt.xticks(rotation=45)
            plt.legend(['用户特征', '物品特征'])
            plt.tight_layout()
            plt.savefig(output_dir / 'sparse_feature_cardinalities.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        # 3. Embedding特征维度图
        if analysis_result['embedding_dimensions']:
            plt.figure(figsize=(10, 6))
            feat_ids = list(analysis_result['embedding_dimensions'].keys())
            dimensions = list(analysis_result['embedding_dimensions'].values())
            
            plt.bar(feat_ids, dimensions, color='lightgreen', alpha=0.7)
            plt.title('Embedding特征维度')
            plt.xlabel('特征ID')
            plt.ylabel('维度')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'embedding_feature_dimensions.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        print(f"✓ 可视化图表已保存到: {output_dir}")
        
    def run_full_analysis(self, output_dir: str = "feature_analysis_output"):
        """运行完整的特征分析流程"""
        print("开始运行完整的特征分析流程...")
        
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 定义特征类型
        self.define_feature_types()
        
        # 3. 分析特征维度
        analysis_result = self.analyze_feature_dimensions()
        
        # 4. 生成报告
        report = self.generate_feature_report(analysis_result)
        
        # 5. 保存报告
        self.save_feature_report(report, output_dir / "feature_dimension_report.txt")
        
        # 6. 生成CSV摘要
        summary_df = self.generate_feature_summary_csv(analysis_result, output_dir / "feature_summary.csv")
        
        # 7. 生成可视化图表
        self.visualize_feature_dimensions(analysis_result, output_dir / "visualizations")
        
        print(f"\n🎉 特征分析完成！所有结果已保存到: {output_dir}")
        print(f"📊 报告文件: {output_dir / 'feature_dimension_report.txt'}")
        print(f"📋 摘要文件: {output_dir / 'feature_summary.csv'}")
        print(f"📈 可视化图表: {output_dir / 'visualizations'}")
        
        return analysis_result, summary_df


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='物品和用户特征维度生成器')
    parser.add_argument('--data_dir', type=str, required=True, help='数据文件目录路径')
    parser.add_argument('--output_dir', type=str, default='feature_analysis_output', help='输出目录路径')
    
    args = parser.parse_args()
    
    # 创建特征维度生成器
    generator = FeatureDimensionGenerator(args.data_dir)
    
    # 运行完整分析
    try:
        analysis_result, summary_df = generator.run_full_analysis(args.output_dir)
        print("\n分析结果预览:")
        print(summary_df.head(10))
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
