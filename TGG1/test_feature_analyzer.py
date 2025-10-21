#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征分析器测试脚本

用于测试特征分析器的基本功能，生成模拟数据进行测试。
"""

import json
import pickle
import numpy as np
from pathlib import Path
import tempfile
import shutil


def create_test_data(test_dir: Path):
    """创建测试数据"""
    print("正在创建测试数据...")
    
    # 创建测试目录
    test_dir.mkdir(exist_ok=True)
    
    # 1. 创建物品特征字典
    item_feat_dict = {}
    for i in range(1000):  # 1000个物品
        item_feat_dict[str(i)] = {
            '100': np.random.randint(0, 1000),  # 稀疏特征
            '101': np.random.randint(0, 500),
            '102': np.random.randint(0, 200),
            '117': np.random.randint(0, 100),
            '111': np.random.randint(0, 300),
            '118': np.random.randint(0, 150),
            '119': np.random.randint(0, 400),
            '120': np.random.randint(0, 250),
            '114': np.random.randint(0, 600),
            '112': np.random.randint(0, 350),
            '121': np.random.randint(0, 450),
            '115': np.random.randint(0, 280),
            '122': np.random.randint(0, 320),
            '116': np.random.randint(0, 180)
        }
    
    # 保存物品特征字典
    with open(test_dir / "item_feat_dict.json", 'w', encoding='utf-8') as f:
        json.dump(item_feat_dict, f, ensure_ascii=False, indent=2)
    
    # 2. 创建索引文件
    indexer = {
        'i': {f'item_{i}': i for i in range(1000)},  # 物品索引
        'u': {f'user_{i}': i for i in range(100)},   # 用户索引
        'f': {  # 特征索引
            '100': {f'feat_100_{i}': i for i in range(1000)},
            '101': {f'feat_101_{i}': i for i in range(500)},
            '102': {f'feat_102_{i}': i for i in range(200)},
            '103': {f'feat_103_{i}': i for i in range(100)},
            '104': {f'feat_104_{i}': i for i in range(80)},
            '105': {f'feat_105_{i}': i for i in range(120)},
            '106': {f'feat_106_{i}': i for i in range(50)},
            '107': {f'feat_107_{i}': i for i in range(60)},
            '108': {f'feat_108_{i}': i for i in range(40)},
            '109': {f'feat_109_{i}': i for i in range(90)},
            '110': {f'feat_110_{i}': i for i in range(70)},
            '111': {f'feat_111_{i}': i for i in range(300)},
            '112': {f'feat_112_{i}': i for i in range(350)},
            '113': {f'feat_113_{i}': i for i in range(400)},
            '114': {f'feat_114_{i}': i for i in range(600)},
            '115': {f'feat_115_{i}': i for i in range(280)},
            '116': {f'feat_116_{i}': i for i in range(180)},
            '117': {f'feat_117_{i}': i for i in range(100)},
            '118': {f'feat_118_{i}': i for i in range(150)},
            '119': {f'feat_119_{i}': i for i in range(400)},
            '120': {f'feat_120_{i}': i for i in range(250)},
            '121': {f'feat_121_{i}': i for i in range(450)},
            '122': {f'feat_122_{i}': i for i in range(320)}
        }
    }
    
    # 保存索引文件
    with open(test_dir / "indexer.pkl", 'wb') as f:
        pickle.dump(indexer, f)
    
    # 3. 创建多模态特征目录（模拟）
    mm_dir = test_dir / "creative_emb"
    mm_dir.mkdir(exist_ok=True)
    
    # 创建一些模拟的embedding文件
    embedding_shapes = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    
    for feat_id, shape in embedding_shapes.items():
        if feat_id == '81':
            # 创建pickle格式的embedding
            emb_dict = {f'item_{i}': np.random.randn(shape).astype(np.float32) for i in range(100)}
            with open(mm_dir / f'emb_{feat_id}_{shape}.pkl', 'wb') as f:
                pickle.dump(emb_dict, f)
        else:
            # 创建json格式的embedding目录
            emb_subdir = mm_dir / f'emb_{feat_id}_{shape}'
            emb_subdir.mkdir(exist_ok=True)
            
            # 创建几个json文件
            for batch in range(3):
                emb_data = []
                for i in range(100):
                    emb_data.append({
                        'anonymous_cid': f'item_{batch * 100 + i}',
                        'emb': np.random.randn(shape).astype(np.float32).tolist()
                    })
                
                with open(emb_subdir / f'batch_{batch}.json', 'w', encoding='utf-8') as f:
                    for item in emb_data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✓ 测试数据已创建在: {test_dir}")
    return test_dir


def test_quick_analyzer(test_dir: Path):
    """测试快速特征分析器"""
    print("\n" + "="*50)
    print("测试快速特征分析器")
    print("="*50)
    
    try:
        # 导入快速分析器
        from quick_feature_analyzer import load_feature_data, define_feature_types, analyze_feature_dimensions, generate_dimension_summary
        
        # 加载数据
        data = load_feature_data(str(test_dir))
        
        # 定义特征类型
        feature_types = define_feature_types()
        
        # 分析特征维度
        analysis_result = analyze_feature_dimensions(data, feature_types)
        
        # 生成摘要
        summary = generate_dimension_summary(analysis_result, feature_types)
        
        print("✓ 快速特征分析器测试成功！")
        print("\n生成的摘要预览:")
        print(summary[:500] + "..." if len(summary) > 500 else summary)
        
        return True
        
    except Exception as e:
        print(f"✗ 快速特征分析器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_analyzer(test_dir: Path):
    """测试完整版特征分析器"""
    print("\n" + "="*50)
    print("测试完整版特征分析器")
    print("="*50)
    
    try:
        # 导入完整版分析器
        from feature_dimension_generator import FeatureDimensionGenerator
        
        # 创建分析器实例
        generator = FeatureDimensionGenerator(str(test_dir))
        
        # 运行分析
        analysis_result, summary_df = generator.run_full_analysis("test_output")
        
        print("✓ 完整版特征分析器测试成功！")
        print(f"输出目录: test_output")
        print(f"分析结果包含 {len(analysis_result)} 个部分")
        print(f"摘要DataFrame形状: {summary_df.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 完整版特征分析器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("开始测试特征分析器...")
    
    # 创建临时测试目录
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir) / "test_data"
        
        try:
            # 1. 创建测试数据
            create_test_data(test_dir)
            
            # 2. 测试快速分析器
            quick_success = test_quick_analyzer(test_dir)
            
            # 3. 测试完整版分析器
            full_success = test_full_analyzer(test_dir)
            
            # 4. 输出测试结果
            print("\n" + "="*50)
            print("测试结果总结")
            print("="*50)
            print(f"快速特征分析器: {'✓ 通过' if quick_success else '✗ 失败'}")
            print(f"完整版特征分析器: {'✓ 通过' if full_success else '✗ 失败'}")
            
            if quick_success and full_success:
                print("\n🎉 所有测试通过！特征分析器工作正常。")
            else:
                print("\n⚠ 部分测试失败，请检查错误信息。")
                
        except Exception as e:
            print(f"测试过程中出现错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
