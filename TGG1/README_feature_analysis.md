# 物品和用户特征维度生成器

这个项目提供了两个脚本来分析和生成物品、用户特征的维度信息，帮助您了解数据集中特征的结构和分布。

## 文件说明

### 1. `feature_dimension_generator.py` - 完整版特征分析器
功能最全面的特征分析脚本，包含：
- 详细的特征维度分析
- 生成完整的分析报告
- 生成CSV格式的特征摘要
- 生成可视化图表
- 支持命令行参数

### 2. `quick_feature_analyzer.py` - 快速特征分析器
简化版本，用于快速获取特征维度信息：
- 快速分析特征维度
- 生成文本格式的摘要
- 轻量级，运行速度快

## 安装依赖

```bash
pip install numpy pandas matplotlib seaborn
```

## 使用方法

### 快速特征分析器（推荐新手使用）

```bash
# 基本用法
python quick_feature_analyzer.py <数据目录路径>

# 指定输出文件
python quick_feature_analyzer.py ./data ./my_feature_summary.txt

# 示例
python quick_feature_analyzer.py ./dataset
```

### 完整版特征分析器

```bash
# 基本用法
python feature_dimension_generator.py --data_dir <数据目录路径>

# 指定输出目录
python feature_dimension_generator.py --data_dir ./data --output_dir ./my_analysis

# 示例
python feature_dimension_generator.py --data_dir ./dataset --output_dir ./feature_analysis
```

## 数据目录结构要求

脚本期望的数据目录结构：
```
data_directory/
├── item_feat_dict.json      # 物品特征字典
├── indexer.pkl              # 索引文件
└── creative_emb/            # 多模态特征目录（可选）
    ├── emb_81_32.pkl
    ├── emb_82_1024/
    ├── emb_83_3584/
    └── ...
```

## 输出结果

### 快速分析器输出
- 文本格式的特征维度摘要
- 包含特征类型、基数、维度、覆盖率等信息

### 完整分析器输出
- `feature_dimension_report.txt` - 详细的分析报告
- `feature_summary.csv` - CSV格式的特征摘要
- `visualizations/` - 可视化图表目录
  - `feature_type_distribution.png` - 特征类型分布饼图
  - `sparse_feature_cardinalities.png` - 稀疏特征基数柱状图
  - `embedding_feature_dimensions.png` - Embedding特征维度图

## 特征类型说明

脚本支持以下特征类型：

### 用户特征
- **稀疏特征** (`user_sparse`): 103, 104, 105, 109
- **数组特征** (`user_array`): 106, 107, 108, 110
- **连续特征** (`user_continual`): 暂无

### 物品特征
- **稀疏特征** (`item_sparse`): 100, 117, 111, 118, 101, 102, 119, 120, 114, 112, 121, 115, 122, 116
- **数组特征** (`item_array`): 暂无
- **Embedding特征** (`item_emb`): 81, 82, 83, 84, 85, 86
- **连续特征** (`item_continual`): 暂无

## 分析内容

### 1. 特征类型统计
统计各类型特征的数量

### 2. 稀疏特征基数
分析稀疏特征的唯一值数量

### 3. Embedding特征维度
分析多模态特征的向量维度

### 4. 数组特征长度
分析数组特征的长度分布

### 5. 特征覆盖率
分析各特征在数据集中的覆盖情况

## 示例输出

```
============================================================
特征维度摘要
============================================================

1. 特征类型统计
------------------------------
user_sparse         :   4 个特征
item_sparse         :  14 个特征
item_array          :   0 个特征
user_array          :   4 个特征
item_emb            :   6 个特征
user_continual      :   0 个特征
item_continual      :   0 个特征

2. 稀疏特征基数
------------------------------
特征 100 (物品):  1000000 个唯一值
特征 101 (物品):   500000 个唯一值
特征 103 (用户):    10000 个唯一值
...

3. Embedding特征维度
------------------------------
特征  81:   32 维
特征  82: 1024 维
特征  83: 3584 维
...
```

## 常见问题

### Q: 脚本运行时提示文件不存在
A: 请检查数据目录路径是否正确，确保包含必要的文件

### Q: 如何修改特征类型定义
A: 编辑脚本中的 `define_feature_types()` 函数，修改特征ID列表

### Q: 输出文件保存位置
A: 默认保存在当前目录，可通过参数指定输出路径

### Q: 支持哪些数据格式
A: 目前支持JSON和Pickle格式，可根据需要扩展

## 扩展功能

如需添加新的特征类型或分析维度，可以：

1. 修改 `define_feature_types()` 函数添加新特征类型
2. 在 `analyze_feature_dimensions()` 函数中添加新的分析逻辑
3. 在 `generate_dimension_summary()` 函数中添加新的输出格式

## 技术支持

如有问题或建议，请检查：
1. Python版本（推荐3.7+）
2. 依赖包是否正确安装
3. 数据文件格式是否符合要求
4. 文件路径是否正确
