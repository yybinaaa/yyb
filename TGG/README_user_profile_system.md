# 用户画像系统 - 持续更新用户数组特征

## 系统概述

本系统是一个完整的用户画像解决方案，专门用于生成和持续更新用户数组特征（106-108、110）。系统采用实时更新和定时批处理相结合的方式，确保用户画像的准确性和时效性。

## 核心功能

### 1. 用户数组特征生成
- **特征106**: 用户兴趣标签数组（基于行为序列计算）
- **特征107**: 用户行为偏好数组（基于交互模式计算）
- **特征108**: 用户活跃度等级（0-3，4个等级）
- **特征110**: 用户类型（0-1，2个类型）

### 2. 实时更新机制
- 事件驱动的实时画像更新
- 批量处理优化性能
- 时间衰减算法保持画像新鲜度

### 3. 定时更新策略
- 每小时增量更新
- 每日深度更新
- 每周全量更新

### 4. 监控和告警
- 画像质量监控
- 系统健康检查
- 自动恢复机制

## 系统架构

```
用户画像系统
├── 数据层
│   ├── 用户行为数据
│   ├── 物品特征数据
│   └── 特征映射索引
├── 处理层
│   ├── 特征提取引擎
│   ├── 实时更新器
│   └── 定时更新器
├── 存储层
│   ├── 用户画像存储
│   ├── 缓存系统
│   └── 备份机制
└── 监控层
    ├── 质量监控
    ├── 性能监控
    └── 告警系统
```

## 文件结构

```
├── user_profile_system.py          # 核心用户画像系统
├── profile_update_pipeline.py      # 实时更新流水线
├── profile_config.yaml             # 系统配置文件
├── profile_system_example.py       # 使用示例
├── deploy_profile_system.py        # 部署和运维脚本
└── README_user_profile_system.md   # 本文档
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install numpy pandas pyyaml schedule

# 创建必要目录
mkdir -p data output/profiles output/reports logs
```

### 2. 基础使用

```python
from user_profile_system import UserProfileSystem

# 初始化系统
profile_system = UserProfileSystem("data/")

# 生成用户行为数据
user_sequence = [
    {"item_id": 12345, "timestamp": 1640995200, "action_type": 0},
    {"item_id": 12346, "timestamp": 1640995800, "action_type": 0},
]

# 更新用户画像
profile = profile_system.update_user_profile(123, user_sequence)

# 获取数组特征
features = profile_system.get_user_array_features(123)
print(features)
```

### 3. 实时更新

```python
from profile_update_pipeline import ProfileUpdatePipeline

# 初始化流水线
pipeline = ProfileUpdatePipeline("data/")

# 启动系统
pipeline.start()

# 添加用户事件
pipeline.add_user_event(
    user_id=123,
    item_id=12345,
    event_type="click",
    additional_data={"dwell_time": 30}
)

# 获取用户特征
features = pipeline.get_user_features(123)
```

### 4. 部署和运维

```bash
# 部署系统
python deploy_profile_system.py --action deploy --monitor

# 检查状态
python deploy_profile_system.py --action status

# 健康检查
python deploy_profile_system.py --action health

# 重启系统
python deploy_profile_system.py --action restart
```

## 配置说明

### 特征配置

```yaml
# 特征106: 兴趣标签配置
interest_tags:
  max_tags: 10                    # 最大标签数量
  min_frequency: 3                # 最小出现频次
  decay_factor: 0.95              # 时间衰减因子
  categories:
    entertainment: [100, 101, 102]  # 娱乐类物品特征
    technology: [111, 112, 113]     # 科技类物品特征

# 特征107: 行为偏好配置
behavior_preferences:
  max_preferences: 8              # 最大偏好数量
  time_windows: [1, 7, 30]        # 时间窗口(天)
  preference_types:
    click_frequency: "点击频率偏好"
    dwell_time: "停留时间偏好"

# 特征108: 活跃度等级配置
activity_levels:
  levels: 4                       # 4个等级
  metrics:
    daily_actions: [0, 5, 15, 30]     # 每日行为次数阈值
    session_duration: [0, 300, 900, 1800]  # 会话时长阈值(秒)

# 特征110: 用户类型配置
user_types:
  types: 2                        # 2个类型
  criteria:
    type_0:                       # 轻度用户
      min_sessions: 1
      max_actions: 10
    type_1:                       # 重度用户
      min_sessions: 5
      min_actions: 20
```

### 更新配置

```yaml
# 实时更新配置
real_time_update:
  update_interval: 300            # 更新间隔(秒)
  batch_size: 1000               # 批处理大小
  queue_size: 10000              # 事件队列大小

# 定时更新配置
scheduled_update:
  hourly_update: true             # 是否启用每小时更新
  daily_deep_update: "02:00"      # 每日深度更新时间
  weekly_full_update: "sunday:03:00"  # 每周全量更新时间
```

## 特征生成算法

### 1. 兴趣标签提取（特征106）

```python
def _extract_interest_tags(self, user_sequence):
    """提取用户兴趣标签"""
    item_category_counts = defaultdict(int)
    
    for record in user_sequence:
        item_id = record.get("item_id")
        if item_id and item_id in self.item_features:
            item_feat = self.item_features[item_id]
            
            # 根据物品特征判断类别
            for category, feature_ids in self.config["interest_tags"]["categories"].items():
                if any(str(fid) in item_feat for fid in feature_ids):
                    item_category_counts[category] += 1
    
    # 选择top-k兴趣标签
    top_categories = sorted(item_category_counts.items(), 
                          key=lambda x: x[1], reverse=True)
    
    interest_tags = []
    for category, count in top_categories[:self.config["interest_tags"]["max_tags"]]:
        if count >= self.config["interest_tags"]["min_frequency"]:
            if category in self.feature_mappings["106"]:
                interest_tags.append(self.feature_mappings["106"][category])
    
    return interest_tags if interest_tags else [1060000000]
```

### 2. 行为偏好提取（特征107）

```python
def _extract_behavior_preferences(self, user_sequence):
    """提取用户行为偏好"""
    preferences = []
    
    # 点击频率偏好
    click_frequency = len(user_sequence) / max(1, len(set(record.get("timestamp", 0) // 86400 for record in user_sequence)))
    if click_frequency > 5:
        preferences.append(self.feature_mappings["107"]["click_frequency"])
    
    # 交互模式偏好
    unique_items_ratio = len(set(record.get("item_id", 0) for record in user_sequence)) / max(1, len(user_sequence))
    if unique_items_ratio > 0.7:
        preferences.append(self.feature_mappings["107"]["interaction_pattern"])
    
    return preferences if preferences else [1070000000]
```

### 3. 活跃度等级计算（特征108）

```python
def _calculate_activity_level(self, user_sequence, total_actions, time_span):
    """计算用户活跃度等级"""
    if time_span == 0:
        return 0
    
    # 计算每日平均行为数
    daily_actions = total_actions / max(1, time_span / 86400)
    
    # 根据阈值确定等级
    thresholds = self.config["activity_levels"]["metrics"]["daily_actions"]
    for level, threshold in enumerate(thresholds):
        if daily_actions <= threshold:
            return level
    
    return len(thresholds) - 1
```

### 4. 用户类型判断（特征110）

```python
def _determine_user_type(self, user_sequence, total_actions):
    """判断用户类型"""
    # 计算会话数
    timestamps = sorted([record.get("timestamp", 0) for record in user_sequence])
    sessions = 1
    for i in range(1, len(timestamps)):
        if timestamps[i] - timestamps[i-1] > 3600:  # 1小时间隔算新会话
            sessions += 1
    
    # 根据配置判断用户类型
    criteria = self.config["user_types"]["criteria"]
    if (sessions >= criteria["type_1"]["min_sessions"] and 
        total_actions >= criteria["type_1"]["min_actions"]):
        return 1  # 重度用户
    else:
        return 0  # 轻度用户
```

## 监控指标

### 1. 画像质量指标

- **覆盖率**: 有画像的用户占总用户的比例
- **平均置信度**: 画像的平均置信度分数
- **数据丰富度**: 平均每个用户的行为数据点数
- **画像新鲜度**: 画像的时间新鲜度
- **特征完整性**: 特征的非默认值比例

### 2. 系统性能指标

- **事件处理速度**: 每秒处理的事件数
- **画像更新延迟**: 从事件到画像更新的时间
- **队列大小**: 待处理事件队列长度
- **错误率**: 处理失败的比例
- **内存使用**: 系统内存占用

### 3. 业务指标

- **用户活跃度分布**: 各活跃度等级的用户分布
- **用户类型分布**: 各用户类型的分布
- **兴趣标签热度**: 热门兴趣标签统计
- **行为偏好趋势**: 行为偏好的变化趋势

## 部署指南

### 1. 单机部署

```bash
# 1. 准备环境
pip install -r requirements.txt

# 2. 配置系统
cp profile_config.yaml.example profile_config.yaml
# 编辑配置文件

# 3. 部署系统
python deploy_profile_system.py --action deploy

# 4. 启动监控
python deploy_profile_system.py --action deploy --monitor
```

### 2. 分布式部署

```bash
# 1. 部署多个实例
python deploy_profile_system.py --action deploy --instance-id 1
python deploy_profile_system.py --action deploy --instance-id 2

# 2. 配置负载均衡
# 使用Nginx或HAProxy进行负载均衡

# 3. 配置数据同步
# 使用Redis或数据库进行数据同步
```

### 3. 容器化部署

```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8080

CMD ["python", "deploy_profile_system.py", "--action", "deploy", "--monitor"]
```

## 运维指南

### 1. 日常运维

```bash
# 检查系统状态
python deploy_profile_system.py --action status

# 健康检查
python deploy_profile_system.py --action health

# 查看日志
tail -f profile_system.log

# 重启系统
python deploy_profile_system.py --action restart
```

### 2. 故障处理

```bash
# 查看错误日志
grep ERROR profile_system.log

# 检查系统资源
top -p $(pgrep -f profile_system)

# 清理缓存
rm -rf output/profiles/cache/*

# 恢复备份
cp output/profiles/backup/* output/profiles/
```

### 3. 性能优化

```yaml
# 调整批处理大小
real_time_update:
  batch_size: 2000  # 增加批处理大小

# 调整更新频率
real_time_update:
  update_interval: 180  # 减少更新间隔

# 调整缓存大小
performance:
  memory:
    max_cache_size: 20000  # 增加缓存大小
```

## 扩展开发

### 1. 添加新特征

```python
# 1. 在配置中添加新特征
new_feature:
  feature_id: "111"
  max_values: 5
  calculation_method: "custom"

# 2. 实现特征提取逻辑
def _extract_new_feature(self, user_sequence):
    # 自定义特征提取逻辑
    pass

# 3. 更新特征映射
mappings["111"] = {i: 1110000000 + i for i in range(5)}
```

### 2. 自定义更新策略

```python
class CustomUpdateStrategy:
    def should_update(self, user_id, new_events):
        # 自定义更新条件
        return len(new_events) > 10
    
    def calculate_priority(self, user_id, events):
        # 自定义优先级计算
        return len(events) * 0.5
```

### 3. 集成外部数据源

```python
class ExternalDataSource:
    def fetch_user_data(self, user_id):
        # 从外部系统获取用户数据
        pass
    
    def fetch_item_features(self, item_id):
        # 从外部系统获取物品特征
        pass
```

## 常见问题

### Q1: 如何提高画像质量？

A1: 
- 增加用户行为数据量
- 优化特征提取算法
- 调整时间衰减参数
- 增加特征维度

### Q2: 如何处理冷启动用户？

A2:
- 使用默认特征值
- 基于相似用户画像
- 利用物品特征进行推荐
- 快速收集用户行为数据

### Q3: 如何保证系统性能？

A3:
- 使用批处理优化
- 合理设置缓存大小
- 监控系统资源使用
- 定期清理过期数据

### Q4: 如何扩展系统？

A4:
- 使用分布式架构
- 添加新的特征类型
- 集成更多数据源
- 优化算法性能

## 联系支持

如有问题或建议，请联系开发团队或提交Issue。

---

**版本**: 1.0.0  
**更新时间**: 2024-01-01  
**维护团队**: 用户画像系统开发组


