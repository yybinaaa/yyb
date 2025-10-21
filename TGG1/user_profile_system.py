"""
用户画像系统 - 持续更新用户数组特征
用于生成和更新用户数组特征 106-108、110
"""

import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import logging
from dataclasses import dataclass
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """用户画像数据结构"""
    user_id: int
    # 特征106: 用户兴趣标签数组 (基于行为序列计算)
    interest_tags: List[int]
    # 特征107: 用户行为偏好数组 (基于交互模式计算)
    behavior_preferences: List[int]
    # 特征108: 用户活跃度等级 (0-3, 4个等级)
    activity_level: int
    # 特征110: 用户类型 (0-1, 2个类型)
    user_type: int
    # 元数据
    last_update: datetime
    confidence_score: float
    data_points: int

class UserProfileSystem:
    """用户画像系统主类"""
    
    def __init__(self, data_path: str, config: Dict[str, Any] = None):
        """
        初始化用户画像系统
        
        Args:
            data_path: 数据路径
            config: 配置参数
        """
        self.data_path = Path(data_path)
        self.config = config or self._default_config()
        
        # 加载数据
        self.indexer = self._load_indexer()
        self.item_features = self._load_item_features()
        
        # 用户画像存储
        self.user_profiles: Dict[int, UserProfile] = {}
        
        # 特征映射
        self.feature_mappings = self._init_feature_mappings()
        
        logger.info("用户画像系统初始化完成")
    
    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            # 特征106: 兴趣标签配置
            "interest_tags": {
                "max_tags": 10,  # 最大标签数量
                "min_frequency": 3,  # 最小出现频次
                "decay_factor": 0.95,  # 时间衰减因子
                "categories": {
                    "entertainment": [100, 101, 102],  # 娱乐类物品特征
                    "technology": [111, 112, 113],
                    "lifestyle": [114, 115, 116],
                    "business": [117, 118, 119]
                }
            },
            
            # 特征107: 行为偏好配置
            "behavior_preferences": {
                "max_preferences": 8,  # 最大偏好数量
                "time_windows": [1, 7, 30],  # 时间窗口(天)
                "preference_types": {
                    "click_frequency": "点击频率偏好",
                    "dwell_time": "停留时间偏好", 
                    "interaction_pattern": "交互模式偏好",
                    "time_preference": "时间偏好"
                }
            },
            
            # 特征108: 活跃度等级配置
            "activity_levels": {
                "levels": 4,  # 4个等级
                "metrics": {
                    "daily_actions": [0, 5, 15, 30],  # 每日行为次数阈值
                    "session_duration": [0, 300, 900, 1800],  # 会话时长阈值(秒)
                    "frequency": [0, 3, 7, 14]  # 访问频率阈值(天)
                }
            },
            
            # 特征110: 用户类型配置
            "user_types": {
                "types": 2,  # 2个类型
                "criteria": {
                    "type_0": {"min_sessions": 1, "max_actions": 10},  # 轻度用户
                    "type_1": {"min_sessions": 5, "min_actions": 20}   # 重度用户
                }
            }
        }
    
    def _load_indexer(self) -> Dict[str, Any]:
        """加载特征索引"""
        try:
            with open(self.data_path / "indexer.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            logger.warning("未找到indexer.pkl，使用空索引")
            return {"f": {}}
    
    def _load_item_features(self) -> Dict[int, Dict[str, int]]:
        """加载物品特征"""
        try:
            with open(self.data_path / "item_feat_dict.json", "r") as f:
                return {int(k): v for k, v in json.load(f).items()}
        except FileNotFoundError:
            logger.warning("未找到item_feat_dict.json，使用空特征")
            return {}
    
    def _init_feature_mappings(self) -> Dict[str, Dict[int, int]]:
        """初始化特征映射"""
        mappings = {}
        
        # 特征106映射: 兴趣标签
        mappings["106"] = {}
        for i, tag in enumerate(self.config["interest_tags"]["categories"].keys()):
            mappings["106"][tag] = 1060000000 + i
        
        # 特征107映射: 行为偏好
        mappings["107"] = {}
        for i, pref in enumerate(self.config["behavior_preferences"]["preference_types"].keys()):
            mappings["107"][pref] = 1070000000 + i
        
        # 特征108映射: 活跃度等级
        mappings["108"] = {i: 1080000000 + i for i in range(4)}
        
        # 特征110映射: 用户类型
        mappings["110"] = {i: 1100000000 + i for i in range(2)}
        
        return mappings
    
    def extract_user_behavior_features(self, user_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        从用户行为序列中提取特征
        
        Args:
            user_sequence: 用户行为序列
            
        Returns:
            提取的特征字典
        """
        if not user_sequence:
            return self._get_default_features()
        
        # 计算基础统计
        total_actions = len(user_sequence)
        unique_items = len(set(record.get("item_id", 0) for record in user_sequence))
        
        # 时间分析
        timestamps = [record.get("timestamp", 0) for record in user_sequence]
        time_span = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
        
        # 特征106: 兴趣标签提取
        interest_tags = self._extract_interest_tags(user_sequence)
        
        # 特征107: 行为偏好提取
        behavior_preferences = self._extract_behavior_preferences(user_sequence)
        
        # 特征108: 活跃度等级计算
        activity_level = self._calculate_activity_level(user_sequence, total_actions, time_span)
        
        # 特征110: 用户类型判断
        user_type = self._determine_user_type(user_sequence, total_actions)
        
        return {
            "interest_tags": interest_tags,
            "behavior_preferences": behavior_preferences,
            "activity_level": activity_level,
            "user_type": user_type,
            "confidence_score": min(1.0, total_actions / 50.0),  # 基于行为数量的置信度
            "data_points": total_actions
        }
    
    def _extract_interest_tags(self, user_sequence: List[Dict[str, Any]]) -> List[int]:
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
        
        return interest_tags if interest_tags else [1060000000]  # 默认标签
    
    def _extract_behavior_preferences(self, user_sequence: List[Dict[str, Any]]) -> List[int]:
        """提取用户行为偏好"""
        preferences = []
        
        # 点击频率偏好
        click_frequency = len(user_sequence) / max(1, len(set(record.get("timestamp", 0) // 86400 for record in user_sequence)))
        if click_frequency > 5:
            preferences.append(self.feature_mappings["107"]["click_frequency"])
        
        # 交互模式偏好 (基于行为序列的多样性)
        unique_items_ratio = len(set(record.get("item_id", 0) for record in user_sequence)) / max(1, len(user_sequence))
        if unique_items_ratio > 0.7:
            preferences.append(self.feature_mappings["107"]["interaction_pattern"])
        
        # 时间偏好 (基于活跃时间段)
        hour_counts = defaultdict(int)
        for record in user_sequence:
            timestamp = record.get("timestamp", 0)
            hour = datetime.fromtimestamp(timestamp).hour
            hour_counts[hour] += 1
        
        peak_hours = [h for h, c in hour_counts.items() if c > np.mean(list(hour_counts.values()))]
        if len(peak_hours) > 0:
            preferences.append(self.feature_mappings["107"]["time_preference"])
        
        return preferences if preferences else [1070000000]  # 默认偏好
    
    def _calculate_activity_level(self, user_sequence: List[Dict[str, Any]], 
                                total_actions: int, time_span: int) -> int:
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
        
        return len(thresholds) - 1  # 最高等级
    
    def _determine_user_type(self, user_sequence: List[Dict[str, Any]], total_actions: int) -> int:
        """判断用户类型"""
        # 计算会话数 (基于时间间隔)
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
    
    def _get_default_features(self) -> Dict[str, Any]:
        """获取默认特征"""
        return {
            "interest_tags": [1060000000],
            "behavior_preferences": [1070000000],
            "activity_level": 0,
            "user_type": 0,
            "confidence_score": 0.0,
            "data_points": 0
        }
    
    def update_user_profile(self, user_id: int, new_sequence: List[Dict[str, Any]]) -> UserProfile:
        """
        更新用户画像
        
        Args:
            user_id: 用户ID
            new_sequence: 新的行为序列
            
        Returns:
            更新后的用户画像
        """
        # 提取新特征
        new_features = self.extract_user_behavior_features(new_sequence)
        
        # 获取现有画像
        existing_profile = self.user_profiles.get(user_id)
        
        if existing_profile is None:
            # 创建新画像
            profile = UserProfile(
                user_id=user_id,
                interest_tags=new_features["interest_tags"],
                behavior_preferences=new_features["behavior_preferences"],
                activity_level=new_features["activity_level"],
                user_type=new_features["user_type"],
                last_update=datetime.now(),
                confidence_score=new_features["confidence_score"],
                data_points=new_features["data_points"]
            )
        else:
            # 更新现有画像 (加权平均)
            decay_factor = self.config["interest_tags"]["decay_factor"]
            time_diff = (datetime.now() - existing_profile.last_update).days
            
            # 时间衰减
            weight = decay_factor ** time_diff
            
            # 合并兴趣标签
            combined_tags = self._merge_interest_tags(
                existing_profile.interest_tags, 
                new_features["interest_tags"], 
                weight
            )
            
            # 合并行为偏好
            combined_preferences = self._merge_behavior_preferences(
                existing_profile.behavior_preferences,
                new_features["behavior_preferences"],
                weight
            )
            
            # 更新其他特征
            profile = UserProfile(
                user_id=user_id,
                interest_tags=combined_tags,
                behavior_preferences=combined_preferences,
                activity_level=new_features["activity_level"],  # 活跃度使用最新值
                user_type=new_features["user_type"],  # 用户类型使用最新值
                last_update=datetime.now(),
                confidence_score=min(1.0, existing_profile.confidence_score + new_features["confidence_score"]),
                data_points=existing_profile.data_points + new_features["data_points"]
            )
        
        # 保存更新后的画像
        self.user_profiles[user_id] = profile
        
        logger.info(f"用户 {user_id} 画像已更新")
        return profile
    
    def _merge_interest_tags(self, existing_tags: List[int], new_tags: List[int], weight: float) -> List[int]:
        """合并兴趣标签"""
        # 简单的标签合并策略
        all_tags = existing_tags + new_tags
        tag_counts = Counter(all_tags)
        
        # 选择出现频次最高的标签
        max_tags = self.config["interest_tags"]["max_tags"]
        top_tags = [tag for tag, count in tag_counts.most_common(max_tags)]
        
        return top_tags if top_tags else [1060000000]
    
    def _merge_behavior_preferences(self, existing_prefs: List[int], new_prefs: List[int], weight: float) -> List[int]:
        """合并行为偏好"""
        # 简单的偏好合并策略
        all_prefs = existing_prefs + new_prefs
        pref_counts = Counter(all_prefs)
        
        # 选择出现频次最高的偏好
        max_prefs = self.config["behavior_preferences"]["max_preferences"]
        top_prefs = [pref for pref, count in pref_counts.most_common(max_prefs)]
        
        return top_prefs if top_prefs else [1070000000]
    
    def get_user_array_features(self, user_id: int) -> Dict[str, List[int]]:
        """
        获取用户的数组特征
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户数组特征字典
        """
        profile = self.user_profiles.get(user_id)
        if profile is None:
            return {
                "106": [1060000000],  # 默认兴趣标签
                "107": [1070000000],  # 默认行为偏好
                "108": [1080000000],  # 默认活跃度等级
                "110": [1100000000]   # 默认用户类型
            }
        
        return {
            "106": profile.interest_tags,
            "107": profile.behavior_preferences,
            "108": [self.feature_mappings["108"][profile.activity_level]],
            "110": [self.feature_mappings["110"][profile.user_type]]
        }
    
    def batch_update_profiles(self, user_sequences: Dict[int, List[Dict[str, Any]]]) -> Dict[int, UserProfile]:
        """
        批量更新用户画像
        
        Args:
            user_sequences: 用户ID到行为序列的映射
            
        Returns:
            更新后的用户画像字典
        """
        updated_profiles = {}
        
        for user_id, sequence in user_sequences.items():
            try:
                profile = self.update_user_profile(user_id, sequence)
                updated_profiles[user_id] = profile
            except Exception as e:
                logger.error(f"更新用户 {user_id} 画像失败: {e}")
        
        logger.info(f"批量更新完成，共更新 {len(updated_profiles)} 个用户画像")
        return updated_profiles
    
    def save_profiles(self, output_path: str):
        """保存用户画像到文件"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存为JSON格式
        profiles_data = {}
        for user_id, profile in self.user_profiles.items():
            profiles_data[user_id] = {
                "interest_tags": profile.interest_tags,
                "behavior_preferences": profile.behavior_preferences,
                "activity_level": profile.activity_level,
                "user_type": profile.user_type,
                "last_update": profile.last_update.isoformat(),
                "confidence_score": profile.confidence_score,
                "data_points": profile.data_points
            }
        
        with open(output_path / "user_profiles.json", "w") as f:
            json.dump(profiles_data, f, indent=2)
        
        # 保存为Pickle格式 (更高效)
        with open(output_path / "user_profiles.pkl", "wb") as f:
            pickle.dump(self.user_profiles, f)
        
        logger.info(f"用户画像已保存到 {output_path}")
    
    def load_profiles(self, input_path: str):
        """从文件加载用户画像"""
        input_path = Path(input_path)
        
        try:
            with open(input_path / "user_profiles.pkl", "rb") as f:
                self.user_profiles = pickle.load(f)
            logger.info(f"从 {input_path} 加载了 {len(self.user_profiles)} 个用户画像")
        except FileNotFoundError:
            logger.warning(f"未找到用户画像文件: {input_path}")
            self.user_profiles = {}
    
    def get_profile_statistics(self) -> Dict[str, Any]:
        """获取画像统计信息"""
        if not self.user_profiles:
            return {"total_users": 0}
        
        total_users = len(self.user_profiles)
        avg_confidence = np.mean([p.confidence_score for p in self.user_profiles.values()])
        avg_data_points = np.mean([p.data_points for p in self.user_profiles.values()])
        
        # 活跃度分布
        activity_dist = Counter(p.activity_level for p in self.user_profiles.values())
        
        # 用户类型分布
        type_dist = Counter(p.user_type for p in self.user_profiles.values())
        
        return {
            "total_users": total_users,
            "avg_confidence_score": avg_confidence,
            "avg_data_points": avg_data_points,
            "activity_level_distribution": dict(activity_dist),
            "user_type_distribution": dict(type_dist)
        }


class UserProfileMonitor:
    """用户画像监控系统"""
    
    def __init__(self, profile_system: UserProfileSystem):
        self.profile_system = profile_system
        self.metrics_history = []
    
    def monitor_profile_quality(self) -> Dict[str, Any]:
        """监控画像质量"""
        stats = self.profile_system.get_profile_statistics()
        
        # 质量指标
        quality_metrics = {
            "coverage_rate": stats["total_users"] / 10000,  # 假设总用户数
            "avg_confidence": stats["avg_confidence_score"],
            "data_richness": stats["avg_data_points"],
            "profile_freshness": self._calculate_freshness(),
            "feature_completeness": self._calculate_completeness()
        }
        
        self.metrics_history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": quality_metrics
        })
        
        return quality_metrics
    
    def _calculate_freshness(self) -> float:
        """计算画像新鲜度"""
        if not self.profile_system.user_profiles:
            return 0.0
        
        now = datetime.now()
        freshness_scores = []
        
        for profile in self.profile_system.user_profiles.values():
            days_old = (now - profile.last_update).days
            freshness = max(0, 1 - days_old / 30)  # 30天衰减
            freshness_scores.append(freshness)
        
        return np.mean(freshness_scores)
    
    def _calculate_completeness(self) -> float:
        """计算特征完整性"""
        if not self.profile_system.user_profiles:
            return 0.0
        
        completeness_scores = []
        
        for profile in self.profile_system.user_profiles.values():
            # 检查每个特征是否有非默认值
            has_interest = len(profile.interest_tags) > 1 or profile.interest_tags[0] != 1060000000
            has_preferences = len(profile.behavior_preferences) > 1 or profile.behavior_preferences[0] != 1070000000
            has_activity = profile.activity_level > 0
            has_type = profile.user_type >= 0
            
            completeness = sum([has_interest, has_preferences, has_activity, has_type]) / 4
            completeness_scores.append(completeness)
        
        return np.mean(completeness_scores)
    
    def generate_report(self) -> str:
        """生成监控报告"""
        metrics = self.monitor_profile_quality()
        
        report = f"""
用户画像系统监控报告
==================
时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

质量指标:
- 覆盖率: {metrics['coverage_rate']:.2%}
- 平均置信度: {metrics['avg_confidence']:.3f}
- 数据丰富度: {metrics['data_richness']:.1f}
- 画像新鲜度: {metrics['profile_freshness']:.2%}
- 特征完整性: {metrics['feature_completeness']:.2%}

建议:
"""
        
        if metrics['coverage_rate'] < 0.5:
            report += "- 覆盖率较低，建议增加数据收集\n"
        
        if metrics['avg_confidence'] < 0.3:
            report += "- 平均置信度较低，建议优化特征提取算法\n"
        
        if metrics['profile_freshness'] < 0.7:
            report += "- 画像新鲜度不足，建议增加更新频率\n"
        
        if metrics['feature_completeness'] < 0.8:
            report += "- 特征完整性不足，建议完善特征工程\n"
        
        return report


# 使用示例
if __name__ == "__main__":
    # 初始化系统
    profile_system = UserProfileSystem("data/")
    
    # 模拟用户行为数据
    sample_sequence = [
        {"item_id": 12345, "timestamp": 1640995200, "action_type": 0},
        {"item_id": 12346, "timestamp": 1640995800, "action_type": 0},
        {"item_id": 12347, "timestamp": 1640996400, "action_type": 0},
    ]
    
    # 更新用户画像
    profile = profile_system.update_user_profile(123, sample_sequence)
    
    # 获取数组特征
    array_features = profile_system.get_user_array_features(123)
    print("用户数组特征:", array_features)
    
    # 保存画像
    profile_system.save_profiles("output/profiles/")
    
    # 监控系统
    monitor = UserProfileMonitor(profile_system)
    report = monitor.generate_report()
    print(report)


