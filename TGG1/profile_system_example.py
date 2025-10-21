"""
用户画像系统使用示例
演示如何使用用户画像系统来生成和更新用户数组特征
"""

import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

from user_profile_system import UserProfileSystem, UserProfileMonitor
from profile_update_pipeline import ProfileUpdatePipeline, UpdateEvent

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_user_data(num_users: int = 100, 
                            actions_per_user: int = 50) -> Dict[int, List[Dict[str, Any]]]:
    """
    生成示例用户行为数据
    
    Args:
        num_users: 用户数量
        actions_per_user: 每个用户的行为数量
        
    Returns:
        用户行为数据字典
    """
    logger.info(f"生成 {num_users} 个用户的示例数据")
    
    user_data = {}
    base_time = int(time.time()) - 30 * 24 * 3600  # 30天前
    
    for user_id in range(1, num_users + 1):
        actions = []
        
        # 生成用户行为序列
        for i in range(actions_per_user):
            # 随机时间戳（最近30天内）
            timestamp = base_time + random.randint(0, 30 * 24 * 3600)
            
            # 随机物品ID
            item_id = random.randint(1000, 10000)
            
            # 随机行为类型
            action_type = random.choice([0, 1, 2])  # 0: 点击, 1: 浏览, 2: 购买
            
            # 随机停留时间
            dwell_time = random.randint(10, 300)
            
            action = {
                "item_id": item_id,
                "timestamp": timestamp,
                "action_type": action_type,
                "dwell_time": dwell_time,
                "device_type": random.choice(["mobile", "desktop", "tablet"]),
                "category": random.choice(["entertainment", "technology", "lifestyle", "business"])
            }
            
            actions.append(action)
        
        # 按时间排序
        actions.sort(key=lambda x: x["timestamp"])
        user_data[user_id] = actions
    
    logger.info(f"生成了 {sum(len(actions) for actions in user_data.values())} 个行为记录")
    return user_data

def demo_basic_profile_system():
    """演示基础用户画像系统"""
    logger.info("=== 基础用户画像系统演示 ===")
    
    # 初始化系统
    profile_system = UserProfileSystem("data/")
    
    # 生成示例数据
    user_data = generate_sample_user_data(num_users=10, actions_per_user=20)
    
    # 批量更新用户画像
    logger.info("开始批量更新用户画像...")
    updated_profiles = profile_system.batch_update_profiles(user_data)
    
    # 显示结果
    logger.info(f"成功更新了 {len(updated_profiles)} 个用户画像")
    
    # 获取用户数组特征
    for user_id in range(1, 6):  # 显示前5个用户
        features = profile_system.get_user_array_features(user_id)
        logger.info(f"用户 {user_id} 的数组特征:")
        logger.info(f"  特征106 (兴趣标签): {features['106']}")
        logger.info(f"  特征107 (行为偏好): {features['107']}")
        logger.info(f"  特征108 (活跃度等级): {features['108']}")
        logger.info(f"  特征110 (用户类型): {features['110']}")
    
    # 获取系统统计
    stats = profile_system.get_profile_statistics()
    logger.info(f"系统统计: {stats}")
    
    # 保存画像
    profile_system.save_profiles("output/profiles/")
    logger.info("用户画像已保存")
    
    return profile_system

def demo_real_time_updates():
    """演示实时更新功能"""
    logger.info("=== 实时更新功能演示 ===")
    
    # 初始化流水线
    pipeline = ProfileUpdatePipeline("data/")
    
    # 启动流水线
    pipeline.start()
    
    try:
        # 模拟实时用户行为
        logger.info("开始模拟实时用户行为...")
        
        for i in range(100):
            user_id = random.randint(1, 10)
            item_id = random.randint(1000, 10000)
            
            # 添加用户事件
            pipeline.add_user_event(
                user_id=user_id,
                item_id=item_id,
                event_type="click",
                additional_data={
                    "dwell_time": random.randint(10, 300),
                    "device_type": random.choice(["mobile", "desktop"]),
                    "category": random.choice(["entertainment", "technology"])
                }
            )
            
            # 每10个事件显示一次统计
            if (i + 1) % 10 == 0:
                stats = pipeline.get_system_stats()
                logger.info(f"已处理 {i + 1} 个事件，队列大小: {stats['real_time_updater']['queue_size']}")
            
            time.sleep(0.1)  # 模拟实时数据流
        
        # 等待处理完成
        logger.info("等待实时处理完成...")
        time.sleep(5)
        
        # 获取用户特征
        for user_id in range(1, 4):
            features = pipeline.get_user_features(user_id)
            logger.info(f"用户 {user_id} 的实时特征:")
            logger.info(f"  特征106: {features['106']}")
            logger.info(f"  特征107: {features['107']}")
            logger.info(f"  特征108: {features['108']}")
            logger.info(f"  特征110: {features['110']}")
        
        # 生成报告
        report = pipeline.generate_report()
        logger.info("系统报告:")
        logger.info(report)
        
    finally:
        # 停止流水线
        pipeline.stop()

def demo_profile_monitoring():
    """演示画像监控功能"""
    logger.info("=== 画像监控功能演示 ===")
    
    # 初始化系统
    profile_system = UserProfileSystem("data/")
    
    # 生成数据并更新画像
    user_data = generate_sample_user_data(num_users=50, actions_per_user=30)
    profile_system.batch_update_profiles(user_data)
    
    # 初始化监控器
    monitor = UserProfileMonitor(profile_system)
    
    # 监控画像质量
    quality_metrics = monitor.monitor_profile_quality()
    logger.info("画像质量指标:")
    for metric, value in quality_metrics.items():
        logger.info(f"  {metric}: {value:.3f}")
    
    # 生成监控报告
    report = monitor.generate_report()
    logger.info("监控报告:")
    logger.info(report)

def demo_feature_evolution():
    """演示特征演化过程"""
    logger.info("=== 特征演化过程演示 ===")
    
    # 初始化系统
    profile_system = UserProfileSystem("data/")
    
    user_id = 1
    
    # 模拟用户行为的时间演化
    base_time = int(time.time()) - 30 * 24 * 3600
    
    for week in range(4):  # 4周的数据
        logger.info(f"--- 第 {week + 1} 周 ---")
        
        # 生成一周的行为数据
        week_actions = []
        for day in range(7):
            day_start = base_time + (week * 7 + day) * 24 * 3600
            
            # 每天3-10个行为
            num_actions = random.randint(3, 10)
            for _ in range(num_actions):
                timestamp = day_start + random.randint(0, 24 * 3600)
                item_id = random.randint(1000, 10000)
                
                action = {
                    "item_id": item_id,
                    "timestamp": timestamp,
                    "action_type": 0,
                    "dwell_time": random.randint(10, 300)
                }
                week_actions.append(action)
        
        # 更新用户画像
        profile = profile_system.update_user_profile(user_id, week_actions)
        
        # 获取特征
        features = profile_system.get_user_array_features(user_id)
        
        logger.info(f"用户 {user_id} 第 {week + 1} 周后的特征:")
        logger.info(f"  兴趣标签数量: {len(features['106'])}")
        logger.info(f"  行为偏好数量: {len(features['107'])}")
        logger.info(f"  活跃度等级: {features['108'][0]}")
        logger.info(f"  用户类型: {features['110'][0]}")
        logger.info(f"  置信度: {profile.confidence_score:.3f}")
        logger.info(f"  数据点: {profile.data_points}")

def demo_batch_processing():
    """演示批处理功能"""
    logger.info("=== 批处理功能演示 ===")
    
    # 初始化系统
    profile_system = UserProfileSystem("data/")
    
    # 生成大量用户数据
    logger.info("生成大量用户数据...")
    user_data = generate_sample_user_data(num_users=1000, actions_per_user=100)
    
    # 分批处理
    batch_size = 100
    total_users = len(user_data)
    
    logger.info(f"开始分批处理 {total_users} 个用户，每批 {batch_size} 个")
    
    start_time = time.time()
    
    for i in range(0, total_users, batch_size):
        batch_data = dict(list(user_data.items())[i:i + batch_size])
        
        # 处理当前批次
        updated_profiles = profile_system.batch_update_profiles(batch_data)
        
        # 显示进度
        processed = min(i + batch_size, total_users)
        logger.info(f"已处理 {processed}/{total_users} 个用户")
    
    end_time = time.time()
    
    # 显示结果
    stats = profile_system.get_profile_statistics()
    logger.info(f"批处理完成，耗时: {end_time - start_time:.2f} 秒")
    logger.info(f"处理了 {stats['total_users']} 个用户画像")
    logger.info(f"平均置信度: {stats['avg_confidence_score']:.3f}")
    
    # 保存结果
    profile_system.save_profiles("output/profiles/")
    logger.info("批处理结果已保存")

def main():
    """主函数"""
    logger.info("开始用户画像系统演示")
    
    try:
        # 1. 基础功能演示
        demo_basic_profile_system()
        
        # 2. 实时更新演示
        demo_real_time_updates()
        
        # 3. 监控功能演示
        demo_profile_monitoring()
        
        # 4. 特征演化演示
        demo_feature_evolution()
        
        # 5. 批处理演示
        demo_batch_processing()
        
        logger.info("所有演示完成")
        
    except Exception as e:
        logger.error(f"演示过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()


