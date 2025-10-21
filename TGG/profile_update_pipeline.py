"""
用户画像实时更新流水线
用于持续更新用户数组特征
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import threading
from queue import Queue, Empty
import schedule

from user_profile_system import UserProfileSystem, UserProfileMonitor

logger = logging.getLogger(__name__)

@dataclass
class UpdateEvent:
    """更新事件"""
    user_id: int
    event_type: str  # 'click', 'view', 'interaction'
    item_id: int
    timestamp: float
    additional_data: Dict[str, Any] = None

class RealTimeProfileUpdater:
    """实时用户画像更新器"""
    
    def __init__(self, profile_system: UserProfileSystem, 
                 update_interval: int = 300,  # 5分钟
                 batch_size: int = 1000):
        """
        初始化实时更新器
        
        Args:
            profile_system: 用户画像系统
            update_interval: 更新间隔(秒)
            batch_size: 批处理大小
        """
        self.profile_system = profile_system
        self.update_interval = update_interval
        self.batch_size = batch_size
        
        # 事件队列
        self.event_queue = Queue(maxsize=10000)
        
        # 用户行为缓存
        self.user_behavior_cache: Dict[int, List[Dict[str, Any]]] = {}
        
        # 更新状态
        self.is_running = False
        self.update_thread = None
        
        # 统计信息
        self.stats = {
            "events_processed": 0,
            "profiles_updated": 0,
            "last_update": None,
            "errors": 0
        }
        
        logger.info("实时用户画像更新器初始化完成")
    
    def add_event(self, event: UpdateEvent):
        """添加更新事件"""
        try:
            self.event_queue.put_nowait(event)
        except:
            logger.warning(f"事件队列已满，丢弃用户 {event.user_id} 的事件")
    
    def start(self):
        """启动实时更新器"""
        if self.is_running:
            logger.warning("实时更新器已在运行")
            return
        
        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        logger.info("实时用户画像更新器已启动")
    
    def stop(self):
        """停止实时更新器"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        
        logger.info("实时用户画像更新器已停止")
    
    def _update_loop(self):
        """更新循环"""
        while self.is_running:
            try:
                # 处理事件队列
                self._process_events()
                
                # 批量更新用户画像
                if len(self.user_behavior_cache) >= self.batch_size:
                    self._batch_update_profiles()
                
                time.sleep(1)  # 短暂休眠
                
            except Exception as e:
                logger.error(f"更新循环错误: {e}")
                self.stats["errors"] += 1
                time.sleep(5)
    
    def _process_events(self):
        """处理事件队列"""
        processed_count = 0
        
        while processed_count < 100:  # 每次最多处理100个事件
            try:
                event = self.event_queue.get_nowait()
                self._handle_event(event)
                processed_count += 1
                self.stats["events_processed"] += 1
                
            except Empty:
                break
            except Exception as e:
                logger.error(f"处理事件错误: {e}")
                self.stats["errors"] += 1
    
    def _handle_event(self, event: UpdateEvent):
        """处理单个事件"""
        user_id = event.user_id
        
        # 初始化用户行为缓存
        if user_id not in self.user_behavior_cache:
            self.user_behavior_cache[user_id] = []
        
        # 添加行为记录
        behavior_record = {
            "item_id": event.item_id,
            "timestamp": event.timestamp,
            "action_type": 0,  # 默认点击行为
            "event_type": event.event_type
        }
        
        if event.additional_data:
            behavior_record.update(event.additional_data)
        
        self.user_behavior_cache[user_id].append(behavior_record)
        
        # 限制缓存大小
        if len(self.user_behavior_cache[user_id]) > 1000:
            self.user_behavior_cache[user_id] = self.user_behavior_cache[user_id][-500:]
    
    def _batch_update_profiles(self):
        """批量更新用户画像"""
        if not self.user_behavior_cache:
            return
        
        try:
            # 更新用户画像
            updated_profiles = self.profile_system.batch_update_profiles(self.user_behavior_cache)
            
            # 更新统计
            self.stats["profiles_updated"] += len(updated_profiles)
            self.stats["last_update"] = datetime.now()
            
            # 清空缓存
            self.user_behavior_cache.clear()
            
            logger.info(f"批量更新完成，共更新 {len(updated_profiles)} 个用户画像")
            
        except Exception as e:
            logger.error(f"批量更新错误: {e}")
            self.stats["errors"] += 1
    
    def force_update(self):
        """强制更新所有缓存的用户画像"""
        if self.user_behavior_cache:
            self._batch_update_profiles()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "queue_size": self.event_queue.qsize(),
            "cache_size": len(self.user_behavior_cache),
            "is_running": self.is_running
        }


class ScheduledProfileUpdater:
    """定时用户画像更新器"""
    
    def __init__(self, profile_system: UserProfileSystem):
        self.profile_system = profile_system
        self.monitor = UserProfileMonitor(profile_system)
        
        # 设置定时任务
        self._setup_schedules()
        
        logger.info("定时用户画像更新器初始化完成")
    
    def _setup_schedules(self):
        """设置定时任务"""
        # 每小时更新一次用户画像
        schedule.every().hour.do(self._hourly_update)
        
        # 每天凌晨2点进行深度更新
        schedule.every().day.at("02:00").do(self._daily_deep_update)
        
        # 每周日凌晨3点进行全量更新
        schedule.every().sunday.at("03:00").do(self._weekly_full_update)
        
        # 每5分钟检查一次系统状态
        schedule.every(5).minutes.do(self._health_check)
    
    def _hourly_update(self):
        """每小时更新"""
        logger.info("开始每小时用户画像更新")
        
        try:
            # 这里可以添加从数据源获取新数据的逻辑
            # 例如从消息队列、数据库等获取用户行为数据
            
            # 模拟获取新数据
            new_data = self._fetch_new_user_data()
            
            if new_data:
                updated_profiles = self.profile_system.batch_update_profiles(new_data)
                logger.info(f"每小时更新完成，共更新 {len(updated_profiles)} 个用户画像")
            
        except Exception as e:
            logger.error(f"每小时更新错误: {e}")
    
    def _daily_deep_update(self):
        """每日深度更新"""
        logger.info("开始每日深度用户画像更新")
        
        try:
            # 深度更新包括：
            # 1. 重新计算所有用户的兴趣标签
            # 2. 更新行为偏好模型
            # 3. 调整活跃度等级阈值
            
            # 获取所有用户数据
            all_user_data = self._fetch_all_user_data()
            
            if all_user_data:
                # 重新计算用户画像
                updated_profiles = self.profile_system.batch_update_profiles(all_user_data)
                
                # 保存画像
                self.profile_system.save_profiles("output/profiles/")
                
                logger.info(f"每日深度更新完成，共更新 {len(updated_profiles)} 个用户画像")
            
        except Exception as e:
            logger.error(f"每日深度更新错误: {e}")
    
    def _weekly_full_update(self):
        """每周全量更新"""
        logger.info("开始每周全量用户画像更新")
        
        try:
            # 全量更新包括：
            # 1. 重新训练特征提取模型
            # 2. 更新特征映射
            # 3. 清理过期数据
            
            # 重新初始化特征映射
            self.profile_system.feature_mappings = self.profile_system._init_feature_mappings()
            
            # 获取全量用户数据
            all_user_data = self._fetch_all_user_data()
            
            if all_user_data:
                # 重新计算所有用户画像
                updated_profiles = self.profile_system.batch_update_profiles(all_user_data)
                
                # 保存画像
                self.profile_system.save_profiles("output/profiles/")
                
                logger.info(f"每周全量更新完成，共更新 {len(updated_profiles)} 个用户画像")
            
        except Exception as e:
            logger.error(f"每周全量更新错误: {e}")
    
    def _health_check(self):
        """健康检查"""
        try:
            # 检查系统状态
            stats = self.profile_system.get_profile_statistics()
            
            # 检查画像质量
            quality_metrics = self.monitor.monitor_profile_quality()
            
            # 记录健康状态
            logger.info(f"系统健康检查 - 用户数: {stats['total_users']}, "
                       f"平均置信度: {quality_metrics['avg_confidence']:.3f}")
            
            # 如果质量指标异常，发送告警
            if quality_metrics['avg_confidence'] < 0.3:
                logger.warning("用户画像质量异常，平均置信度过低")
            
        except Exception as e:
            logger.error(f"健康检查错误: {e}")
    
    def _fetch_new_user_data(self) -> Dict[int, List[Dict[str, Any]]]:
        """获取新的用户数据"""
        # 这里应该实现从实际数据源获取数据的逻辑
        # 例如从Kafka、数据库等获取用户行为数据
        
        # 模拟返回空数据
        return {}
    
    def _fetch_all_user_data(self) -> Dict[int, List[Dict[str, Any]]]:
        """获取所有用户数据"""
        # 这里应该实现从实际数据源获取全量数据的逻辑
        
        # 模拟返回空数据
        return {}
    
    def start(self):
        """启动定时更新器"""
        logger.info("定时用户画像更新器已启动")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次


class ProfileUpdatePipeline:
    """用户画像更新流水线主类"""
    
    def __init__(self, data_path: str, config: Dict[str, Any] = None):
        """
        初始化更新流水线
        
        Args:
            data_path: 数据路径
            config: 配置参数
        """
        self.data_path = data_path
        
        # 初始化用户画像系统
        self.profile_system = UserProfileSystem(data_path, config)
        
        # 初始化实时更新器
        self.real_time_updater = RealTimeProfileUpdater(self.profile_system)
        
        # 初始化定时更新器
        self.scheduled_updater = ScheduledProfileUpdater(self.profile_system)
        
        logger.info("用户画像更新流水线初始化完成")
    
    def start(self):
        """启动更新流水线"""
        # 启动实时更新器
        self.real_time_updater.start()
        
        # 启动定时更新器（在单独线程中）
        update_thread = threading.Thread(
            target=self.scheduled_updater.start, 
            daemon=True
        )
        update_thread.start()
        
        logger.info("用户画像更新流水线已启动")
    
    def stop(self):
        """停止更新流水线"""
        self.real_time_updater.stop()
        logger.info("用户画像更新流水线已停止")
    
    def add_user_event(self, user_id: int, item_id: int, 
                      event_type: str = "click", 
                      additional_data: Dict[str, Any] = None):
        """添加用户事件"""
        event = UpdateEvent(
            user_id=user_id,
            event_type=event_type,
            item_id=item_id,
            timestamp=time.time(),
            additional_data=additional_data or {}
        )
        
        self.real_time_updater.add_event(event)
    
    def get_user_features(self, user_id: int) -> Dict[str, List[int]]:
        """获取用户数组特征"""
        return self.profile_system.get_user_array_features(user_id)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        profile_stats = self.profile_system.get_profile_statistics()
        updater_stats = self.real_time_updater.get_stats()
        
        return {
            "profile_system": profile_stats,
            "real_time_updater": updater_stats
        }
    
    def generate_report(self) -> str:
        """生成系统报告"""
        monitor = UserProfileMonitor(self.profile_system)
        return monitor.generate_report()


# 使用示例
if __name__ == "__main__":
    # 初始化流水线
    pipeline = ProfileUpdatePipeline("data/")
    
    # 启动流水线
    pipeline.start()
    
    try:
        # 模拟用户行为
        for i in range(100):
            user_id = i % 10 + 1
            item_id = 1000 + i
            
            # 添加用户事件
            pipeline.add_user_event(
                user_id=user_id,
                item_id=item_id,
                event_type="click",
                additional_data={"dwell_time": 30 + i % 60}
            )
            
            time.sleep(0.1)
        
        # 等待处理
        time.sleep(10)
        
        # 获取用户特征
        features = pipeline.get_user_features(1)
        print("用户1的数组特征:", features)
        
        # 生成报告
        report = pipeline.generate_report()
        print(report)
        
        # 获取系统统计
        stats = pipeline.get_system_stats()
        print("系统统计:", stats)
        
    finally:
        # 停止流水线
        pipeline.stop()


