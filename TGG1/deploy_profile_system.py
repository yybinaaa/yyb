"""
用户画像系统部署和运维脚本
用于部署、监控和管理用户画像系统
"""

import os
import sys
import json
import time
import signal
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import threading
from datetime import datetime

from user_profile_system import UserProfileSystem, UserProfileMonitor
from profile_update_pipeline import ProfileUpdatePipeline

class ProfileSystemDeployer:
    """用户画像系统部署器"""
    
    def __init__(self, config_path: str = "profile_config.yaml"):
        """
        初始化部署器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.pipeline = None
        self.is_running = False
        
        # 设置日志
        self._setup_logging()
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            import yaml
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except ImportError:
            self.logger.warning("PyYAML未安装，使用默认配置")
            return self._default_config()
        except FileNotFoundError:
            self.logger.warning(f"配置文件 {self.config_path} 不存在，使用默认配置")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            "system": {
                "data_path": "data/",
                "output_path": "output/profiles/",
                "log_level": "INFO"
            },
            "real_time_update": {
                "update_interval": 300,
                "batch_size": 1000
            }
        }
    
    def _setup_logging(self):
        """设置日志"""
        log_level = self.config.get("system", {}).get("log_level", "INFO")
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('profile_system.log')
            ]
        )
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        self.logger.info(f"接收到信号 {signum}，开始优雅关闭...")
        self.stop()
        sys.exit(0)
    
    def deploy(self):
        """部署系统"""
        self.logger.info("开始部署用户画像系统...")
        
        try:
            # 创建必要的目录
            self._create_directories()
            
            # 初始化系统
            self._initialize_system()
            
            # 启动系统
            self._start_system()
            
            self.logger.info("用户画像系统部署完成")
            
        except Exception as e:
            self.logger.error(f"部署失败: {e}")
            raise
    
    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            self.config["system"]["output_path"],
            "output/reports/",
            "logs/"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"创建目录: {directory}")
    
    def _initialize_system(self):
        """初始化系统"""
        data_path = self.config["system"]["data_path"]
        
        # 初始化流水线
        self.pipeline = ProfileUpdatePipeline(data_path, self.config)
        
        self.logger.info("系统初始化完成")
    
    def _start_system(self):
        """启动系统"""
        if self.pipeline is None:
            raise RuntimeError("系统未初始化")
        
        # 启动流水线
        self.pipeline.start()
        self.is_running = True
        
        self.logger.info("系统已启动")
    
    def stop(self):
        """停止系统"""
        if self.pipeline and self.is_running:
            self.pipeline.stop()
            self.is_running = False
            self.logger.info("系统已停止")
    
    def status(self) -> Dict[str, Any]:
        """获取系统状态"""
        if not self.pipeline:
            return {"status": "not_initialized"}
        
        try:
            stats = self.pipeline.get_system_stats()
            return {
                "status": "running" if self.is_running else "stopped",
                "stats": stats,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        if not self.pipeline:
            return {"healthy": False, "reason": "system_not_initialized"}
        
        try:
            # 检查系统状态
            stats = self.pipeline.get_system_stats()
            
            # 检查关键指标
            profile_stats = stats.get("profile_system", {})
            updater_stats = stats.get("real_time_updater", {})
            
            # 健康检查条件
            is_healthy = (
                self.is_running and
                updater_stats.get("is_running", False) and
                updater_stats.get("errors", 0) < 100 and
                profile_stats.get("total_users", 0) > 0
            )
            
            return {
                "healthy": is_healthy,
                "details": {
                    "system_running": self.is_running,
                    "updater_running": updater_stats.get("is_running", False),
                    "error_count": updater_stats.get("errors", 0),
                    "user_count": profile_stats.get("total_users", 0),
                    "queue_size": updater_stats.get("queue_size", 0)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "reason": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def restart(self):
        """重启系统"""
        self.logger.info("重启用户画像系统...")
        self.stop()
        time.sleep(2)
        self._start_system()
        self.logger.info("系统重启完成")


class ProfileSystemMonitor:
    """用户画像系统监控器"""
    
    def __init__(self, deployer: ProfileSystemDeployer):
        self.deployer = deployer
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval: int = 60):
        """开始监控"""
        if self.monitoring:
            self.logger.warning("监控已在运行")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info(f"开始监控，间隔 {interval} 秒")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("监控已停止")
    
    def _monitor_loop(self, interval: int):
        """监控循环"""
        while self.monitoring:
            try:
                # 健康检查
                health = self.deployer.health_check()
                
                if not health["healthy"]:
                    self.logger.warning(f"系统健康检查失败: {health}")
                    
                    # 尝试自动恢复
                    self._attempt_recovery()
                
                # 记录状态
                status = self.deployer.status()
                self._log_status(status)
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
                time.sleep(interval)
    
    def _attempt_recovery(self):
        """尝试自动恢复"""
        self.logger.info("尝试自动恢复系统...")
        
        try:
            # 检查系统状态
            status = self.deployer.status()
            
            if status["status"] == "error":
                # 重启系统
                self.deployer.restart()
                self.logger.info("系统已自动重启")
            
        except Exception as e:
            self.logger.error(f"自动恢复失败: {e}")
    
    def _log_status(self, status: Dict[str, Any]):
        """记录状态"""
        # 这里可以添加状态记录逻辑
        # 例如写入数据库、发送告警等
        pass


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="用户画像系统部署和运维工具")
    parser.add_argument("--config", default="profile_config.yaml", help="配置文件路径")
    parser.add_argument("--action", choices=["deploy", "start", "stop", "restart", "status", "health"], 
                       default="deploy", help="操作类型")
    parser.add_argument("--monitor", action="store_true", help="启动监控")
    parser.add_argument("--monitor-interval", type=int, default=60, help="监控间隔(秒)")
    
    args = parser.parse_args()
    
    # 创建部署器
    deployer = ProfileSystemDeployer(args.config)
    
    try:
        if args.action == "deploy":
            deployer.deploy()
            
            if args.monitor:
                monitor = ProfileSystemMonitor(deployer)
                monitor.start_monitoring(args.monitor_interval)
                
                # 保持运行
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    monitor.stop_monitoring()
                    deployer.stop()
        
        elif args.action == "start":
            deployer._initialize_system()
            deployer._start_system()
        
        elif args.action == "stop":
            deployer.stop()
        
        elif args.action == "restart":
            deployer.restart()
        
        elif args.action == "status":
            status = deployer.status()
            print(json.dumps(status, indent=2, ensure_ascii=False))
        
        elif args.action == "health":
            health = deployer.health_check()
            print(json.dumps(health, indent=2, ensure_ascii=False))
    
    except Exception as e:
        logging.error(f"操作失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


