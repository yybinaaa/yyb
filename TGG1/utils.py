# in utils.py
import torch
import json
import os
# [核心修正] 从 pathlib 库中导入 Path 对象
from pathlib import Path

def save_checkpoint(epoch, global_step, model, optimizer, metrics, checkpoint_dir_base):
    """
    [平台规范版] 结构化模型检查点保存函数。
    """
    # 1. 获取平台指定的根目录
    # 如果环境变量不存在，则使用本地的 'checkpoints' 目录
    base_path = Path(os.environ.get(checkpoint_dir_base, 'checkpoints'))
    
    # 2. 构造符合规范的目录名: global_step_{...}
    # 为了让文件名更具可读性，我们添加其他信息作为后缀
    val_loss = metrics.get('validation_loss', metrics.get('preference_loss', 0.0))
    loss_str = f"{val_loss:.4f}".replace('.', '_') # e.g., 0.8279 -> 0_8279
    
    dir_name = f"global_step_{global_step}_epoch_{epoch}_loss_{loss_str}"
    checkpoint_path = base_path / dir_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # 3. 在该目录下保存所有文件
    torch.save(model.state_dict(), checkpoint_path / "model.pth")
    torch.save(optimizer.state_dict(), checkpoint_path / "optimizer.pth")
    
    with open(checkpoint_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\n✅ Checkpoint saved successfully to: {checkpoint_path}")