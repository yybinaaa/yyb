import argparse
import json
import os
import time
import math
from pathlib import Path
import pickle

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset08281 import MyDataset
from model08281 import ThreeTowerModel

# --- 核心修改 1: 一个新的、自动确定 Action Type 词典大小的函数 ---
def get_action_vocab_size_from_seq(data_dir: Path, log_dir: Path) -> int:
    """
    通过扫描 seq.jsonl 来自动确定 action_type 的词典大小，并将结果缓存。
    """
    cache_path = log_dir / 'action_type_info.json'
    seq_file = data_dir / 'seq.jsonl'

    # 如果有缓存，直接读取
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                info = json.load(f)
            print(f"✅ 从缓存 {cache_path} 加载 Action Type 信息。词典大小: {info['vocab_size']}")
            return info['vocab_size']
        except Exception as e:
            print(f"⚠️ 读取缓存失败: {e}。将重新扫描。")

    if not seq_file.exists():
        raise FileNotFoundError(f"未找到 seq.jsonl at {seq_file}")

    print("--- 正在扫描 seq.jsonl 以确定 Action Type 词典大小 ---")
    action_values = set()
    with open(seq_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="扫描 Action Types"):
            try:
                sequence = json.loads(line)
                for record in sequence:
                    if record[4] is not None:
                        action_values.add(record[4])
            except (json.JSONDecodeError, IndexError):
                continue

    if not action_values:
        raise ValueError("在 seq.jsonl 中未找到任何有效的 action_type 值。")

    max_action_id = max(action_values)
    # 词典大小应该是最大ID + 1
    vocab_size = max_action_id + 1

    print(f"✅ 扫描完成。发现的 Action Type 值: {sorted(list(action_values))}")
    print(f"   - 最大 Action ID: {max_action_id}")
    print(f"   - 推断的词典大小 (vocab_size): {vocab_size}")
    
    # 写入缓存
    info = {'max_id': max_action_id, 'vocab_size': vocab_size}
    try:
        with open(cache_path, 'w') as f:
            json.dump(info, f, indent=4)
        print(f"   - 扫描结果已缓存至: {cache_path}")
    except OSError as e:
        print(f"⚠️ 无法写入缓存: {e}")

    return vocab_size

def get_args():
    parser = argparse.ArgumentParser()
    # 训练超参数
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--warmup_steps', default=1000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    
    # 模型结构参数
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--num_blocks', default=6, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--dropout_rate', default=0.07, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    
    # 运行环境参数
    parser.add_argument('--device', default='cuda', type=str)
    
    # 特征相关参数
    parser.add_argument('--mm_emb_id', nargs='+', default=[], type=str)
    
    # --- 核心修改 2: 移除了 --action_feature_id 参数 ---
    return parser.parse_args()

# run_validation 和 get_lr 函数保持不变
def run_validation(model, data_loader, device, desc=""):
    model.eval()
    loss_sum, acc10_sum, steps = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc):
            (seq_ids, pos_ids, action_seqs, 
             user_sparse, user_arrays,
             seq_sparse, pos_sparse, 
             seq_mm, pos_mm, full_histories) = batch
            seq_ids, pos_ids, action_seqs, user_sparse = seq_ids.to(device), pos_ids.to(device), action_seqs.to(device), user_sparse.to(device)
            user_arrays = {k: v.to(device) for k, v in user_arrays.items()}
            seq_sparse, pos_sparse = seq_sparse.to(device), pos_sparse.to(device)
            seq_mm = {k: v.to(device) for k, v in seq_mm.items()}
            pos_mm = {k: v.to(device) for k, v in pos_mm.items()}
            full_histories = [h.to(device) for h in full_histories]

            loss = model(seq_ids, pos_ids, action_seqs, user_sparse, user_arrays,
                         seq_sparse, pos_sparse, seq_mm, pos_mm, full_histories)
            
            loss_sum += loss.item()
            acc10_sum += getattr(model, 'last_acc10', 0.0)
            steps += 1
    return (loss_sum / steps, acc10_sum / steps) if steps > 0 else (float('inf'), 0.0)

def get_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps: return base_lr * (step + 1) / warmup_steps
    if total_steps <= warmup_steps: return base_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

if __name__ == '__main__':
    # --- 1. 初始化设置 ---
    data_path = Path(os.environ.get('TRAIN_DATA_PATH', 'data'))
    log_path = Path(os.environ.get('TRAIN_LOG_PATH', 'logs'))
    ckpt_path = Path(os.environ.get('TRAIN_CKPT_PATH', 'checkpoints'))
    tf_events_path = Path(os.environ.get('TRAIN_TF_EVENTS_PATH', 'tensorboard'))
    
    args = get_args()
    
    log_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    tf_events_path.mkdir(parents=True, exist_ok=True)
    
    log_file = open(log_path / f'train_log_{time.strftime("%Y%m%d-%H%M%S")}.txt', 'w')
    writer = SummaryWriter(str(tf_events_path))

    print(f"数据路径: {data_path}")
    print(f"日志路径: {log_path}")

    # --- 核心修改 3: 使用新的自动扫描函数 ---
    try:
        args.action_type_vocab_size = get_action_vocab_size_from_seq(data_path, log_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ 初始化失败: {e}")
        exit()

    # --- 后续所有代码 (数据加载, 模型初始化, 训练循环) 都保持不变 ---
    dataset = MyDataset(data_path, args)
    
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.9 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=MyDataset.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=MyDataset.collate_fn)
    
    model = ThreeTowerModel(
        user_num=dataset.usernum,
        item_num=dataset.itemnum,
        feat_statistics=dataset.indexer['f'],
        feat_types=dataset.feature_types,
        args=args
    )
    model.to(args.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    global_step = 0
    total_steps = len(train_loader) * args.num_epochs
    print(f"🚀 开始训练... 总步数: {total_steps}")
    
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_train_loss, epoch_train_acc10, num_train_steps = 0.0, 0.0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs} [Training]")
        for step, batch in enumerate(pbar):
            lr = get_lr(global_step, total_steps, args.warmup_steps, args.lr)
            for param_group in optimizer.param_groups: param_group['lr'] = lr
            
            (seq_ids, pos_ids, action_seqs, 
             user_sparse, user_arrays,
             seq_sparse, pos_sparse, 
             seq_mm, pos_mm, full_histories) = batch
            seq_ids, pos_ids, action_seqs, user_sparse = seq_ids.to(args.device), pos_ids.to(args.device), action_seqs.to(args.device), user_sparse.to(args.device)
            user_arrays = {k: v.to(args.device) for k, v in user_arrays.items()}
            seq_sparse, pos_sparse = seq_sparse.to(args.device), pos_sparse.to(args.device)
            seq_mm = {k: v.to(args.device) for k, v in seq_mm.items()}
            pos_mm = {k: v.to(device) for k, v in pos_mm.items()}
            full_histories = [h.to(args.device) for h in full_histories]
            
            train_loss = model(seq_ids, pos_ids, action_seqs, user_sparse, user_arrays,
                               seq_sparse, pos_sparse, seq_mm, pos_mm, full_histories, writer)
            
            optimizer.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            pbar.set_postfix(loss=f"{train_loss.item():.4f}", acc10=f"{model.last_acc10:.4f}", lr=f"{lr:.6f}")
            writer.add_scalar('Loss/train_step', train_loss.item(), global_step)
            writer.add_scalar('Acc/train_step_acc10', model.last_acc10, global_step)

            epoch_train_loss += train_loss.item()
            epoch_train_acc10 += model.last_acc10
            num_train_steps += 1
            global_step += 1
        
        avg_train_loss = epoch_train_loss / num_train_steps
        avg_train_acc10 = epoch_train_acc10 / num_train_steps

        valid_loss_avg, valid_acc10_avg = run_validation(
            model, valid_loader, args.device, desc=f"Epoch {epoch}/{args.num_epochs} [Validation]"
        )
        
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('Loss/valid_epoch', valid_loss_avg, epoch)
        writer.add_scalar('Acc/train_epoch_acc10', avg_train_acc10, epoch)
        writer.add_scalar('Acc/valid_epoch_acc10', valid_acc10_avg, epoch)
        writer.add_scalar('Hyperparameters/learning_rate', lr, epoch)
        
        log_str = (f"\n📊 Epoch {epoch} Summary: \n"
                   f"  [Train] Avg Loss={avg_train_loss:.4f}, Avg Acc@10={avg_train_acc10:.4f}\n"
                   f"  [Valid] Avg Loss={valid_loss_avg:.4f}, Avg Acc@10={valid_acc10_avg:.4f}\n"
                   f"  [Monitor] PosSim={model.last_pos_sim:.4f}, NegSim={model.last_neg_sim:.4f}")
        print(log_str)
        log_file.write(log_str + '\n'); log_file.flush()

        save_dir = ckpt_path / f"epoch_{epoch}_step_{global_step}"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "model.pt"
        try:
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model saved to: {save_path}")
        except Exception as e:
            print(f"❌ Failed to save model to {save_path}. Error: {e}")
        
    print("🎉 Training finished.")
    writer.close()
    log_file.close()