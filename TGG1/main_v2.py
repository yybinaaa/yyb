import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import os
from pathlib import Path

# 确保能从您的项目中正确导入
from dataset_v2 import OneRecV2Dataset
from model_v2 import OneRecV2Model, ModelConfig
from utils import save_checkpoint # 导入我们新的保存函数

def get_args():
    parser = argparse.ArgumentParser(description="[Platform Compliant] Training script for OneRec-V2 (SFT)")
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--mm_emb_id', nargs='+', default=[], type=str)
    parser.add_argument('--codebook_size', default=8192, type=int)
    parser.add_argument('--output_model_name', default=None, type=str,
                        help="If specified, save the final model to this file name.")
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device(args.device)
    
    # [平台规范] 初始化 TensorBoard writer
    tf_events_path = os.environ.get('TRAIN_TF_EVENTS_PATH', 'tensorboard/sft')
    writer = SummaryWriter(tf_events_path)
    print(f"TensorBoard logs will be saved to: {tf_events_path}")

    # --- 1. 数据加载 ---
    print("Loading dataset...")
    full_dataset = OneRecV2Dataset(args.data_dir, args, codebook_size=args.codebook_size)
    
    # 划分训练集和验证集
    val_split = 0.1 # 使用10%的数据作为验证集
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples.")

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=full_dataset.collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=full_dataset.collate_fn, pin_memory=True
    )
    
    # --- 2. 模型初始化 ---
    print("Initializing model...")
    config = ModelConfig()
    model = OneRecV2Model(config, full_dataset).to(device)
    print(f"Model on {device}. Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # --- 3. 损失函数和优化器 ---
    criterion = nn.CrossEntropyLoss(ignore_index=full_dataset.PAD_TOKEN_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- 4. 训练与验证循环 ---
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        # --- 训练部分 ---
        model.train()
        progress_bar_train = tqdm(train_loader, desc=f"SFT Epoch {epoch} Training", unit="batch")
        for batch in progress_bar_train:
            if batch is None: continue
            batch = {k: v.to(device) for k, v in batch.items()}
            
            context_features = {
                'user_sparse': batch['user_sparse'],
                'history_item_ids': batch['history_item_ids'],
                'history_item_sparse': batch['history_item_sparse']
            }
            decoder_input_ids = batch['decoder_input_ids']
            decoder_labels = batch['decoder_labels']

            optimizer.zero_grad()
            logits = model(context_features, decoder_input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), decoder_labels.view(-1))
            loss.backward()
            optimizer.step()
            
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            progress_bar_train.set_postfix(loss=f"{loss.item():.4f}")
            global_step += 1

        # --- 验证部分 ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"SFT Epoch {epoch} Validation", unit="batch"):
                if batch is None: continue
                batch = {k: v.to(device) for k, v in batch.items()}
                
                context_features = {
                    'user_sparse': batch['user_sparse'],
                    'history_item_ids': batch['history_item_ids'],
                    'history_item_sparse': batch['history_item_sparse']
                }
                decoder_input_ids = batch['decoder_input_ids']
                decoder_labels = batch['decoder_labels']

                logits = model(context_features, decoder_input_ids)
                val_loss = criterion(logits.view(-1, logits.size(-1)), decoder_labels.view(-1))
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        writer.add_scalar('Loss/validation_epoch', avg_val_loss, epoch)
        print(f"\nEpoch {epoch} - Avg Validation Loss: {avg_val_loss:.4f}")

        # --- 保存最佳模型 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"🎉 New best model found! Saving checkpoint...")
            metrics = {"epoch": epoch, "global_step": global_step, "validation_loss": avg_val_loss}
            save_checkpoint(epoch, global_step, model, optimizer, metrics, 'TRAIN_CKPT_PATH')

    writer.close()
    if args.output_model_name:
        print(f"Saving final model to: {args.output_model_name}")
        torch.save(model.state_dict(), args.output_model_name)

    writer.close()
    print("Supervised Fine-Tuning complete!")


if __name__ == '__main__':
    main()