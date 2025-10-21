import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import os
from pathlib import Path
import copy

# 确保能从您的项目中正确导入
from rl_dataset_v2 import PreferenceDataset
from model_v2 import OneRecV2Model, ModelConfig
from rl_utils_v2 import PreferenceLoss
from utils import save_checkpoint

def get_args():
    parser = argparse.ArgumentParser(description="[Platform Compliant] Preference-based RL Fine-tuning for OneRec-V2")
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--base_model_path', required=True, type=str,
                        help="Path to the SFT pretrained model checkpoint (.pth).")
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=1e-6, type=float, help="Learning rate for preference fine-tuning.")
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--mm_emb_id', nargs='+', default=[], type=str)
    parser.add_argument('--codebook_size', default=8192, type=int)
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device(args.device)
    
    # [平台规范] 初始化 TensorBoard writer
    tf_events_path = os.environ.get('TRAIN_TF_EVENTS_PATH', 'tensorboard/rl')
    writer = SummaryWriter(tf_events_path)
    print(f"TensorBoard logs will be saved to: {tf_events_path}")

    # --- 1. 数据加载 ---
    print("Loading Preference dataset...")
    pref_dataset = PreferenceDataset(args.data_dir, args, codebook_size=args.codebook_size)
    pref_loader = DataLoader(
        pref_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=PreferenceDataset.collate_fn, pin_memory=True
    )
    
    # --- 2. 模型初始化 ---
    print("Initializing model...")
    config = ModelConfig()
    policy_model = OneRecV2Model(config, pref_dataset).to(device)
    
    print(f"Loading base model weights from: {args.base_model_path}")
    pretrained_dict = torch.load(args.base_model_path, map_location=device)
    model_dict = policy_model.state_dict()
    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    policy_model.load_state_dict(model_dict, strict=False)
    print("Model initialized and weights transferred.")

    # --- 3. 损失函数和优化器 ---
    preference_loss_fn = PreferenceLoss(margin=0.5)
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=args.lr)

    # --- 4. 偏好学习微调循环 ---
    global_step = 0
    for epoch in range(1, args.num_epochs + 1):
        policy_model.train()
        total_loss = 0
        
        progress_bar = tqdm(pref_loader, desc=f"Preference Epoch {epoch}/{args.num_epochs}", unit="batch")
        
        for batch in progress_bar:
            if batch is None: continue
            
            batch = {k: v.to(device) for k, v in batch.items()}
            
            context_features = {
                'user_sparse': batch['user_sparse'],
                'history_item_ids': batch['history_item_ids'],
            }
            positive_action_ids = batch['positive_action_ids']
            negative_action_ids = batch['negative_action_ids']

            optimizer.zero_grad()

            positive_scores = policy_model.get_action_prob(context_features, positive_action_ids)
            negative_scores = policy_model.get_action_prob(context_features, negative_action_ids)
            
            loss = preference_loss_fn(positive_scores, negative_scores)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            writer.add_scalar('Loss/preference_step', loss.item(), global_step)
            progress_bar.set_postfix(pref_loss=f"{loss.item():.4f}")
            global_step += 1

        avg_loss = total_loss / len(pref_loader) if len(pref_loader) > 0 else 0
        writer.add_scalar('Loss/preference_epoch', avg_loss, epoch)
        print(f"\nPreference Epoch {epoch} finished. Average Loss: {avg_loss:.4f}\n")
        
        # [平台规范] 每个 epoch 结束后都保存检查点
        metrics = {"epoch": epoch, "global_step": global_step, "preference_loss": avg_loss}
        save_checkpoint(epoch, global_step, policy_model, optimizer, metrics, 'TRAIN_CKPT_PATH')

    writer.close()
    print("Preference-based Fine-tuning complete!")

if __name__ == '__main__':
    main()