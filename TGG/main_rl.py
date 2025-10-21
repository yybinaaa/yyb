import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import copy

# 从我们创建的文件中导入所有必要的模块
from rl_dataset import RLDataset
from model_v2 import OneRecV2Model, ModelConfig
from rl_utils import SimpleRewardCalculator, GBPOLoss

def get_args():
    parser = argparse.ArgumentParser(description="RL Fine-tuning script for OneRec-V2")
    # --- 路径参数 ---
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--base_model_path', required=True, type=str,
                        help="Path to the pretrained base model checkpoint (.pth).")
    
    # --- 训练参数 ---
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=1e-5, type=float, 
                        help="Learning rate for RL fine-tuning (usually smaller).")
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--mm_emb_id', nargs='+', default=[], type=str)
    parser.add_argument('--codebook_size', default=8192, type=int)
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device(args.device)
    
    # --- 1. 数据加载 ---
    print("Loading RL dataset...")
    rl_dataset = RLDataset(args.data_dir, args, codebook_size=args.codebook_size)
    # 在RL中，通常不随机 shuffle，而是顺序处理经验
    rl_loader = DataLoader(
        rl_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=RLDataset.collate_fn, pin_memory=True
    )
    
    # --- 2. 模型初始化 ---
    print("Initializing models...")
    config = ModelConfig()

    # (A) 当前策略模型 (π_θ)，这是我们需要训练的模型
    policy_model = OneRecV2Model(config, rl_dataset).to(device)

    # (B) 旧策略模型 (π_θ_old)
    old_policy_model = copy.deepcopy(policy_model).to(device)

    # --- [核心修正] 智能加载预训练权重 ---
    print(f"Loading base model weights from: {args.base_model_path}")
    pretrained_state_dict = torch.load(args.base_model_path, map_location=device)
    current_model_dict = policy_model.state_dict()

    # 1. 创建一个新的 state_dict，只包含那些形状匹配的权重
    new_state_dict = {}
    for k, v in pretrained_state_dict.items():
        if k in current_model_dict and v.shape == current_model_dict[k].shape:
            new_state_dict[k] = v
        else:
            print(f"  - Skipping layer '{k}' due to shape mismatch.")

    # 2. 用新的 state_dict 更新当前模型的权重
    current_model_dict.update(new_state_dict)

    # 3. 加载更新后的权重
    # strict=False 允许我们只加载 state_dict 中的一部分键
    policy_model.load_state_dict(current_model_dict, strict=False)
    old_policy_model.load_state_dict(current_model_dict, strict=False)

    # 将旧模型设置为评估模式，不计算梯度
    old_policy_model.eval()

    print("\nModels initialized. Weights successfully transferred from base model.")
    # --- 修正结束 ---

    # --- 3. 初始化奖励计算器和损失函数 ---
    reward_calculator = SimpleRewardCalculator(device=device)
    gbpo_loss_fn = GBPOLoss()
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=args.lr)

    # --- 4. 强化学习微调循环 ---
    for epoch in range(1, args.num_epochs + 1):
        policy_model.train()
        total_loss = 0
        
        progress_bar = tqdm(rl_loader, desc=f"RL Epoch {epoch}/{args.num_epochs}", unit="batch")
        
        for batch in progress_bar:
            if batch is None: continue
            
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 从批次中解包 State, Action, Reward Info
            context_features = {
                'user_sparse': batch['user_sparse'],
                'history_item_ids': batch['history_item_ids'],
                'history_item_sparse': batch['history_item_sparse']
            }
            action_ids = batch['action_ids']
            action_types = batch['action_type']

            # --- RL 核心步骤 ---
            # a. 使用旧策略计算旧的对数概率 log_probs_old
            # detach() 确保我们不会在这个步骤中计算梯度
            with torch.no_grad():
                log_probs_old = old_policy_model.get_action_prob(context_features, action_ids)

            # b. 使用当前策略计算新的对数概率 log_probs_new
            log_probs_new = policy_model.get_action_prob(context_features, action_ids)
            
            # c. 使用奖励计算器计算奖励
            rewards = reward_calculator.compute_rewards(action_types)
            
            # d. 计算 GBPO 损失
            optimizer.zero_grad()
            loss = gbpo_loss_fn(log_probs_new, log_probs_old.detach(), rewards)
            
            # e. 反向传播和优化
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(gbpo_loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(rl_loader)
        print(f"\nRL Epoch {epoch} finished. Average GBPO Loss: {avg_loss:.4f}\n")
        
        # [可选] 在每个 epoch 结束后，可以更新 old_policy_model
        # old_policy_model.load_state_dict(policy_model.state_dict())
        
    print("RL Fine-tuning complete!")
    
    # 保存微调后的模型
    final_model_path = "onerec_v2_rl_tuned.pth"
    torch.save(policy_model.state_dict(), final_model_path)
    print(f"Fine-tuned model saved to {final_model_path}")


if __name__ == '__main__':
    main()