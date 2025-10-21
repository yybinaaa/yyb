import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRewardCalculator:
    """
    实现了我们讨论的【方案一：简单层级奖励】。
    根据 action_type 计算奖励分数。
    """
    def __init__(self, device):
        # 定义奖励映射
        # key: action_type from dataset
        # value: reward score
        self.reward_map = {
            4: 1.0,   # purchase (购买)
            3: 0.5,   # add_to_cart (加购)
            2: 0.2,   # click (点击)
            -1: -1.0, # negative_feedback (负反馈)
        }
        self.default_reward = 0.0 # impression (曝光)
        self.device = device

    @torch.no_grad()
    def compute_rewards(self, action_types: torch.Tensor) -> torch.Tensor:
        """
        将一个批次的 action_type 张量转换为奖励分数张量。
        
        Args:
            action_types (torch.Tensor): a batch of action types, shape: [batch_size].
            
        Returns:
            torch.Tensor: a batch of reward scores, shape: [batch_size].
        """
        # 使用 PyTorch 的 apply_ 方法高效地进行映射
        rewards = action_types.clone().float()
        for action_type, reward_val in self.reward_map.items():
            rewards[action_types == action_type] = reward_val
        
        # 将不在map中的action_type设置为默认奖励
        # 创建一个mask来识别哪些action_type在map的key中
        known_actions_mask = torch.zeros_like(action_types, dtype=torch.bool)
        for action_type in self.reward_map.keys():
            known_actions_mask |= (action_types == action_type)
        print(f"\n[DEBUG] Received action_types: {torch.unique(action_types)}")
        
        rewards = action_types.clone().float()
        
        rewards[~known_actions_mask] = self.default_reward
        
        return rewards.to(self.device)


class GBPOLoss(nn.Module):
    """
    实现论文中的 GBPO (Gradient-Bounded Policy Optimization) 损失函数。
    (见论文公式 11, 12)
    """
    def __init__(self, epsilon=0.2): # epsilon 是一个通用超参数，这里暂时保留
        super().__init__()
        self.epsilon = epsilon

    def forward(self, 
                log_probs_new: torch.Tensor, 
                log_probs_old: torch.Tensor, 
                rewards: torch.Tensor):
        """
        计算 GBPO 损失。
        
        Args:
            log_probs_new (torch.Tensor): 当前策略 π_θ 对动作的对数概率。
            log_probs_old (torch.Tensor): 旧策略 π_θ_old 对动作的对数概率。
            rewards (torch.Tensor): 计算出的奖励 (Advantage)，在这里就是 +1, -1, 0 等。
        """
        # 1. 计算策略比例 (Policy Ratio) r = π_θ / π_θ_old
        # 在对数空间中，log(r) = log(π_θ) - log(π_θ_old)
        # r = exp(log(π_θ) - log(π_θ_old))
        ratio = torch.exp(log_probs_new - log_probs_old)

        # 2. 实现论文公式 (12): 计算 π'_θ_old (动态边界)
        # sg 是 stop_gradient 的缩写
        probs_new_sg = torch.exp(log_probs_new).detach() # sg(π_θ)
        
        # 创建正奖励和负奖励的掩码
        positive_mask = (rewards > 0)
        negative_mask = (rewards < 0)

        # 初始化动态边界 π'_θ_old
        pi_prime_old = torch.exp(log_probs_old).detach() # π_θ_old

        # 对正奖励样本: π'_θ_old = max(π_θ_old, sg(π_θ))
        pi_prime_old[positive_mask] = torch.max(
            pi_prime_old[positive_mask], 
            probs_new_sg[positive_mask]
        )
        
        # 对负奖励样本: π'_θ_old = max(π_θ_old, 1 - sg(π_θ))
        pi_prime_old[negative_mask] = torch.max(
            pi_prime_old[negative_mask], 
            1 - probs_new_sg[negative_mask]
        )
        
        # 3. 计算修正后的策略比例 r' = π_θ / π'_θ_old
        ratio_prime = torch.exp(log_probs_new) / (pi_prime_old + 1e-8) # 加上eps防止除以0

        # 4. 计算最终损失 (论文公式 11)
        # L = - r' * Advantage
        # 我们需要最大化这个目标，所以取负号
        loss = - (ratio_prime * rewards)
        
        # 返回批次的平均损失
        return loss.mean()