import torch
import torch.nn as nn

class PreferenceLoss(nn.Module):
    """
    实现一个简单的偏好损失 (Hinge Loss)。
    目标: score(positive) > score(negative) + margin
    """
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, 
                positive_scores: torch.Tensor, 
                negative_scores: torch.Tensor):
        """
        Args:
            positive_scores (torch.Tensor): 模型对正样本的打分 (对数概率)。
            negative_scores (torch.Tensor): 模型对负样本的打分 (对数概率)。
        """
        # Loss = max(0, margin - (score_pos - score_neg))
        loss = torch.clamp(self.margin - (positive_scores - negative_scores), min=0)
        return loss.mean()