import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class VisualAlignmentLoss(nn.Module):
    """
    计算预测特征和目标特征之间的视觉对齐损失 (VF Loss)。
    VF Loss 由两个部分组成：
    1. Marginal Cosine Similarity Loss: 鼓励预测特征和目标特征在方向上保持一致。
    2. Marginal Distance Similarity Loss: 鼓励预测特征和目标特征在L2距离上接近。
    """
    
    def __init__(self, 
                 cosine_weight: float = 0.5, 
                 distance_weight: float = 0.5):
        """
        初始化视觉对齐损失模块。

        Args:
            cosine_weight (float): 余弦相似度损失的权重。
            distance_weight (float): L2距离损失的权重。
        """
        super().__init__()
        
        if cosine_weight + distance_weight != 1.0:
            print(f"Warning: Loss weights do not sum to 1.0 (cosine: {cosine_weight}, distance: {distance_weight})")
            
        self.cosine_weight = cosine_weight
        self.distance_weight = distance_weight
        
        # 定义期望处理的视角
        self.views = ['front', 'side', 'top']
        
    def forward(self, 
                predicted_features: Dict[str, torch.Tensor], 
                target_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算并返回包含各项损失的字典。

        Args:
            predicted_features (Dict[str, torch.Tensor]): 
                由 g2f 模块预测的特征向量字典。
                例如: {'front': tensor, 'side': tensor, 'top': tensor}
            
            target_features (Dict[str, torch.Tensor]): 
                由 2D VFM 提取的目标（真实）特征向量字典，作为监督信号。
                例如: {'front': tensor, 'side': tensor, 'top': tensor}

        Returns:
            Dict[str, torch.Tensor]: 包含总损失和各项子损失的字典。
        """
        
        total_cosine_loss = 0.0
        total_distance_loss = 0.0
        num_valid_views = 0
        
        per_view_losses = {}

        for view in self.views:
            # 确保预测和目标都包含当前视角
            if view not in predicted_features or view not in target_features:
                continue
            
            pred_feat = predicted_features[view]
            target_feat = target_features[view]
            
            # 确保特征向量已经L2归一化，这对于稳定的余弦相似度计算很重要
            pred_feat = F.normalize(pred_feat, p=2, dim=-1)
            target_feat = F.normalize(target_feat, p=2, dim=-1)

            # 1. 计算 Marginal Cosine Similarity Loss
            # 我们希望余弦相似度接近1，所以损失是 1 - similarity
            cosine_loss_view = (1 - F.cosine_similarity(pred_feat, target_feat, dim=-1)).mean()
            
            # 2. 计算 Marginal Distance Similarity Loss (使用 L2 距离)
            distance_loss_view = F.mse_loss(pred_feat, target_feat)
            
            # 累加损失
            total_cosine_loss += cosine_loss_view
            total_distance_loss += distance_loss_view
            num_valid_views += 1
            
            # 记录每个视角的单独损失（用于详细日志）
            per_view_losses[f'cosine_{view}'] = cosine_loss_view.detach()
            per_view_losses[f'distance_{view}'] = distance_loss_view.detach()

        if num_valid_views == 0:
            # 如果没有任何有效的视角对，返回零损失
            return {
                'vf_loss': torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=True),
                'avg_cosine_loss': torch.tensor(0.0),
                'avg_distance_loss': torch.tensor(0.0)
            }

        # 计算平均损失
        avg_cosine_loss = total_cosine_loss / num_valid_views
        avg_distance_loss = total_distance_loss / num_valid_views
        
        # 计算加权的最终 VF Loss
        vf_loss = (self.cosine_weight * avg_cosine_loss) + \
                  (self.distance_weight * avg_distance_loss)
                  
        # 准备返回的损失字典
        loss_dict = {
            'vf_loss': vf_loss,
            'avg_cosine_loss': avg_cosine_loss.detach(),
            'avg_distance_loss': avg_distance_loss.detach()
        }
        
        # 将每个视角的损失也加入字典
        loss_dict.update(per_view_losses)
        
        return loss_dict

# --- 使用示例 (用于测试和演示) ---
def test_visual_alignment_loss():
    print("🧪 测试 VisualAlignmentLoss 模块...")
    
    # 初始化损失函数
    loss_fn = VisualAlignmentLoss(cosine_weight=0.6, distance_weight=0.4)
    
    # 模拟输入数据
    batch_size = 8
    feature_dim = 384 # DINOv2-small 的特征维度
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 模拟 2D VFM 输出的真实特征
    mock_target_features = {
        'front': torch.randn(batch_size, feature_dim, device=device),
        'side': torch.randn(batch_size, feature_dim, device=device),
        'top': torch.randn(batch_size, feature_dim, device=device)
    }
    
    # 模拟 g2f 输出的预测特征 (与真实特征相似但有噪声)
    mock_predicted_features = {
        view: feat + torch.randn_like(feat) * 0.1 
        for view, feat in mock_target_features.items()
    }
    
    print("\n模拟输入:")
    print(f"  Batch size: {batch_size}, Feature dim: {feature_dim}")
    print(f"  预测特征键: {list(mock_predicted_features.keys())}")
    print(f"  目标特征键: {list(mock_target_features.keys())}")
    
    # 计算损失
    loss_dict = loss_fn(mock_predicted_features, mock_target_features)
    
    print("\n✅ 损失计算成功！")
    print("输出的损失字典:")
    for key, value in loss_dict.items():
        print(f"  - {key}: {value.item():.6f}")
        
    # 测试只提供部分视角的情况
    print("\n测试部分视角输入:")
    mock_predicted_features_partial = {'front': mock_predicted_features['front']}
    mock_target_features_partial = {'front': mock_target_features['front']}
    loss_dict_partial = loss_fn(mock_predicted_features_partial, mock_target_features_partial)
    print("部分视角损失字典:")
    for key, value in loss_dict_partial.items():
        print(f"  - {key}: {value.item():.6f}")


if __name__ == '__main__':
    test_visual_alignment_loss()
