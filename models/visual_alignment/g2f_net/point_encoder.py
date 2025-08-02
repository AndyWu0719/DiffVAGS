import torch
import torch.nn as nn
import torch.nn.functional as F

class LightPointEncoder(nn.Module):
    """轻量级点云特征编码器"""
    
    def __init__(self, 
                 point_dim: int = 3, 
                 attr_dim: int = 56,
                 hidden_dim: int = 256,
                 output_dim: int = 768):
        super().__init__()
        
        # XYZ坐标编码器
        self.coord_encoder = nn.Sequential(
            nn.Linear(point_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 属性编码器
        self.attr_encoder = nn.Sequential(
            nn.Linear(attr_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 特征融合器
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # 注意力池化
        self.attention_pool = nn.Linear(output_dim, 1)
    
    def forward(self, points: torch.Tensor, attributes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: [B, N, 3] 点云坐标
            attributes: [B, N, D] 点云属性
        Returns:
            [B, output_dim] 全局特征
        """
        # 编码坐标和属性
        coord_feat = self.coord_encoder(points)  # [B, N, hidden_dim]
        attr_feat = self.attr_encoder(attributes)  # [B, N, hidden_dim]
        
        # 融合特征
        fused = torch.cat([coord_feat, attr_feat], dim=-1)  # [B, N, hidden_dim*2]
        fused = self.fusion(fused)  # [B, N, output_dim]
        
        # 注意力池化
        attn_scores = self.attention_pool(fused)  # [B, N, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        global_feat = torch.sum(fused * attn_weights, dim=1)  # [B, output_dim]
        
        return global_feat