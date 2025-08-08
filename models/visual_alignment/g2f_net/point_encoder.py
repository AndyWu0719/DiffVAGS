import torch
import torch.nn as nn
import torch.nn.functional as F

class LightPointEncoder(nn.Module):
    
    def __init__(self, 
                 point_dim: int = 3, 
                 attr_dim: int = 56,
                 hidden_dim: int = 512,
                 output_dim: int = 768):
        super().__init__()
        
        self.coord_encoder = nn.Sequential(
            nn.Linear(point_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.attr_encoder = nn.Sequential(
            nn.Linear(attr_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.attention_pool = nn.Linear(output_dim, 1)
    
    def forward(self, points: torch.Tensor, attributes: torch.Tensor) -> torch.Tensor:

        coord_feat = self.coord_encoder(points)  # [B, N, hidden_dim]
        attr_feat = self.attr_encoder(attributes)  # [B, N, hidden_dim]
        
        fused = torch.cat([coord_feat, attr_feat], dim=-1)  # [B, N, hidden_dim*2]
        fused = self.fusion(fused)  # [B, N, output_dim]
        
        attn_scores = self.attention_pool(fused)  # [B, N, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        global_feat = torch.sum(fused * attn_weights, dim=1)  # [B, output_dim]
        
        return global_feat