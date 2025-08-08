import os
import torch
import torch.nn as nn
from typing import Dict, Tuple

from .g2f_net.g2fnet import EfficientFeaturePredictor

class GaussianToFeature(nn.Module):
    def __init__(self, 
                 checkpoint_path: str, 
                 device: str = 'cuda',
                 spatial_dim: int = 768, 
                 feature_dim: int = 768):
        super().__init__()
        self.device = torch.device(device)

        print(f"Initializing GaussianToFeature (G2F) module...")

        self.model = EfficientFeaturePredictor(
            spatial_dim=spatial_dim,
            feature_dim=feature_dim
        ).to(self.device)
        
        try:
            print(f"Loading pretrained weights from '{checkpoint_path}'...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            print(f"Success: epoch {checkpoint.get('epoch', 'N/A')}, loss: {checkpoint.get('loss', 'N/A'):.4f}")

        except FileNotFoundError:
            print(f"Error: Checkpoint file '{checkpoint_path}' not found. Model will use randomly initialized weights.")
        except Exception as e:
            print(f"Error: Failed to load weights: {e}. Model will use randomly initialized weights.")

        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False
            
    def _reconstruct_gaussian_attributes(self, 
                                         pred_color: torch.Tensor, 
                                         pred_gs: torch.Tensor, 
                                         pred_occ: torch.Tensor) -> torch.Tensor:
        pred_scale = pred_gs[:, :, :3]
        pred_rotation = pred_gs[:, :, 3:7]
        
        if pred_occ.dim() == 2:
            pred_occ = pred_occ.unsqueeze(-1) # [B, N] -> [B, N, 1]

        attributes = torch.cat([
            pred_occ,       # [:, 3:4]
            pred_color,     # [:, 4:52]
            pred_scale,     # [:, 52:55]
            pred_rotation   # [:, 55:59]
        ], dim=-1)
        
        expected_dim = 1 + 48 + 3 + 4
        assert attributes.shape[-1] == expected_dim, \
            f"Error, expect {expected_dim}, but got {attributes.shape[-1]}"
            
        return attributes

    def forward(self, 
                gaussian_xyz: torch.Tensor,
                pred_color: torch.Tensor, 
                pred_gs: torch.Tensor, 
                pred_occ: torch.Tensor) -> Dict[str, torch.Tensor]:
        gaussian_xyz = gaussian_xyz.to(self.device)
        pred_color = pred_color.to(self.device)
        pred_gs = pred_gs.to(self.device)
        pred_occ = pred_occ.to(self.device)
        
        reconstructed_attributes = self._reconstruct_gaussian_attributes(
            pred_color, pred_gs, pred_occ
        )
        
        batch_data = {
            'gaussian_xyz': gaussian_xyz,
            'gt_gaussian': reconstructed_attributes
        }
        
        predicted_features = self.model(batch_data)
            
        return predicted_features