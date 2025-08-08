import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class VisualAlignmentLoss(nn.Module):
    def __init__(self, 
                 cosine_weight: float = 0.5, 
                 distance_weight: float = 0.5):
        super().__init__()
        
        if cosine_weight + distance_weight != 1.0:
            print(f"Warning: Loss weights do not sum to 1.0 (cosine: {cosine_weight}, distance: {distance_weight})")
            
        self.cosine_weight = cosine_weight
        self.distance_weight = distance_weight
        
        self.views = ['front', 'side', 'top']
        
    def forward(self, 
                predicted_features: Dict[str, torch.Tensor], 
                target_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        total_cosine_loss = 0.0
        total_distance_loss = 0.0
        num_valid_views = 0
        
        per_view_losses = {}

        for view in self.views:
            if view not in predicted_features or view not in target_features:
                continue
            
            pred_feat = predicted_features[view]
            target_feat = target_features[view]
            
            pred_feat = F.normalize(pred_feat, p=2, dim=-1)
            target_feat = F.normalize(target_feat, p=2, dim=-1)

            cosine_loss_view = (1 - F.cosine_similarity(pred_feat, target_feat, dim=-1)).mean()
            
            distance_loss_view = F.mse_loss(pred_feat, target_feat)
            
            total_cosine_loss += cosine_loss_view
            total_distance_loss += distance_loss_view
            num_valid_views += 1
            
            per_view_losses[f'cosine_{view}'] = cosine_loss_view.detach()
            per_view_losses[f'distance_{view}'] = distance_loss_view.detach()

        if num_valid_views == 0:
            return {
                'vf_loss': torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=True),
                'avg_cosine_loss': torch.tensor(0.0),
                'avg_distance_loss': torch.tensor(0.0)
            }

        avg_cosine_loss = total_cosine_loss / num_valid_views
        avg_distance_loss = total_distance_loss / num_valid_views
        
        vf_loss = (self.cosine_weight * avg_cosine_loss) + \
                  (self.distance_weight * avg_distance_loss)
                  
        loss_dict = {
            'vf_loss': vf_loss,
            'avg_cosine_loss': avg_cosine_loss.detach(),
            'avg_distance_loss': avg_distance_loss.detach()
        }
        
        loss_dict.update(per_view_losses)
        
        return loss_dict