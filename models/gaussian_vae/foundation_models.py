import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_mae_encoder():
    """
    Load the MAE pretrained ViT-L encoder from the timm library.
    """
    model = timm.create_model("hf-hub:timm/vit_large_patch16_224.mae", pretrained=True, dynamic_img_size=True)
    model.requires_grad_(False)
    return model

def get_dinov2_encoder():
    """
    Load the DINOv2 pretrained ViT-L encoder from the timm library.
    """
    model = timm.create_model("hf-hub:timm/vit_large_patch14_dinov2.lvd142m", pretrained=True, dynamic_img_size=True)
    model.requires_grad_(False)
    return model

def create_foundation_model(
    type,
):
    assert type in ['mae', 'dinov2'], f"Unsupported foundation model type: {type}"

    if type == 'mae':
        return get_mae_encoder(), 1024
    elif type == 'dinov2':
        return get_dinov2_encoder(), 1024

class aux_foundation_model(nn.Module):
    """
    Load the foundation model and forward the input image to get 
    the feature maps.
    """
    def __init__(self, type):
        super().__init__()
        self.model, feature_dim = create_foundation_model(type)
        self.type = type
        self.feature_dim = feature_dim
        
        if type == 'mae':
            self.patch_size = 16
            self.input_size = 224
        elif type == 'dinov2':
            self.patch_size = 14
            self.input_size = 224

    def preprocess_image(self, x):
        if x.shape[1] == 4:
            rgb = x[:, :3, :, :]
            alpha = x[:, 3:4, :, :]
            x = rgb * alpha + (1 - alpha)
        elif x.shape[1] == 3:
            pass
        else:
            raise ValueError(f"Unexpected number of channels: {x.shape[1]}")
        
        if x.max() > 1.0:
            x = x / 255.0
        
        x = F.interpolate(x, size=(self.input_size, self.input_size), 
                         mode='bilinear', align_corners=False)
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        
        return x

    def forward_mae(self, x):
        b, c, h, w = x.shape
        x = self.preprocess_image(x)
        features = self.model.forward_features(x)
        patch_features = features[:, 1:, :]
        feat_h = feat_w = int(patch_features.shape[1] ** 0.5)
        features_map = patch_features.reshape(b, feat_h, feat_w, self.feature_dim)
        features_map = features_map.permute(0, 3, 1, 2)  # [B, feature_dim, feat_h, feat_w]
        
        return features_map
    
    def forward_dinov2(self, x):
        b, c, h, w = x.shape
        
        x = self.preprocess_image(x)
        
        features = self.model.forward_features(x)  # [B, N_patches+1, feature_dim]
        
        patch_features = features[:, 1:, :]  # [B, N_patches, feature_dim]
        
        feat_h = feat_w = int(patch_features.shape[1] ** 0.5)
        
        features_map = patch_features.reshape(b, feat_h, feat_w, self.feature_dim)
        features_map = features_map.permute(0, 3, 1, 2)  # [B, feature_dim, feat_h, feat_w]
        
        return features_map
        
    def forward(self, x):
        with torch.no_grad():
            if self.type == 'mae':
                return self.forward_mae(x)
            elif self.type == 'dinov2':
                return self.forward_dinov2(x)
            
    def extract_global_features(self, x):
        with torch.no_grad():
            x = self.preprocess_image(x)
            
            features = self.model.forward_features(x)  # [B, N_patches+1, feature_dim]
            
            if self.type in ['mae', 'dinov2']:
                global_features = features[:, 0, :]  # [B, feature_dim]
            else:
                patch_features = features[:, 1:, :]
                global_features = torch.mean(patch_features, dim=1)  # [B, feature_dim]
            
            return global_features