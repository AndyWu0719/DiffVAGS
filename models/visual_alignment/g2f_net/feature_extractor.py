import torch
import torch.nn as nn
import timm
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, List

import torch
import torch.nn as nn
import timm
from PIL import Image
import torchvision.transforms as transforms

class EfficientFeatureExtractor(nn.Module):
    """高效图像特征提取器"""
    
    def __init__(self, 
                 model_name: str = "vit_small_patch14_dinov2.lvd142m",  # 更小更快的模型
                 feature_dim: int = 384,
                 freeze_backbone: bool = True):
        super().__init__()
        
        # 加载预训练视觉模型
        self.backbone = timm.create_model(
            model_name, 
            pretrained=True,
            num_classes=0  # 移除分类头
        )
        
        # 冻结主干
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 特征投影层
        self.feature_adapter = nn.Sequential(
            nn.Linear(self.backbone.num_features, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((518, 518)),  # 更小的尺寸
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, H, W] 输入图像
        Returns:
            torch.Tensor: [B, feature_dim] 特征向量
        """
        # 使用混合精度
        with torch.cuda.amp.autocast():
            features = self.backbone(images)
        
        # 投影到目标维度
        return self.feature_adapter(features)
    
    @torch.no_grad()
    def extract_view_features(self, image_path: str) -> torch.Tensor:
        """从单张图像提取特征"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)  # [1, 3, H, W]
        return self.forward(image_tensor)