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
    
    def __init__(self, 
                 model_name: str = "vit_base_patch14_dinov2.lvd142m",
                 feature_dim: int = 768,
                 freeze_backbone: bool = True):
        super().__init__()
        
        self.backbone = timm.create_model(
            model_name, 
            pretrained=True,
            num_classes=0
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.transform = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:

        with torch.cuda.amp.autocast():
            features = self.backbone(images)
        
        return features
    
    @torch.no_grad()
    def extract_view_features(self, image_path: str) -> torch.Tensor:

        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        return self.forward(image_tensor)