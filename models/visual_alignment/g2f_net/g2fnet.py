import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os
import numpy as np
from typing import Dict, List

# 导入自定义模块
from point_encoder import LightPointEncoder
from feature_extractor import EfficientFeatureExtractor
from gs_dataloader import MultiViewGaussianDataset

class EfficientFeaturePredictor(nn.Module):
    """高效多视角特征预测器"""
    
    def __init__(self, 
                 spatial_dim: int = 768, 
                 feature_dim: int = 384):
        super().__init__()
        
        # 点云编码器
        self.point_encoder = LightPointEncoder(
            point_dim=3, 
            attr_dim=56,
            hidden_dim=256,
            output_dim=spatial_dim
        )
        
        # 特征对齐模块
        self.feature_aligner = nn.Sequential(
            nn.Linear(spatial_dim, spatial_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(spatial_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # 视角特定的投影头
        self.view_projectors = nn.ModuleDict({
            view: nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim)
            ) for view in ['front', 'side', 'top']
        })
    
    def forward(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 提取点云数据
        points = batch_data['gaussian_xyz']  # [B, N, 3]
        attributes = batch_data['gt_gaussian']  # [B, N, 56]
        
        # 编码点云
        spatial_feat = self.point_encoder(points, attributes)  # [B, spatial_dim]
        
        # 特征对齐
        aligned_feat = self.feature_aligner(spatial_feat)  # [B, feature_dim]
        
        # 生成多视角特征
        view_features = {}
        for view, projector in self.view_projectors.items():
            view_features[f'{view}_features'] = projector(aligned_feat)
        
        return view_features

class FocusedFeatureLoss(nn.Module):
    """聚焦特征对齐损失"""
    
    def __init__(self, views: List[str] = ['front', 'side', 'top']):
        super().__init__()
        self.views = views
        self.mse = nn.MSELoss()
        self.cos = nn.CosineEmbeddingLoss()
    
    def forward(self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = 0.0
        for view in self.views:
            pred_key = f'{view}_features'
            target_key = f'{view}_features'
            
            if pred_key in pred and target_key in target:
                p = pred[pred_key]
                t = target[target_key]
                
                # MSE损失
                mse_loss = self.mse(p, t)
                
                # 余弦相似度损失
                target_ones = torch.ones(p.size(0)).to(p.device)
                cos_loss = self.cos(p, t, target_ones)
                
                # 组合损失
                loss += mse_loss + 0.3 * cos_loss
        
        return loss

def train_model():
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据集
    train_dataset = MultiViewGaussianDataset(
        gaussian_data_path="/media/andywu/WD6TB/WD6TB/Andy/Datasets/lightdiffgsdata/03001627/convert_data",
        image_data_path="/media/andywu/WD6TB/WD6TB/Andy/Datasets/lightdiffgsdata/03001627/training_data"
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,  # 增大批大小
        shuffle=True,
        num_workers=4,  # 增加工作线程
        pin_memory=True,
        persistent_workers=True  # 保持工作进程
    )
    
    # 创建模型
    model = EfficientFeaturePredictor().to(device)
    feature_extractor = EfficientFeatureExtractor(freeze_backbone=True).to(device)
    
    # 优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = FocusedFeatureLoss()
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 混合精度训练的梯度缩放器
    scaler = GradScaler()
    
    # 训练循环
    num_epochs = 100
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in pbar:
            # 移动数据到设备
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 混合精度训练
            with autocast():
                # 预测特征
                pred_features = model(batch)
                
                # 提取真实特征
                target_features = {}
                for view in ['front', 'side', 'top']:
                    img_key = f'{view}_image'
                    if img_key in batch:
                        target_features[f'{view}_features'] = feature_extractor(batch[img_key])
                
                # 计算损失
                loss = criterion(pred_features, target_features)
            
            # 梯度缩放和反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 更新进度
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        # 计算平均损失
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1} 平均损失: {avg_loss:.4f}')
        
        # 更新学习率
        scheduler.step(avg_loss)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, 'best_model.pth')
            print(f"保存新的最佳模型，损失: {avg_loss:.4f}")

if __name__ == "__main__":
    train_model()