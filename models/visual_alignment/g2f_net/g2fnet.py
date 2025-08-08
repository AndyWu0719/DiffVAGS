import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os
import numpy as np
from typing import Dict, List
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from point_encoder import LightPointEncoder
from feature_extractor import EfficientFeatureExtractor
from gs_dataloader import MultiViewGaussianDataset

class EfficientFeaturePredictor(nn.Module):
    
    def __init__(self, 
                 spatial_dim: int = 768, 
                 feature_dim: int = 768):
        super().__init__()
        
        self.point_encoder = LightPointEncoder(
            point_dim=3, 
            attr_dim=56,
            hidden_dim=512,
            output_dim=spatial_dim
        )
        
        self.feature_aligner = nn.Sequential(
            nn.Linear(spatial_dim, spatial_dim),
            nn.GELU(),
            nn.LayerNorm(spatial_dim),
            nn.Dropout(0.1),
            nn.Linear(spatial_dim, feature_dim)
        )
        
        self.view_projectors = nn.ModuleDict({
            view: nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim * 2, feature_dim)
            ) for view in ['front', 'side', 'top']
        })
    
    def forward(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        points = batch_data['gaussian_xyz']  # [B, N, 3]
        attributes = batch_data['gt_gaussian']  # [B, N, 56]
        
        spatial_feat = self.point_encoder(points, attributes)  # [B, spatial_dim]
        
        aligned_feat = self.feature_aligner(spatial_feat)  # [B, feature_dim]
        
        view_features = {}
        for view, projector in self.view_projectors.items():
            view_features[f'{view}_features'] = projector(aligned_feat)
        
        return view_features

class FocusedFeatureLoss(nn.Module):
    
    def __init__(self, views: List[str] = ['front', 'side', 'top']):
        super().__init__()
        self.views = views
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses = {}
        total_loss = 0.0
        
        for view in self.views:
            pred_key = f'{view}_features'
            target_key = f'{view}_features'
            
            if pred_key in pred and target_key in target:
                p = pred[pred_key]
                t = target[target_key]
                
                mse_loss = self.mse(p, t)
                cos_loss = 1 - F.cosine_similarity(p, t, dim=-1).mean()
                l1_loss = self.l1(p, t)
                
                view_loss = mse_loss + 0.8 * cos_loss + 0.1 * l1_loss
                
                total_loss += view_loss
                losses[f'{view}_loss'] = view_loss
        
        losses['total_loss'] = total_loss
        return losses

def train_model(local_rank, gaussian_path, image_path, epochs, batch_size, lr, num_workers, resume_checkpoint=None):
    device = torch.device(f'cuda:{local_rank}')
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])

    if rank == 0:
        print(f"device: {device}, size: {world_size}")
    
    train_dataset = MultiViewGaussianDataset(
        gaussian_data_path=gaussian_path,
        image_data_path=image_path
    )
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler
    )
    
    model = EfficientFeaturePredictor().to(device)
    feature_extractor = EfficientFeatureExtractor(freeze_backbone=True).to(device)
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = FocusedFeatureLoss()
    
    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scaler = GradScaler()
    
    start_epoch = 0
    best_loss = float('inf')

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        if rank == 0:
            print(f"Load checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        if rank == 0:
            print(f"Complete loading. Start from epoch {start_epoch}, best_loss: {best_loss:.4f}")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        
        epoch_losses = {}
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', disable=(rank != 0))
        for batch in pbar:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast():
                pred_features = model(batch)
                target_features = {}
                with torch.no_grad():
                    for view in ['front', 'side', 'top']:
                        img_key = f'{view}_image'
                        if img_key in batch:
                            target_features[f'{view}_features'] = feature_extractor(batch[img_key])
                losses = criterion(pred_features, target_features)
                loss = losses['total_loss']
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0.0
                epoch_losses[k] += v.item()
            num_batches += 1

            if rank == 0:
                pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        
        scheduler.step()

        avg_losses = {}
        for k, v in epoch_losses.items():
            avg_loss_tensor = torch.tensor([v / num_batches], device=device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
            avg_losses[k] = avg_loss_tensor.item()
        
        if rank == 0:
            print(f'\nEpoch {epoch+1} avg.loss:')
            for k, v in avg_losses.items():
                print(f"  {k}: {v:.6f}")
        
        if rank == 0 and avg_losses['total_loss'] < best_loss:
            best_loss = avg_losses['total_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'all_losses': avg_losses
            }, '/home/dwubf/workplace/DiffVAGS/experiments/g2f/best_model.pth')
            print(f"Save best model, loss: {best_loss:.6f}")

            if rank == 0 and ((epoch + 1) % 50 == 0 or (epoch + 1) == epochs):
                checkpoint_path = f'/home/dwubf/workplace/DiffVAGS/experiments/g2f/checkpoint_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_losses['total_loss'],
                    'all_losses': avg_losses
                }, checkpoint_path)
                print(f"Save checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    train_model()