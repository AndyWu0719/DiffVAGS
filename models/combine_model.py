#!/usr/bin/env python3

import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from einops import reduce

from models.gaussian_vae.gaussian_model import GaussianModel
from models.gaussian_vae.gaussian_encoder import GaussianEncoder
from models.diffusion.diff_model import DiffusionModel
from models.diffusion.condition_net import DiffusionNet

from models.visual_alignment.g2f import GaussianToFeature
from models.visual_alignment.foundation_models import aux_foundation_model
from models.visual_alignment.visual_alignment import VisualAlignmentLoss

class CombinedModel(pl.LightningModule):
    def __init__(self, specs):
        super().__init__()
        self.specs = specs
        self.task = specs['training_task']

        print(f"Initializing CombinedModel for task: {self.task}")

        va_specs = self.specs.get("VisualAlignmentSpecs", {})
        self.enable_visual_alignment = va_specs.get("enable", False)

        if self.task in ('combined', 'modulation'):
            self.gaussian_model = GaussianModel(specs=specs)
            
            model_specs = specs.get("GaussianModelSpecs", {})
            feature_dim = model_specs.get("latent_dim", 256)
            modulation_dim = feature_dim * 3
            latent_std = self.specs.get("latent_std", 0.25) # 从顶层读取
            hidden_dims = [modulation_dim] * 5
            self.vae_model = GaussianEncoder(in_channels=feature_dim*3, latent_dim=modulation_dim, hidden_dims=hidden_dims, kl_std=latent_std)

            print(f"{'VA' if self.enable_visual_alignment else 'Standard'} mode")

            # --- 2. 初始化视觉对齐模块 (如果配置中启用) ---
            if self.enable_visual_alignment:
                print("Visual Alignment is ENABLED. Initializing modules...")
                
                # 初始化 G2F 模块 (用于从高斯参数预测特征)
                self.g2f_module = GaussianToFeature(
                    checkpoint_path=model_specs["g2f_checkpoint_path"],
                    device=self.device
                )
                
                # 初始化 2D VFM (用于从真实图像提取监督信号)，它会自动加载 adapter 的权重
                self.feature_extractor = aux_foundation_model(
                    type='dinov2',
                    adapter_weights_path=model_specs["adapter_weights_path"]
                )
                
                # 初始化 VF Loss 计算模块
                self.vf_loss_criterion = VisualAlignmentLoss(
                    cosine_weight=model_specs.get("cosine_weight", 0.5),
                    distance_weight=model_specs.get("distance_weight", 0.5)
                )
                
                # 获取 VF Loss 在总损失中的权重
                self.vf_loss_weight = model_specs.get("vf_loss_weight", 1.0)
                print("✅ Visual Alignment modules initialized successfully.")

        if self.task in ('combined', 'diffusion'):
            if "diffusion_specs" in specs and "diffusion_model_specs" in specs:
                diff_specs = specs["diffusion_specs"]
                model_specs = specs["diffusion_model_specs"]
                
                condition_net = DiffusionNet(**{
                    **model_specs,
                    "num_timesteps": diff_specs["timesteps"]
                })
                
                self.diffusion_model = DiffusionModel(
                    model=condition_net,
                    **diff_specs
                )
                print("Diffusion model initialized")

    def training_step(self, x, idx):
        if self.task == 'combined':
            return self.train_combined(x)
        elif self.task == 'modulation':
            return self.train_modulation(x)
        elif self.task == 'diffusion':
            return self.train_diffusion(x)

    def configure_optimizers(self):
        learning_rates = self.specs["learning_rates"]
        
        if self.task == 'combined':
            params_list = [
                {'params': list(self.gaussian_model.parameters()) + list(self.vae_model.parameters()), 
                 'lr': learning_rates['sdf_lr']},
                {'params': self.diffusion_model.parameters(), 
                 'lr': learning_rates['diff_lr']}
            ]
        elif self.task == 'modulation':
            params_list = [{'params': self.parameters(), 'lr': learning_rates['sdf_lr']}]
        elif self.task == 'diffusion':
            params_list = [{'params': self.parameters(), 'lr': learning_rates['diff_lr']}]

        optimizer = torch.optim.Adam(params_list)
        return {"optimizer": optimizer}

    def train_modulation(self, x):
        occ_xyz = x['occ_xyz']
        occ = x['occ']
        gt = x['gt_gaussian']
        gs = x['gaussians']
        gaussian_xyz = x['gaussian_xyz']

        plane_features = self.gaussian_model.pointnet.get_plane_features(gs)
        original_features = torch.cat(plane_features, dim=1)

        out = self.vae_model(original_features)

        reconstructed_plane_feature, latent = out[0], out[1]

        pred_color, pred_gs = self.gaussian_model.forward_with_plane_features(reconstructed_plane_feature, gaussian_xyz)
        pred_occ = self.gaussian_model.forward_with_plane_features_occ(reconstructed_plane_feature, occ_xyz)
        
        try:
            vae_loss_dict = self.vae_model.loss_function(*out, M_N=self.specs["kld_weight"])
            vae_loss = vae_loss_dict['VAEloss']
        except Exception as e:
            print(f"VAE loss failed at epoch {self.current_epoch}: {e}")
            return None

        color_loss = F.l1_loss(pred_color[:,:,0:48], gt[:,:,0:48], reduction='none')
        color_loss = reduce(color_loss, 'b ... -> b (...)', 'mean').mean()

        scale_loss = F.l1_loss(pred_gs[:,:,0:3], gt[:,:,49:52], reduction='none')
        scale_loss = reduce(scale_loss, 'b ... -> b (...)', 'mean').mean()

        rotation_loss = F.l1_loss(pred_gs[:,:,3:7], gt[:,:,52:56], reduction='none')
        rotation_loss = reduce(rotation_loss, 'b ... -> b (...)', 'mean').mean()

        occ_loss = F.l1_loss(pred_occ.squeeze(-1), occ.squeeze(-1), reduction='none')
        occ_loss = reduce(occ_loss, 'b ... -> b (...)', 'mean').mean()

        # (D) 将所有损失相加
        total_loss = color_loss + vae_loss + occ_loss + scale_loss + rotation_loss

        # (E) 记录所有损失用于监控
        loss_dict = {
            "loss": total_loss,
            "color": color_loss,
            "vae_total": vae_loss,
            "occ": occ_loss,
            "scale": scale_loss,
            "rotation": rotation_loss,
            "kld": vae_loss_dict.get('KLD_total', 0)
        }
        
        # 记录 VF Loss 的详细分解
        if self.enable_visual_alignment:
            # 计算 VF Loss
            predicted_view_features = self.g2f_module(gaussian_xyz, pred_color, pred_gs, pred_occ)
            target_view_features = {
                view: self.feature_extractor.extract_global_features(img.unsqueeze(0) if img.dim() == 3 else img)
                for view, img in x['multiview_images'].items()
            }
            vf_loss_details = self.vf_loss_criterion(predicted_view_features, target_view_features)
            vf_loss = vf_loss_details.get('vf_loss', torch.tensor(0.0, device=self.device))
            
            # 将 VF Loss 添加到总损失
            total_loss += self.vf_loss_weight * vf_loss
            loss_dict['loss'] = total_loss # 更新总损失

            # 记录所有视觉对齐相关的损失
            loss_dict.update({
                "vf_loss": vf_loss,
                "vf_cosine": vf_loss_details.get('avg_cosine_loss', 0),
                "vf_distance": vf_loss_details.get('avg_distance_loss', 0),
            })

        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)
        return total_loss

    def train_diffusion(self, x):
        self.train()
        
        latent = x['latent']  # (B, D)
        
        if self.specs['diffusion_model_specs']['cond']:
            cond = x.get('gaussians', None)
        else:
            cond = None

        diff_loss, diff_100_loss, diff_1000_loss, _, __ = self.diffusion_model.diffusion_model_from_latent(
            latent, cond=cond
        )

        loss_dict = {
            "total": diff_loss,
            "diff100": diff_100_loss,
            "diff1000": diff_1000_loss
        }
        
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)
        return diff_loss