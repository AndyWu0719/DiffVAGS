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
            latent_std = self.specs.get("latent_std", 0.25)
            hidden_dims = [modulation_dim] * 5
            self.vae_model = GaussianEncoder(in_channels=feature_dim*3, latent_dim=modulation_dim, hidden_dims=hidden_dims, kl_std=latent_std)

            print(f"{'VA' if self.enable_visual_alignment else 'Standard'} mode")

            if self.enable_visual_alignment:
                print("Visual Alignment is ENABLED. Initializing modules...")
                
                self.g2f_module = GaussianToFeature(
                    checkpoint_path=va_specs["g2f_checkpoint_path"],
                    device=self.device
                )
                
                self.feature_extractor = aux_foundation_model(
                    type='dinov2'
                )
                
                self.vf_loss_criterion = VisualAlignmentLoss(
                    cosine_weight=va_specs.get("cosine_weight", 0.5),
                    distance_weight=va_specs.get("distance_weight", 0.5)
                )
                
                self.vf_loss_weight = va_specs.get("vf_loss_weight", 1.0)
                print("Visual Alignment modules initialized successfully.")

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

        vae_loss_dict = self.vae_model.loss_function(*out, M_N=self.specs["kld_weight"], global_step=self.global_step)
        vae_loss = vae_loss_dict['VAEloss']
        kld_unweighted = vae_loss_dict.get('kld_unweighted')

        reconstructed_plane_feature, latent = out[0], out[1]

        pred_color, pred_gs = self.gaussian_model.forward_with_plane_features(reconstructed_plane_feature, gaussian_xyz)
        pred_occ = self.gaussian_model.forward_with_plane_features_occ(reconstructed_plane_feature, occ_xyz)

        color_loss = F.l1_loss(pred_color[:,:,0:48], gt[:,:,0:48], reduction='none')
        color_loss = reduce(color_loss, 'b ... -> b (...)', 'mean').mean()

        scale_loss = F.l1_loss(pred_gs[:,:,0:3], gt[:,:,49:52], reduction='none')
        scale_loss = reduce(scale_loss, 'b ... -> b (...)', 'mean').mean()

        rotation_loss = F.l1_loss(pred_gs[:,:,3:7], gt[:,:,52:56], reduction='none')
        rotation_loss = reduce(rotation_loss, 'b ... -> b (...)', 'mean').mean()

        occ_loss = F.l1_loss(pred_occ.squeeze(-1), occ.squeeze(-1), reduction='none')
        occ_loss = reduce(occ_loss, 'b ... -> b (...)', 'mean').mean()

        total_loss = color_loss + vae_loss + occ_loss + scale_loss + rotation_loss

        loss_dict = {
            "loss": total_loss,
            "color": color_loss,
            "vae_total": vae_loss,
            "occ": occ_loss,
            "scale": scale_loss,
            "rotation": rotation_loss,
            "kld": kld_unweighted
        }
        
        if self.enable_visual_alignment:
            predicted_view_features = self.g2f_module(gaussian_xyz, pred_color, pred_gs, pred_occ)
            target_images_batch = []
            view_order = []
            
            for view in self.vf_loss_criterion.views:
                img_key = f'{view}_image'
                if img_key in x:
                    target_images_batch.append(x[img_key])
                    view_order.append(view)
            
            if target_images_batch:
                stacked_images = torch.cat(target_images_batch, dim=0)
                
                extracted_features = self.feature_extractor(stacked_images)
                
                target_view_features = {
                    view: extracted_features[i].unsqueeze(0) for i, view in enumerate(view_order)
                }
            else:
                target_view_features = {}
            vf_loss_details = self.vf_loss_criterion(predicted_view_features, target_view_features)
            vf_loss = vf_loss_details.get('vf_loss', torch.tensor(0.0, device=self.device))
            
            total_loss += self.vf_loss_weight * vf_loss
            loss_dict['loss'] = total_loss

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

        self.log("diff_total", diff_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("diff_100", diff_100_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("diff_1000", diff_1000_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

        return diff_loss