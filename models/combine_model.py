#!/usr/bin/env python3

import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from einops import reduce

from models.gaussian_vae.gaussian_model import GaussianModel
from models.gaussian_vae.gaussian_encoder import GaussianEncoder
from models.diffusion.diff_model import DiffusionModel
from models.diffusion.condition_net import DiffusionNet

class CombinedModel(pl.LightningModule):
    def __init__(self, specs):
        super().__init__()
        self.specs = specs
        self.task = specs['training_task']

        print(f"Initializing CombinedModel for task: {self.task}")

        if self.task in ('combined', 'modulation'):
            self.gaussian_model = GaussianModel(specs=specs)
            feature_dim = specs.get("GaussianModelSpecs", {}).get("latent_dim", 256)
            modulation_dim = feature_dim * 3
            latent_std = specs.get("GaussianModelSpecs", {}).get("latent_std", 0.25)
            hidden_dims = [modulation_dim, modulation_dim, modulation_dim, modulation_dim, modulation_dim]
            self.vae_model = GaussianEncoder(in_channels=feature_dim*3, latent_dim=modulation_dim, hidden_dims=hidden_dims, kl_std=latent_std, 
                                             enable_vavae=specs.get("GaussianModelSpecs", {}).get("enable_vavae", False))
            self.enable_vavae = specs.get("GaussianModelSpecs", {}).get("enable_vavae", False)

            print(f"{'VAVAE' if self.enable_vavae else 'Standard VAE'} mode")

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

        if self.enable_vavae:
            multiview_images = x.get('multiview_images', None)
            has_multiview = x.get('has_multiview_data', False)
            
            if has_multiview and multiview_images is not None:
                out = self.vae_model(original_features, multiview_images)
            else:
                print("Warning: VAVAE mode but no multiview images in batch")
                return None
        else:
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

        total_loss = color_loss + vae_loss + occ_loss + scale_loss + rotation_loss

        loss_dict = {
            "loss": total_loss,
            "color": color_loss,
            "vae": vae_loss,
            "occ": occ_loss,
            "scale": scale_loss,
            "rotation": rotation_loss
        }
        
        if self.enable_vavae:
            loss_dict.update({
                "kld_geo": vae_loss_dict.get('KLD_geo', torch.tensor(0.0)),
                "kld_sem": vae_loss_dict.get('KLD_sem', torch.tensor(0.0)),
                "kld_total": vae_loss_dict.get('KLD_total', torch.tensor(0.0)),
                "vfloss": vae_loss_dict.get('VF_Loss', torch.tensor(0.0)),
                "reconstruction": vae_loss_dict.get('Reconstruction_Loss', torch.tensor(0.0)),
                "vf_pointwise_loss": vae_loss_dict.get('VF_Pointwise_Loss', torch.tensor(0.0)),
                "vf_distance_loss": vae_loss_dict.get('VF_Distance_Loss', torch.tensor(0.0)),
                "base_vae_loss": vae_loss_dict.get('Base_VAE_Loss', torch.tensor(0.0)),
                "interaction_strength": vae_loss_dict.get('Interaction_Strength', torch.tensor(0.5)),
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