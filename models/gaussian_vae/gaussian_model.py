#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl 
import numpy as np
import math

from einops import rearrange, reduce
from models.gaussian_vae.gaussian_encoder import GaussianEncoder
from models.gaussian_vae.gaussian_decoder import TriplaneDecoder, ColorDecoder, ProbabilityDecoder, TransformDecoder
from input_encoder.conv_pointlite import ConvPointnetLite

class GaussianModel(pl.LightningModule):

    def __init__(self, specs):
        super().__init__()
        
        self.specs = specs
        model_specs = self.specs["GaussianModelSpecs"]
        self.hidden_dim = model_specs["hidden_dims"]
        self.latent_dim = model_specs["latent_dim"]
        self.skip_connection = model_specs.get("skip_connection", True)
        self.tanh_act = model_specs.get("tanh_act", False)
        self.pn_hidden = model_specs.get("pn_hidden_dim", 128)
        fusion_outfeatures = model_specs.get("fusion_outfeatures", 512)
        fusion_outchannels = model_specs.get("fusion_outchannels", 128)

        self.pointnet = ConvPointnetLite(
            feature_dim=self.pn_hidden,  # 🔧 使用pn_hidden而不是latent_dim
            input_dim=3,                 # 🔧 3D坐标，不是59
            hidden_dim=self.pn_hidden, 
            plane_resolution=64, 
            plane_types=['xy', 'xz', 'yz']
        )

        decoder_input_size = 3 + self.pn_hidden  # 坐标(3) + 点特征(pn_hidden)

        self.color_decoder = ColorDecoder(
            latent_size=self.pn_hidden,
            hidden_dim=self.hidden_dim[-1],
            input_size=decoder_input_size,
            skip_connection=self.skip_connection,
            tanh_act=self.tanh_act
        )
        
        self.probability_decoder = ProbabilityDecoder(
            latent_size=self.pn_hidden,
            hidden_dim=self.hidden_dim[-1],
            input_size=decoder_input_size,
            skip_connection=self.skip_connection,
            tanh_act=self.tanh_act
        )
        
        self.transform_decoder = TransformDecoder(
            latent_size=self.pn_hidden,
            hidden_dim=self.hidden_dim[-1],
            input_size=decoder_input_size,
            skip_connection=self.skip_connection,
            tanh_act=self.tanh_act
        )

        self.color_decoder.train()
        self.probability_decoder.train()
        self.transform_decoder.train()

        # Gaussian VAE Encoder
        self.gaussian_encoder = GaussianEncoder(
            in_channels=fusion_outchannels,  # 768
            latent_dim=self.latent_dim,      # 512
            hidden_dims=self.hidden_dim,     # [16, 24, 40]
            kl_std=1.0,
            beta=4,
            gamma=10.,
            max_capacity=25,
            capacity_max_iteration=1e5,
            loss_type='B'
        )

        # 使用 TriplaneDecoder，将隐向量转换为三分支输出
        self.triplane_decoder = TriplaneDecoder(
            latent_dim=self.latent_dim, 
            grid_size=(64, 64), 
            hidden_dim=self.hidden_dim[-1]
        )
            
    def forward(self, fused_features):
        encoded_features = self.gaussian_encoder(fused_features)
        recon_x, input_data, mu, logvar, z = encoded_features

        gau_pf, gau_cf, gau_tf = self.triplane_decoder(z)

        return gau_pf, gau_cf, gau_tf

    def forward_with_plane_features(self, plane_features, query_xyz):
        """
        当输入包含 plane_features 时，
        用 multi_modal_encoder 提取融合特征后映射到 latent 空间，再解码出颜色和几何变换信息。
        """
        device = plane_features.device
        query_xyz = query_xyz.to(device)
        # 1. 从三平面特征中提取点特征
        point_features = self.pointnet.forward_with_plane_features(plane_features, query_xyz)
        # point_features: [B, N, pn_hidden]

        point_features = point_features.to(device)
        query_xyz = query_xyz.to(device)
        
        # 2. 🔧 关键修复：拼接坐标和特征 (仿照GsModel)
        combined_features = torch.cat((query_xyz, point_features), dim=-1)
        # combined_features: [B, N, 3 + pn_hidden]
        
        # 3. 使用独立解码器预测
        pred_color = self.color_decoder(combined_features)     # [B, N, 48]
        pred_transform = self.transform_decoder(combined_features)  # [B, N, 7]
        
        return pred_color, pred_transform

    def forward_with_plane_features_pf(self, plane_features, query_xyz):
        """
        使用 plane_features 时，专门提取密度/概率信息。
        """
        device = plane_features.device
        query_xyz = query_xyz.to(device)
        # 1. 从三平面特征中提取点特征
        point_features = self.pointnet.forward_with_plane_features(plane_features, query_xyz)
        
        # 2. 🔧 拼接坐标和特征
        combined_features = torch.cat((query_xyz, point_features), dim=-1)
        
        # 3. 占用预测
        pred_occ = self.probability_decoder(combined_features)  # [B, N, 1]
        
        return pred_occ
    
    def forward_with_plane_features_occ(self, plane_features, query_xyz):
        """🔧 添加：与GsModel兼容的占用预测接口"""
        return self.forward_with_plane_features_pf(plane_features, query_xyz)
    
    def get_vae_loss(self, fused_features, **kwargs):
        """
        🔧 新增：获取VAE损失
        Args:
            fused_features: [B, 768, 64, 64] 多模态融合特征
        Returns:
            dict: VAE损失字典
        """
        vae_outputs = self.gaussian_encoder(fused_features)
    
        # 🔧 修复：确保传递正确的参数
        if 'minibatch_weight' not in kwargs:
            kwargs['minibatch_weight'] = 1.0
            
        vae_loss_dict = self.gaussian_encoder.loss_function(*vae_outputs, **kwargs)
        return vae_loss_dict
    
    def get_latent_vector(self, fused_features):
        """
        🔧 新增：获取潜在向量
        Args:
            fused_features: [B, 768, 64, 64] 多模态融合特征
        Returns:
            z: [B, 512] 潜在向量
        """
        return self.gaussian_encoder.get_latent(fused_features)
    
    def decode_from_latent(self, z):
        """
        🔧 新增：从潜在向量解码三平面特征
        Args:
            z: [B, 512] 潜在向量
        Returns:
            tuple: (gau_pf, gau_cf, gau_tf) 三平面特征
        """
        return self.triplane_decoder(z)
    
    def reconstruct(self, fused_features):
        """
        🔧 新增：重构输入特征
        Args:
            fused_features: [B, 768, 64, 64] 多模态融合特征
        Returns:
            recon_x: [B, 768, 64, 64] 重构特征
        """
        return self.gaussian_encoder.generate(fused_features)