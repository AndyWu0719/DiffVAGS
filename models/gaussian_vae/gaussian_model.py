#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl 
import numpy as np
import math

from einops import rearrange, reduce
from models.gaussian_vae.gaussian_encoder import GaussianEncoder
from models.gaussian_vae.gaussian_decoder import ColorDecoder, OccDecoder, TransformDecoder
from conv_pointlite import ConvPointnet

class GaussianModel(pl.LightningModule):

    def __init__(self, specs):
        super().__init__()
        
        self.specs = specs
        model_specs = specs.get("GaussianModelSpecs", {})
        self.hidden_dim = model_specs.get("hidden_dim", 512)
        self.latent_dim = model_specs.get("latent_dim", 256) 
        self.skip_connection = model_specs.get("skip_connection", True)
        self.tanh_act = model_specs.get("tanh_act", False)
        self.pn_hidden = model_specs.get("pn_hidden_dim", self.latent_dim)

        self.enable_vavae = model_specs.get("enable_vavae", False)
        self.foundation_model_type = model_specs.get("foundation_model_type", "dinov2")

        if self.enable_vavae:
            print(f"Initializing VAVAE mode with {self.foundation_model_type.upper()}")
            assert self.foundation_model_type in ['dinov2', 'mae'], f"Unsupported foundation model: {self.foundation_model_type}"
        else:
            print("Initializing standard VAE mode")

        self.pointnet = ConvPointnet(
            c_dim=self.latent_dim,
            dim=59,
            hidden_dim=self.pn_hidden,
            plane_resolution=64,
        )

        decoder_kwargs = {
            "latent_size": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "skip_connection": self.skip_connection,
            "tanh_act": self.tanh_act,
        }

        self.transform_decoder = TransformDecoder(**decoder_kwargs)
        self.occ_decoder = OccDecoder(**decoder_kwargs)
        self.color_decoder = ColorDecoder(**decoder_kwargs)

        self.occ_decoder.train()
        self.color_decoder.train()

            
    def forward(self, pc, gs):
        shape_features = self.pointnet(pc, gs)

        return self.transform_decoder(gs, shape_features).squeeze()

    def forward_with_plane_features(self, plane_features, gs):
        gs = gs[:,:,:3]
        point_features = self.pointnet.forward_with_plane_features(plane_features, gs) # point_features: B, N, D
        pred_color = self.color_decoder(torch.cat((gs, point_features),dim=-1))
        pred_gs = self.transform_decoder(torch.cat((gs, point_features),dim=-1))
        return pred_color, pred_gs # [B, num_points] 
    

    def forward_with_plane_features_occ(self, plane_features, gs):
        point_features = self.pointnet.forward_with_plane_features(plane_features, gs) # point_features: B, N, D
        pred_occ = self.occ_decoder(torch.cat((gs, point_features),dim=-1))  
        return pred_occ # [B, num_points] 
    
    def get_model_type(self):
        if self.enable_vavae:
            return f"VAVAE_{self.foundation_model_type.upper()}"
        else:
            return "VAE"