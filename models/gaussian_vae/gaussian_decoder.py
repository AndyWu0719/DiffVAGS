import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from torch.autograd import Function

inverse_sigmoid = lambda x: np.log(x / (1 - x))

class _TruncExp(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)
    @staticmethod
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))
trunc_exp = _TruncExp.apply

class TransformLayer(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.feature_channels = {"scaling": 3, "rotation": 4}
        self.clip_scaling = 0.2
        self.init_scaling = -5.0
        self.init_density = 0.1

        self.out_layers = nn.ModuleList()
        for key, out_ch in self.feature_channels.items():
            layer = nn.Linear(hidden_dim, out_ch)

            if key == "scaling":
                nn.init.constant_(layer.bias, self.init_scaling)
            elif key == "rotation":
                nn.init.constant_(layer.bias[0], 1.0)
            else:
                nn.init.constant_(layer.weight, 0)
                nn.init.constant_(layer.bias, 0)

            self.out_layers.append(layer)

    def forward(self, x):
        ret = {}
        for k, layer in zip(self.feature_channels.keys(), self.out_layers):
            v = layer(x)
            if k == "rotation":
                v = F.normalize(v, dim=-1, eps=1e-6)
            elif k == "scaling":
                v = trunc_exp(v)
                if self.clip_scaling is not None:
                    v = torch.clamp(v, min=0, max=self.clip_scaling)
            ret[k] = v
        return ret
    

class ColorLayer(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.feature_channels = {"shs": 48}
        self.clip_scaling = 0.2
        self.init_scaling = -5.0
        self.init_density = 0.1

        self.out_layers = nn.ModuleList()
        for key, out_ch in self.feature_channels.items():
            layer = nn.Linear(hidden_dim, out_ch)

            if key == "scaling":
                nn.init.constant_(layer.bias, self.init_scaling)
            elif key == "rotation":
                nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                nn.init.constant_(layer.bias, inverse_sigmoid(self.init_density))
            else:
                nn.init.constant_(layer.weight, 0)
                nn.init.constant_(layer.bias, 0)

            self.out_layers.append(layer)

    def forward(self, x):
        ret = {}
        for k, layer in zip(self.feature_channels.keys(), self.out_layers):
            v = layer(x)
            if k == "rotation":
                v = torch.nn.functional.normalize(v, dim=-1)
            elif k == "scaling":
                v = trunc_exp(v)
                if self.clip_scaling is not None:
                    v = torch.clamp(v, min=0, max=self.clip_scaling)
            elif k == "opacity":
                v = torch.sigmoid(v)
            ret[k] = v
        return ret
    

class TransformDecoder(nn.Module):
    def __init__(self, 
                 latent_size=256, 
                 hidden_dim=512, 
                 input_size=None, 
                 skip_connection=True, 
                 tanh_act=False, 
                 geo_init=True
                 ):
        super().__init__()
        self.latent_size = latent_size
        self.input_size = latent_size + 3 if input_size is None else input_size
        self.skip_connection = skip_connection
        self.tanh_act = tanh_act

        skip_dim = hidden_dim + self.input_size if skip_connection else hidden_dim

        self.block1 = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.block2 = nn.Sequential(
            nn.Linear(skip_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.trans_layer = TransformLayer(hidden_dim=hidden_dim)

    def forward(self, x):
        # x: [B, N, D + 3]
        block1_out = self.block1(x)

        if self.skip_connection:
            block2_in = torch.cat([x, block1_out], dim=-1) 
        else:
            block2_in = block1_out

        block2_out = self.block2(block2_in)
        out_ret = self.trans_layer(block2_out)

        out = torch.cat((out_ret['scaling'], out_ret['rotation']), dim=-1)  # [B, N, 7]
        return out  # GauTF


class ColorDecoder(nn.Module):
    def __init__(self, 
                 latent_size=256, 
                 hidden_dim=512, 
                 input_size=None, 
                 skip_connection=True, 
                 tanh_act=False, 
                 geo_init=True
                 ):
        super().__init__()
        self.latent_size = latent_size
        self.input_size = latent_size + 3 if input_size is None else input_size
        self.skip_connection = skip_connection
        self.tanh_act = tanh_act

        skip_dim = hidden_dim + self.input_size if skip_connection else hidden_dim

        self.block1 = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.block2 = nn.Sequential(
            nn.Linear(skip_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        self.color_layer = ColorLayer(hidden_dim=hidden_dim)

    def forward(self, x):
        block1_out = self.block1(x)

        if self.skip_connection:
            block2_in = torch.cat([x, block1_out], dim=-1)
        else:
            block2_in = block1_out

        block2_out = self.block2(block2_in)
        out_ret = self.color_layer(block2_out)

        out = out_ret['shs']   # [B, N, 48]
        return out  # GauCF
    

class OccDecoder(nn.Module):
    def __init__(self, 
                 latent_size=256, 
                 hidden_dim=512, 
                 input_size=None, 
                 skip_connection=True, 
                 tanh_act=False, 
                 geo_init=True
                 ):
        super().__init__()
        self.latent_size = latent_size
        self.input_size = latent_size + 3 if input_size is None else input_size
        self.skip_connection = skip_connection
        self.tanh_act = tanh_act

        skip_dim = hidden_dim + self.input_size if skip_connection else hidden_dim

        self.block1 = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Linear(skip_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )


        self.block3 = nn.Linear(hidden_dim, 1)
        if geo_init:
            for m in self.block3.modules():
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, mean=2 * np.sqrt(np.pi) / np.sqrt(hidden_dim), std=0.000001)
                    init.constant_(m.bias, -0.5)

            for m in self.block2.modules():
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, mean=0.0, std=np.sqrt(2) / np.sqrt(hidden_dim))
                    init.constant_(m.bias, 0.0)

            for m in self.block1.modules():
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, mean=0.0, std=np.sqrt(2) / np.sqrt(hidden_dim))
                    init.constant_(m.bias, 0.0)

    def forward(self, x):
        block1_out = self.block1(x)

        if self.skip_connection:
            block2_in = torch.cat([x, block1_out], dim=-1)
        else:
            block2_in = block1_out

        block2_out = self.block2(block2_in)
        out = self.block3(block2_out)  # [B, N, 1]

        if self.tanh_act:
            out = nn.Tanh()(out)

        out = torch.abs(out)
        return out  # GauPF
