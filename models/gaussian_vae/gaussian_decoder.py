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
    def __init__(self, hidden_dim=40):
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
                v = F.normalize(v, dim=-1)
            elif k == "scaling":
                v = trunc_exp(v)
                if self.clip_scaling is not None:
                    v = torch.clamp(v, min=0, max=self.clip_scaling)
            ret[k] = v
        return ret
    

class ColorLayer(nn.Module):
    def __init__(self, hidden_dim=40):
        super().__init__()
        self.feature_channels = {"shs": 48}
        self.clip_scaling = 0.2
        self.init_scaling = -5.0
        self.init_density = 0.1
        self.out_layers = nn.ModuleList()
        for key, out_ch in self.feature_channels.items():
            layer = nn.Linear(hidden_dim, out_ch)
            nn.init.constant_(layer.weight, 0)
            nn.init.constant_(layer.bias, 0)
            self.out_layers.append(layer)

    def forward(self, x):
        ret = {}
        for k, layer in zip(self.feature_channels.keys(), self.out_layers):
            v = layer(x)
            ret[k] = v
        return ret
    

class TransformDecoder(nn.Module):
    def __init__(self, latent_size=512, hidden_dim=40, input_size=None, skip_connection=True, tanh_act=False, geo_init=True):
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
        self.gs_layer = TransformLayer(hidden_dim=hidden_dim)

    def forward(self, x):
        # x: [B, N, input_size]
        block1_out = self.block1(x)
        if self.skip_connection:
            block2_in = torch.cat([x, block1_out], dim=-1) 
        else:
            block2_in = block1_out
        block2_out = self.block2(block2_in)
        out_ret = self.gs_layer(block2_out)   # dict with 'scaling' and 'rotation'
        out = torch.cat((out_ret['scaling'], out_ret['rotation']), dim=-1)  # [B, N, 7]
        return out  # GauTF


class ColorDecoder(nn.Module):
    def __init__(self, latent_size=512, hidden_dim=40, input_size=None, skip_connection=True, tanh_act=False, geo_init=True):
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
        out_ret = self.color_layer(block2_out)  # dict with 'shs'
        out = out_ret['shs']   # [B, N, 48]
        return out  # GauCF
    

class ProbabilityDecoder(nn.Module):
    def __init__(self, latent_size=512, hidden_dim=40, input_size=None, skip_connection=True, tanh_act=False, geo_init=True):
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
                    init.normal_(m.weight, mean=2 * np.sqrt(np.pi) / np.sqrt(hidden_dim), std=1e-6)
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
            out = torch.tanh(out)
        out = torch.abs(out)
        return out  # GauPF


# VAE Decoder
class TriplaneDecoder(nn.Module):
    """
    Triplane Decoder: 接受 GaussianEncoder 采样得到的隐向量 z, 
    通过扩展到一个固定的坐标网格, 再利用三个分支分别解码出: 
        GauPF: 概率/密度图；
        GauCF: 颜色图；
        GauTF: 几何变换参数（由 scaling 与 rotation 拼接而成）。
    最终输出的张量尺寸为: [B, C, H, W]。
    """
    def __init__(self, latent_dim=768, grid_size=(64, 64), hidden_dim=40):
        super(TriplaneDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.grid_size = grid_size  # (H, W)
        self.hidden_dim = hidden_dim
        self.num_points = grid_size[0] * grid_size[1]
        
        # 三个解码分支均接受的输入尺寸为 latent_dim + 3（后者为坐标）
        input_size = latent_dim + 3
        self.gs_decoder = TransformDecoder(latent_size=latent_dim, hidden_dim=hidden_dim, input_size=input_size, skip_connection=True)
        self.color_decoder = ColorDecoder(latent_size=latent_dim, hidden_dim=hidden_dim, input_size=input_size, skip_connection=True)
        self.occ_decoder = ProbabilityDecoder(latent_size=latent_dim, hidden_dim=hidden_dim, input_size=input_size, skip_connection=True)
        
        # 构造固定的二维坐标网格, 归一化到 [-1, 1]
        H, W = grid_size
        xs = torch.linspace(-1, 1, steps=W)
        ys = torch.linspace(-1, 1, steps=H)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        
        self.register_buffer('coord_grid', torch.stack([grid_x, grid_y], dim=-1).view(-1, 2))  # [H*W, 2]
        # 为获得 3D 坐标, 增加第三个维度（这里设为 0）
        zeros = torch.zeros(self.coord_grid.shape[0], 1)
        self.register_buffer('coord_grid_3d', torch.cat([self.coord_grid, zeros], dim=-1))  # [H*W, 3]
        
    def forward(self, z):
        """
        Args:
            z: 隐向量, [B, latent_dim]
        返回: 
            gau_pf: 概率图 [B, 1, H, W]
            gau_cf: 颜色图 [B, 48, H, W]
            gau_tf: 变换图 [B, 7, H, W]
        """
        B = z.size(0)
        # 将隐向量扩展到每个网格点: [B, latent_dim] -> [B, num_points, latent_dim]
        z_expanded = z.unsqueeze(1).expand(B, self.num_points, self.latent_dim)
        # 坐标: [num_points, 3] -> [B, num_points, 3]
        coords = self.coord_grid_3d.unsqueeze(0).expand(B, self.num_points, 3)
        # 拼接后输入解码器: [B, num_points, latent_dim+3]
        decoder_input = torch.cat([z_expanded, coords], dim=-1)
        
        gau_tf = self.gs_decoder(decoder_input)      # [B, num_points, 7]
        gau_cf = self.color_decoder(decoder_input)     # [B, num_points, 48]
        gau_pf = self.occ_decoder(decoder_input)       # [B, num_points, 1]
        
        # 重塑为 [B, C, H, W]
        H, W = self.grid_size
        gau_tf = gau_tf.transpose(1, 2).view(B, 7, H, W)
        gau_cf = gau_cf.transpose(1, 2).view(B, 48, H, W)
        gau_pf = gau_pf.transpose(1, 2).view(B, 1, H, W)
        
        return gau_pf, gau_cf, gau_tf