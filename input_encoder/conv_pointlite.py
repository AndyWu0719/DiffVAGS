import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max
from torch.nn import init

#############################
# 辅助卷积与上采样函数
#############################
def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

#############################
# DownConv模块
#############################
class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool

#############################
# UpConv模块
#############################
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.upconv = upconv2x2(in_channels, out_channels, mode=up_mode)
        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(2*out_channels, out_channels)
        else:
            self.conv1 = conv3x3(out_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), input_dim=1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

#############################
# UNet模块 (完整版本)
#############################
class UNet(nn.Module):
    """
    UNet, 基于 https://arxiv.org/abs/1505.04597 的结构，支持两种上采样方式和合并模式。
    """
    def __init__(self, num_classes, in_channels=3, depth=5, 
                 start_filts=64, up_mode='transpose', same_channels=False,
                 merge_mode='concat', **kwargs):
        super(UNet, self).__init__()
        if up_mode not in ('transpose', 'upsample'):
            raise ValueError(f"Invalid up_mode '{up_mode}'")
        if merge_mode not in ('concat', 'add'):
            raise ValueError(f"Invalid merge_mode '{merge_mode}'")
        if up_mode == 'upsample' and merge_mode == 'add':
            raise ValueError("up_mode 'upsample' is incompatible with merge_mode 'add'")
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.down_convs = []
        self.up_convs = []
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = start_filts*(2**i) if not same_channels else self.in_channels
            pooling = True if i < depth-1 else False
            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)
        for i in range(depth-1):
            ins = outs
            outs = ins // 2 if not same_channels else ins 
            up_conv = UpConv(ins, outs, up_mode=up_mode, merge_mode=merge_mode)
            self.up_convs.append(up_conv)
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.conv_final = conv1x1(outs, self.num_classes)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def reset_params(self):
        for m in self.modules():
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []
        for module in self.down_convs:
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)
        x = self.conv_final(x)
        return x

#############################
# 轻量化ResNet FC Block
#############################
class ResnetBlockFC_Lite(nn.Module):
    def __init__(self, size_in, size_out=None, size_h=None):
        super(ResnetBlockFC_Lite, self).__init__()
        if size_out is None:
            size_out = size_in
        if size_h is None:
            size_h = min(size_in, size_out)
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()
        self.shortcut = nn.Linear(size_in, size_out, bias=False) if size_in != size_out else None
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.actvn(self.fc_0(x))
        dx = self.actvn(self.fc_1(net))
        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        return x_s + dx

#############################
# 轻量化 ConvPointnet 模块
#############################
class ConvPointnetLite(nn.Module):
    """
    轻量化的 ConvPointnet 版本，基于 PointNet Lite 思想，
    同时支持可选的局部聚合操作和 UNet 后处理平面特征。
    
    Args:
        feature_dim (int): 输出特征维度
        input_dim (int): 输入点云维度 (默认3)
        hidden_dim (int): 隐藏层维度
        num_blocks (int): ResNet 块数量
        plane_resolution (int): 平面特征图分辨率
        plane_types (list of str): 使用的平面类型，如 ['xz', 'xy', 'yz']
        padding (float): 坐标归一化时的 padding
        scatter_type (str): 特征聚合方式，'mean' 或 'max'
        use_local_aggregation (bool): 是否使用局部聚合操作(默认True)
        use_unet (bool): 是否采用 UNet 后处理平面特征(默认True)
    """
    def __init__(self, 
                 feature_dim=256, 
                 input_dim=3, 
                 hidden_dim=64, 
                 num_blocks=3, 
                 plane_resolution=64, 
                 plane_types=['xz', 'xy', 'yz'],
                 padding=0.1, 
                 scatter_type='mean',
                 use_local_aggregation=True,
                 use_unet=True
                 ):
        super(ConvPointnetLite, self).__init__()
        self.feature_dim = feature_dim
        self.use_local_aggregation = use_local_aggregation
        self.use_unet = use_unet

        # 点坐标转换为高维表示
        self.fc_pos = nn.Linear(input_dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC_Lite(2 * hidden_dim, hidden_dim) for _ in range(num_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, feature_dim)
        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        self.reso_plane = plane_resolution
        self.plane_types = plane_types
        self.padding = padding

        if scatter_type == 'max':
            self.scatter = scatter_max
        else:
            self.scatter = scatter_mean

        # UNet后处理
        if self.use_unet:
            self.unet = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
                nn.ReLU()
            )
        else:
            self.unet = None

    def normalize_coordinate(self, p, plane='xz'):
        if p.input_dim() == 2:
            p = p.unsqueeze(0)
        if plane == 'xz':
            xy = p[:, :, [0, 2]]
        elif plane == 'xy':
            xy = p[:, :, [0, 1]]
        else:
            xy = p[:, :, [1, 2]]
        xy_new = (xy + 0.5) / (1 + self.padding + 1e-6)
        return torch.clamp(xy_new, 0, 1 - 1e-6)

    def coordinate2index(self, x, reso):
        x_int = (x * reso).long()
        index = x_int[:, :, 0] + reso * x_int[:, :, 1]
        return index[:, None, :]

    def generate_plane_features(self, p, c, plane='xz'):
        xy = self.normalize_coordinate(p.clone(), plane=plane)
        index = self.coordinate2index(xy, self.reso_plane)
        fea_plane = c.new_zeros(p.size(0), self.feature_dim, self.reso_plane**2)
        c_perm = c.permute(0, 2, 1)
        fea_plane = self.scatter(c_perm, index, input_dim_size=self.reso_plane**2)
        fea_plane = fea_plane.reshape(p.size(0), self.feature_dim, self.reso_plane, self.reso_plane)
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)
        return fea_plane

    def pool_local(self, xy, index, c):
        if self.use_local_aggregation:
            bs, fea_input_dim = c.size(0), c.size(2)
            c_out = 0
            for key in xy.keys():
                fea = self.scatter(c.permute(0, 2, 1), index[key], input_dim_size=self.reso_plane**2)
                if self.scatter == scatter_max:
                    fea = fea[0]
                fea = fea.gather(input_dim=2, index=index[key].expand(-1, fea_input_dim, -1))
                c_out += fea
            return c_out.permute(0, 2, 1)
        else:
            return torch.mean(c, input_dim=1, keepinput_dim=True).expand(-1, c.size(1), -1)

    def sample_plane_feature(self, query, plane_feature, plane):
        B, C_plane, H, W = plane_feature.shape
        B, N, _ = query.shape

        device = plane_feature.device
        query = query.to(device)
        
        # 根据平面类型选择坐标
        if plane == 'xy':
            coords_2d = query[:, :, [0, 1]]  # x, y
        elif plane == 'xz':
            coords_2d = query[:, :, [0, 2]]  # x, z  
        elif plane == 'yz':
            coords_2d = query[:, :, [1, 2]]  # y, z
        else:
            raise ValueError(f"Unsupported plane type: {plane}")
        
        coords_2d = coords_2d.to(device)
        coords_2d = coords_2d * 2.0 - 1.0  # 归一化到[-1, 1]
        coords_2d = coords_2d.unsqueeze(1)  # [B, 1, N, 2]
        
        sampled = F.grid_sample(
            plane_feature, 
            coords_2d, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=False
        )  # [B, C_plane, 1, N]
        
        sampled = sampled.squeeze(2)  # [B, C_plane, N]
        return sampled

    def get_plane_features(self, p):
        net = self.fc_pos(p)
        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            coord_all = self.normalize_coordinate(p, plane='xz')
            index_all = self.coordinate2index(coord_all, self.reso_plane)
            pooled = self.pool_local({'all': coord_all}, {'all': index_all}, net)
            net = torch.cat([net, pooled], input_dim=2)
            net = block(net)
        c = self.fc_c(net)
        fea = {}
        for plane in self.plane_types:
            fea[plane] = self.generate_plane_features(p, c, plane=plane)
        return tuple(fea[p] for p in self.plane_types)

    def forward(self, p, query):
        """
        🔧 原始DiffGS的forward方法: 每个平面完整特征, 相加融合
        """
        plane_feats = self.get_plane_features(p)
        plane_feat_sum = 0
        
        for i, plane in enumerate(self.plane_types):
            plane_feat_sum += self.sample_plane_feature(query, plane_feats[i], plane)
        
        return plane_feat_sum.transpose(2, 1)

    def forward_with_plane_features(self, plane_features, query):
        """
        🔧 修复: 按照原始DiffGS设计, 每个平面使用完整特征, 相加融合
        """
        B, C, H, W = plane_features.shape
        B, N, _ = query.shape

        device = plane_features.device
        query = query.to(device)
        
        # 🔧 严格验证维度
        assert C == self.feature_dim, f"输入特征维度 {C} 必须等于 feature_dim {self.feature_dim}。请检查上游模块的输出配置。"
        
        # 🔧 每个平面使用完整特征，相加融合
        plane_feat_sum = 0
        
        for plane in self.plane_types:
            sampled = self.sample_plane_feature(query, plane_features, plane)
            plane_feat_sum += sampled
        
        # 转置为 [B, N, feature_dim]
        combined_features = plane_feat_sum.transpose(1, 2)
        
        return combined_features

    def forward_with_pc_features(self, c, p, query):
        """
        🔧 使用点云特征的方法，遵循相加融合原则
        """
        fea = {}
        fea['xz'] = self.generate_plane_features(p, c, plane='xz')
        fea['xy'] = self.generate_plane_features(p, c, plane='xy')
        fea['yz'] = self.generate_plane_features(p, c, plane='yz')
        
        plane_feat_sum = 0
        for plane in self.plane_types:
            plane_feat_sum += self.sample_plane_feature(query, fea[plane], plane)
        
        return plane_feat_sum.transpose(2, 1)

    def get_point_cloud_features(self, p):
        net = self.fc_pos(p)
        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            coord_all = self.normalize_coordinate(p, plane='xz')
            index_all = self.coordinate2index(coord_all, self.reso_plane)
            pooled = self.pool_local({'all': coord_all}, {'all': index_all}, net)
            net = torch.cat([net, pooled], input_dim=2)
            net = block(net)
        c = self.fc_c(net)
        return c
    
    def forward_with_plane_features_pf(self, plane_features, query_xyz):
        """
        🔧 占用预测方法（使用修复后的相加融合设计）
        """
        features = self.forward_with_plane_features(plane_features, query_xyz)
    
        if not hasattr(self, 'occupancy_head'):
            self.occupancy_head = nn.Linear(self.feature_dim, 1).to(features.device)
        
        occupancy = self.occupancy_head(features)
        return occupancy