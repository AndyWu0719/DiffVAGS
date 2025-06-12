import torch
import torch.utils.data 
from torch.nn import functional as F
import pytorch_lightning as pl
import time
from einops import reduce

from models.gaussian_vae.gaussian_model import GaussianModel
from input_encoder.text_encoder import TextEncoder
from input_encoder.image_encoder import ImageEncoder
from input_encoder.multimodal_encoder import MultiModalEncoder
from input_encoder.conv_pointlite import ConvPointnetLite
from models.gaussian_vae.gaussian_encoder import GaussianEncoder
from models.diffusion.diff_model import DiffusionModel
from models.diffusion.condition_net import ConditionNet
from dataloader.gaussian_loader import GaussianLoader
from .multimodal_va_vae import MultimodalVAVAE

class CombinedModel(pl.LightningModule):
    def __init__(self, specs):
        super().__init__()
        self.specs = specs
        self.task = specs['training_task']
        self.training_specs = specs['TrainingSpecs']
        self.multimodal_specs = specs['MultiModalEncoderSpecs']
        self.vae_specs = specs['VAESpecs']
        self.gaussian_specs = specs['GaussianModelSpecs']

        self._init_multimodal_encoder()

        if specs.get('VAVAESpecs', {}).get('enable', False):
            self._init_va_vae()
        else:
            self._init_standard_vae()
        
        if self.task in ('combined', 'modulation'):
            self.gaussian_model = GaussianModel(specs)
        
        if self.task in ('combined', 'diffusion'):
            self._init_diffusion_model()
        
        self._load_constant_data()

    def _init_multimodal_encoder(self):
        """初始化多模态编码器"""
        text_specs = self.multimodal_specs['text_encoder']
        image_specs = self.multimodal_specs['image_encoder']
        fusion_specs = self.multimodal_specs['fusion']
        
        text_encoder = TextEncoder(
            embed_dim=text_specs['embed_dim']
        )
        
        image_encoder = ImageEncoder(
            embed_dim=image_specs['embed_dim']
        )
        
        self.multimodal_encoder = MultiModalEncoder(
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            output_channels=fusion_specs['output_channels'],
            output_features=fusion_specs['output_features'],
            output_resolution=fusion_specs['output_resolution'],
            freeze_encoders=fusion_specs['freeze_encoders']
        )

    def _init_standard_vae(self):
        """初始化标准VAE"""
        encoder_specs = self.vae_specs['encoder']
        loss_specs = self.vae_specs['loss']
        
        self.vae_model = GaussianEncoder(
            in_channels=encoder_specs['in_channels'],
            latent_dim=encoder_specs['latent_dim'],
            hidden_dims=encoder_specs['hidden_dims'],
            beta=loss_specs['beta'],
            gamma=loss_specs['gamma'],
            kl_std=loss_specs['kl_std'],
            loss_type=loss_specs['loss_type'],
            max_capacity=loss_specs['max_capacity'],
            capacity_max_iteration=loss_specs['capacity_max_iteration']
        )
        self.use_va_vae = False

    def _init_va_vae(self):
        """初始化VA-VAE"""
        va_vae_specs = self.specs['VAVAESpecs']
        vae_encoder_params = self.vae_specs['encoder'].copy()
        vae_loss_params = self.vae_specs['loss'].copy()

        latent_dim = vae_encoder_params.pop('latent_dim')
        
        self.multimodal_va_vae = MultimodalVAVAE(
            # VA-VAE特有参数
            latent_dim=latent_dim,
            visual_model=va_vae_specs['visual_model']['type'],
            language_model=va_vae_specs['language_model']['type'],
            visual_weight=va_vae_specs['alignment']['visual_weight'],
            language_weight=va_vae_specs['alignment']['language_weight'],
            cross_modal_weight=va_vae_specs['alignment']['cross_modal_weight'],
            semantic_weight=va_vae_specs['alignment']['semantic_consistency_weight'],
            # VAE编码器参数（已移除latent_dim）
            **vae_encoder_params,
            # VAE损失参数
            **vae_loss_params
        )
        self.use_va_vae = True

    def _init_diffusion_model(self):
        """初始化扩散模型"""
        diff_model_specs = self.specs['DiffusionSpecs']['model']
        diff_training_specs = self.specs['DiffusionSpecs']['training']
        
        condition_net = ConditionNet(
            num_classes=diff_model_specs['dim_in_out'],
            in_channels=diff_model_specs['dim'],
            depth=diff_model_specs['depth'],
            cond_dim=diff_model_specs['condition_dim'] if diff_model_specs['use_condition'] else 0
        )
        
        self.diffusion_model = DiffusionModel(
            model=condition_net,
            timesteps=diff_training_specs['timesteps'],
            objective=diff_training_specs['objective'],
            loss_type=diff_training_specs['loss_type']
        )
    
    def _load_constant_data(self):
        """加载常量数据"""
        data_path = self.specs['Gaussian_Data_path']
        gaussian_loader = GaussianLoader(data_path)
        self.constant_gt = gaussian_loader.__getitem__(0)

    def configure_optimizers(self):
        """配置优化器"""
        lr_specs = self.training_specs['learning_rates']
        
        if self.task == 'combined':
            params_list = [
                {'params': list(self.gaussian_model.parameters()), 'lr': lr_specs['gaussian_lr']},
                {'params': list(self.diffusion_model.parameters()), 'lr': lr_specs['diffusion_lr']}
            ]
            if hasattr(self, 'multimodal_va_vae'):
                params_list.append({'params': list(self.multimodal_va_vae.parameters()), 'lr': lr_specs['vae_lr']})
            elif hasattr(self, 'vae_model'):
                params_list.append({'params': list(self.vae_model.parameters()), 'lr': lr_specs['vae_lr']})
                
        elif self.task == 'modulation':
            params_list = [{'params': self.parameters(), 'lr': lr_specs['vae_lr']}]
            
        elif self.task == 'diffusion':
            params_list = [{'params': self.parameters(), 'lr': lr_specs['diffusion_lr']}]
        
        optimizer = torch.optim.Adam(params_list)
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        # 内存监控
        if batch_idx % 50 == 0 and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            if allocated > 15.0:
                torch.cuda.empty_cache()
        
        x = batch
        if self.task == 'combined':
            return self.train_combined(x)
        elif self.task == 'modulation':
            return self.train_modulation(x)
        elif self.task == 'diffusion':
            return self.train_diffusion(x)
        
    def train_modulation(self, x):
        """VAE调制训练"""
        # 获取配置参数
        loss_weights = self.training_specs['loss_weights']
        query_specs = self.gaussian_specs['query_points']
        
        # 准备数据
        device = x.device
        occ_xyz = self.constant_gt['occ_xyz'].to(device)
        occ = self.constant_gt['occ'].to(device)
        gt = self.constant_gt['gt_gaussian'].to(device)
        gaussian_xyz = self.constant_gt['gaussian_xyz'].to(device)

        # 为occ_xyz添加batch维度
        if occ_xyz.dim() == 2:
            occ_xyz = occ_xyz.unsqueeze(0)  # [80000, 3] -> [1, 80000, 3]
        occ_xyz = occ_xyz.to(device)
        
        # 为occ添加batch维度
        if occ.dim() == 2:
            occ = occ.unsqueeze(0)  # [80000, 1] -> [1, 80000, 1]
        occ = occ.to(device)
        
        # 为gt添加batch维度
        if gt.dim() == 2:
            gt = gt.unsqueeze(0)  # [16000, 56] -> [1, 16000, 56]
        gt = gt.to(device)
        
        # 确保gaussian_xyz有正确维度
        if gaussian_xyz.dim() == 2:
            gaussian_xyz = gaussian_xyz.unsqueeze(0)  # [16000, 3] -> [1, 16000, 3]
        gaussian_xyz = gaussian_xyz.to(device)
        
        # 确保batch维度
        for tensor_name, tensor in [('occ_xyz', occ_xyz), ('occ', occ), ('gt', gt), ('gaussian_xyz', gaussian_xyz)]:
            if tensor.dim() == 2:
                locals()[tensor_name] = tensor.unsqueeze(0)
        
        # 限制查询点数量
        max_gaussian_pts = query_specs['max_gaussian_points']
        max_occ_pts = query_specs['max_occupancy_points']
        
        if gaussian_xyz.shape[1] > max_gaussian_pts:
            indices = torch.randperm(gaussian_xyz.shape[1])[:max_gaussian_pts]
            gaussian_xyz = gaussian_xyz[:, indices, :]
            gt = gt[:, indices, :]
        
        if occ_xyz.shape[1] > max_occ_pts:
            indices = torch.randperm(occ_xyz.shape[1])[:max_occ_pts]
            occ_xyz = occ_xyz[:, indices, :]
            occ = occ[:, indices, :]
        
        # VAE前向传播
        x = x.squeeze(1)
        if self.use_va_vae:
            # 使用VA-VAE
            text_tokens = getattr(x, 'text_tokens', None)
            recon_x, input_data, mu, logvar, z, alignment_loss = self.multimodal_va_vae(x, text_tokens)
            
            vae_loss_dict = self.multimodal_va_vae.compute_total_loss(
                recon_x, input_data, mu, logvar, z, alignment_loss,
                minibatch_weight=loss_weights['kld_weight']
            )
            vae_loss = vae_loss_dict['total_loss']
        else:
            # 使用标准VAE
            out = self.vae_model(x)
            recon_x, input_data, mu, logvar, z = out
            
            vae_loss_dict = self.vae_model.loss_function(
                recon_x, input_data, mu, logvar,
                minibatch_weight=loss_weights['kld_weight']
            )
            vae_loss = vae_loss_dict['VAEloss']
        
        # 高斯预测
        try:
            pred_color, pred_gs = self.gaussian_model.forward_with_plane_features(
                recon_x, gaussian_xyz
            )
            pred_occ = self.gaussian_model.forward_with_plane_features_pf(
                recon_x, occ_xyz
            )
        except Exception as e:
            print(f"❌ 高斯预测错误: {e}")
            return None
        
        # 计算高斯损失
        gaussian_params_specs = self.gaussian_specs['gaussian_params']
        
        # 颜色损失（前48维）
        color_loss = torch.nn.functional.l1_loss(
            pred_color[:, :, :gaussian_params_specs['color_dim']], 
            gt[:, :, :gaussian_params_specs['color_dim']]
        )
        
        # 几何损失
        scale_start = gaussian_params_specs['color_dim'] + gaussian_params_specs['opacity_dim']
        scale_end = scale_start + gaussian_params_specs['scale_dim']
        rotation_start = scale_end
        rotation_end = rotation_start + gaussian_params_specs['rotation_dim']
        
        scale_loss = torch.nn.functional.l1_loss(
            pred_gs[:, :, :gaussian_params_specs['scale_dim']], 
            gt[:, :, scale_start:scale_end]
        )
        
        rotation_loss = torch.nn.functional.l1_loss(
            pred_gs[:, :, gaussian_params_specs['scale_dim']:gaussian_params_specs['scale_dim']+gaussian_params_specs['rotation_dim']], 
            gt[:, :, rotation_start:rotation_end]
        )
        
        # 占用损失
        occ_loss = torch.nn.functional.l1_loss(pred_occ.squeeze(-1), occ.squeeze(-1))
        
        # 总损失
        total_loss = (
            loss_weights['vae_weight'] * vae_loss +
            loss_weights['gaussian_weight'] * (color_loss + scale_loss + rotation_loss + occ_loss)
        )
        
        # 记录损失
        loss_dict = {
            'train/total_loss': total_loss,
            'train/vae_loss': vae_loss,
            'train/color_loss': color_loss,
            'train/scale_loss': scale_loss,
            'train/rotation_loss': rotation_loss,
            'train/occ_loss': occ_loss
        }
        
        for key, value in loss_dict.items():
            self.log(key, value, prog_bar=True)
        
        return total_loss
    
    def train_diffusion(self, x):
        """扩散训练"""
        latent = x['latent']
        
        diff_specs = self.specs['DiffusionSpecs']['model']
        if diff_specs['use_condition']:
            cond = x.get('gaussians', None)
        else:
            cond = None
        
        diff_loss, diff_100_loss, diff_1000_loss, _, _ = self.diffusion_model.diffusion_model_from_latent(
            latent, cond=cond
        )
        
        loss_dict = {
            'train/diffusion_loss': diff_loss,
            'train/diff_100_loss': diff_100_loss,
            'train/diff_1000_loss': diff_1000_loss
        }
        
        self.log_dict(loss_dict, prog_bar=True)
        return diff_loss
    
    def train_combined(self, x):
        """联合训练"""
        loss_mod = self.train_modulation(x)
        loss_diff = self.train_diffusion(x)
        total_loss = loss_mod + loss_diff
        
        self.log('train/combined_loss', total_loss, prog_bar=True)
        return total_loss
    
    '''    # GaussianModel 内部包含 MultiModalEncoder + GaussianEncoder + TriplaneDecoder
        if self.task in ('combined', 'modulation'):
            self.gaussian_model = GaussianModel(specs=specs)
            # feature_dim 取自 GaussianModelSpecs 的 latent_dim
            feature_dim = specs["GaussianModelSpecs"]["latent_dim"]
            latent_std = specs.get("latent_std", 0.25)
            self.vae_model = GaussianEncoder(
                in_channels=128,
                latent_dim=feature_dim*3,
                hidden_dims=specs.get("hidden_dims", [16, 24, 40]),
                kl_std=latent_std,
                beta=specs.get("beta", 4),
                gamma=specs.get("gamma", 10.),
                max_capacity=specs.get("max_capacity", 25),
                capacity_max_iteration=specs.get("capacity_max_iteration", 1e5),
                loss_type=specs.get("loss_type", 'B'),
                **specs
            )
        text_encoder_instance = TextEncoder()
        image_encoder_instance = ImageEncoder()
        self.multimodal_encoder = MultiModalEncoder(
            text_encoder=text_encoder_instance,
            image_encoder=image_encoder_instance,
            output_channels=128,
            output_features=512,
            freeze_encoders=True
        )

        if self.task in ('combined', 'diffusion'):
            self.diffusion_model = DiffusionModel(
                model=ConditionNet(**specs["diffusion_model_specs"]), **specs["diffusion_specs"]
            )

        gaussian_loader = GaussianLoader(specs["Data_path"]).__getitem__(0) # 要是当前gaussian的那个文件夹的idx
        self.constant_gt = gaussian_loader
    
    def training_step(self, batch, batch_idx):
        x = batch
        if self.task == 'combined':
            return self.train_combined(x)
        elif self.task == 'modulation':
            return self.train_modulation(x)
        elif self.task == 'diffusion':
            return self.train_diffusion(x)
        
    def configure_optimizers(self):
        if self.task == 'combined':
            params_list = [
                {'params': list(self.gaussian_model.parameters()), 'lr': self.specs['sdf_lr']},
                {'params': self.diffusion_model.parameters(), 'lr': self.specs['diff_lr']}
            ]
        elif self.task == 'modulation':
            params_list = [{'params': self.parameters(), 'lr': self.specs['sdf_lr']}]
        elif self.task == 'diffusion':
            params_list = [{'params': self.parameters(), 'lr': self.specs['diff_lr']}]
        optimizer = torch.optim.Adam(params_list)
        return {"optimizer": optimizer}

    # ---------- training step for modulation (combined reconstruction+VAE loss) ----------
    def train_modulation(self, x):
        # 假设 x 包含以下字段：
        #   'occ_xyz': 点云坐标用于占用预测（GT）
        #   'occ': 点云占用/密度GT
        #   'gt_gaussian': GT 的高斯系数，包含颜色和几何变换部分
        #   'gaussians': 多模态输入，供融合编码使用
        #   'gaussian_xyz': 坐标信息，用于 triplane 解码时与潜向量拼接

        occ_xyz = self.constant_gt['occ_xyz']  # (B, 3, 64, 64)
        occ = self.constant_gt['occ']  # (B, 1, 64, 64)
        gt = self.constant_gt['gt_gaussian']   # 假设 gt 的前 48 通道对应颜色，后续通道对应几何变换
        gs = self.constant_gt['gaussians']
        gaussian_xyz = self.constant_gt['gaussian_xyz']    

        if gaussian_xyz.dim() == 2:
            gaussian_xyz = gaussian_xyz.unsqueeze(0)  # [1, N, 3]

        device = x.device

        # 为occ_xyz添加batch维度
        if occ_xyz.dim() == 2:
            occ_xyz = occ_xyz.unsqueeze(0)  # [80000, 3] -> [1, 80000, 3]
        occ_xyz = occ_xyz.to(device)
        
        # 为occ添加batch维度
        if occ.dim() == 2:
            occ = occ.unsqueeze(0)  # [80000, 1] -> [1, 80000, 1]
        occ = occ.to(device)
        
        # 为gt添加batch维度
        if gt.dim() == 2:
            gt = gt.unsqueeze(0)  # [16000, 56] -> [1, 16000, 56]
        gt = gt.to(device)
        
        # 为gs添加batch维度（如果需要）
        if gs.dim() == 2:
            gs = gs.unsqueeze(0)  # [100000, 59] -> [1, 100000, 59]
        gs = gs.to(device)
        
        # 确保gaussian_xyz有正确维度
        if gaussian_xyz.dim() == 2:
            gaussian_xyz = gaussian_xyz.unsqueeze(0)  # [16000, 3] -> [1, 16000, 3]
        gaussian_xyz = gaussian_xyz.to(device)

        x = x.squeeze(1)
        out = self.vae_model(x)
        print("VAE encoder completed")
        reconstructed_plane_feature, latent = out[0], out[-1]

        # 🔧 限制查询点数量避免GPU内存不足
        max_gaussian_points = 1000
        max_occ_points = 2000
        
        # 限制gaussian查询点数量
        if gaussian_xyz.shape[1] > max_gaussian_points:
            sample_indices = torch.randperm(gaussian_xyz.shape[1])[:max_gaussian_points]
            gaussian_xyz_sample = gaussian_xyz[:, sample_indices, :]
            gt_sample = gt[:, sample_indices, :]
        else:
            gaussian_xyz_sample = gaussian_xyz
            gt_sample = gt
        
        # 限制占用查询点数量
        if occ_xyz.shape[1] > max_occ_points:
            sample_indices = torch.randperm(occ_xyz.shape[1])[:max_occ_points]
            occ_xyz_sample = occ_xyz[:, sample_indices, :]
            occ_sample = occ[:, sample_indices, :]
        else:
            occ_xyz_sample = occ_xyz
            occ_sample = occ

        # 通过多模态编码器融合各模态特征，输出形状 [B, m_out_dim]
        # 投影到 latent 空间

        # 预测
        try:
            pred_color, pred_gs = self.gaussian_model.forward_with_plane_features(
                reconstructed_plane_feature, gaussian_xyz_sample
            )
            pred_occ = self.gaussian_model.forward_with_plane_features_pf(
                reconstructed_plane_feature, occ_xyz_sample
            )
            
        except Exception as e:
            print(f"❌ 预测错误: {e}")
            import traceback
            traceback.print_exc()
            return None

        # 🔧 修复：VAE损失计算参数名
        try:
            vae_loss_dict = self.vae_model.loss_function(
                *out, 
                minibatch_weight=self.specs.get("kld_weight", 1.0)  # 🔧 修复：使用minibatch_weight而不是M_N
            )
            
            # 处理损失字典或直接返回的损失值
            if isinstance(vae_loss_dict, dict):
                vae_loss = vae_loss_dict['VAEloss']
                print("VAE损失详情:", {k: v.item() if torch.is_tensor(v) else v for k, v in vae_loss_dict.items()})
            else:
                vae_loss = vae_loss_dict
                print("VAE损失:", vae_loss.item())
                
        except Exception as e:
            print(f"❌ VAE损失计算错误: {e}")
            import traceback
            traceback.print_exc()
            
            # 🔧 备用方案：手动计算VAE损失
            try:
                if len(out) >= 5:
                    recon, input, mu, log_var, z = out[:5]
                    # 重构损失
                    recon_loss = F.mse_loss(recon, input, reduction='sum')
                    # KL散度损失
                    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)
                    # 总损失
                    vae_loss = recon_loss + self.specs.get("kld_weight", 1.0) * kld_loss
                    print(f"手动计算VAE损失 - 重构: {recon_loss.item():.6f}, KLD: {kld_loss.item():.6f}, 总计: {vae_loss.item():.6f}")
                else:
                    print("❌ VAE输出格式错误，无法计算损失")
                    return None
            except Exception as e2:
                print(f"❌ 手动VAE损失计算也失败: {e2}")
                return None

        # 🔧 确保维度匹配进行损失计算
        try:
            # 调整维度以匹配
            min_points = min(pred_color.shape[1], gt_sample.shape[1])
            pred_color = pred_color[:, :min_points, :]
            pred_gs = pred_gs[:, :min_points, :]
            gt_sample = gt_sample[:, :min_points, :]
            
            min_occ_points = min(pred_occ.shape[1], occ_sample.shape[1])
            pred_occ = pred_occ[:, :min_occ_points, :]
            occ_sample = occ_sample[:, :min_occ_points, :]

            # 颜色损失 (前48维)
            color_loss = F.l1_loss(pred_color[:, :, 0:48], gt_sample[:, :, 0:48], reduction='none')
            color_loss = reduce(color_loss, 'b ... -> b (...)', 'mean').mean()

            # 几何损失
            scale_loss = F.l1_loss(pred_gs[:, :, 0:3], gt_sample[:, :, 49:52], reduction='none')
            scale_loss = reduce(scale_loss, 'b ... -> b (...)', 'mean').mean()
            
            rotation_loss = F.l1_loss(pred_gs[:, :, 3:7], gt_sample[:, :, 52:56], reduction='none')
            rotation_loss = reduce(rotation_loss, 'b ... -> b (...)', 'mean').mean()

            # 占用损失
            occ_loss = F.l1_loss(pred_occ.squeeze(-1), occ_sample.squeeze(-1), reduction='none')
            occ_loss = reduce(occ_loss, 'b ... -> b (...)', 'mean').mean()

            # 总损失
            loss = color_loss + vae_loss + occ_loss + scale_loss + rotation_loss

            loss_dict = {
                "loss": loss,
                "color": color_loss,
                "vae": vae_loss,
                "occ": occ_loss,
                "scale": scale_loss,
                "rotation": rotation_loss
            }
            
            # 打印损失详情
            print("损失计算结果:")
            for k, v in loss_dict.items():
                if torch.is_tensor(v):
                    print(f"{k}: {v.item():.6f}")
                else:
                    print(f"{k}: {v}")
            
            self.log_dict(loss_dict, prog_bar=True, enable_graph=False)
            return loss
            
        except Exception as e:
            print(f"❌ 损失计算错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ---------- training step for diffusion ----------
    def train_diffusion(self, x):
        self.train()
        # 假设 x 包含：
        #  'gaussians': 条件信息（或False，用于无条件采样）
        #  'latent': 潜向量（满足扩散模型输入要求）
        latent = x['latent']
        if self.specs['diffusion_model_specs']['cond']:
            gs = x['gaussians'] # (B, 1024, 3) or False if unconditional 
            cond = gs
        else:
            cond = None 

        diff_loss, diff_100_loss, diff_1000_loss, _, __ = self.diffusion_model.diffusion_model_from_latent(latent, cond=cond)
        loss_dict = {
            "total": diff_loss,
            "diff100": diff_100_loss,
            "diff1000": diff_1000_loss
        }
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)
        return diff_loss

    def train_combined(self, x):
        # 可根据任务需求将 modulation 与 diffusion 的 loss 进行加权组合
        loss_mod = self.train_modulation(x)
        loss_diff = self.train_diffusion(x)
        total_loss = loss_mod + loss_diff
        return total_loss
    
    def training_step(self, batch, batch_idx):
        # 🔧 添加内存监控
        if batch_idx % 50 == 0:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"Step {batch_idx}: GPU内存 - 分配: {allocated:.2f}GB, 保留: {reserved:.2f}GB")
                
                # 🔧 如果内存使用过高，强制清理
                if allocated > 15.0:  # 如果超过15GB
                    torch.cuda.empty_cache()
                    print("执行GPU内存清理")
        
        # 原有训练逻辑
        x = batch
        if self.task == 'combined':
            return self.train_combined(x)
        elif self.task == 'modulation':
            return self.train_modulation(x)
        elif self.task == 'diffusion':
            return self.train_diffusion(x)
    
    def on_train_epoch_end(self):
        """在每个epoch结束时清理内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"Epoch结束后GPU内存: {allocated:.2f}GB")'''