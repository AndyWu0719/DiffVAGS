import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import clip
from transformers import AutoModel, AutoTokenizer

class MultimodalVAVAE(nn.Module):
    """
    多模态VA-VAE，结合DINOv2和CLIP文本编码器约束潜在空间
    """
    def __init__(self, 
                 latent_dim: int = 768,
                 visual_model: str = "dinov2",
                 language_model: str = "clip_text",
                 visual_weight: float = 0.1,
                 language_weight: float = 0.05,
                 cross_modal_weight: float = 0.02,
                 semantic_weight: float = 0.01,
                 **vae_kwargs):
        super().__init__()
        
        # 导入原有的VAE编码器
        from ..gaussian_vae.gaussian_encoder import GaussianEncoder
        self.vae_encoder = GaussianEncoder(latent_dim=latent_dim, **vae_kwargs)
        
        # 预训练模型约束器
        self.visual_prior = self._build_visual_prior(visual_model)
        self.language_prior = self._build_language_prior(language_model)
        
        # 多模态对齐模块
        self.multimodal_aligner = MultimodalAligner(
            visual_dim=1024 if visual_model == "dinov2" else 768,
            text_dim=512 if language_model == "clip_text" else 768,
            latent_dim=latent_dim
        )
        
        # 损失权重
        self.visual_weight = visual_weight
        self.language_weight = language_weight
        self.cross_modal_weight = cross_modal_weight
        self.semantic_weight = semantic_weight
        
        # 冻结预训练模型
        self._freeze_pretrained_models()
    
    def _build_visual_prior(self, model_name: str):
        """构建视觉先验模型"""
        if model_name == "dinov2":
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            return model
        elif model_name == "clip_visual":
            model, _ = clip.load("ViT-B/32", device="cuda")
            return model.encode_image
        else:
            raise ValueError(f"Unsupported visual model: {model_name}")
    
    def _build_language_prior(self, model_name: str):
        """构建语言先验模型"""
        if model_name == "clip_text":
            model, _ = clip.load("ViT-B/32", device="cuda")
            return model.encode_text
        elif model_name == "bert":
            model = AutoModel.from_pretrained('bert-base-uncased')
            return model
        else:
            raise ValueError(f"Unsupported language model: {model_name}")
    
    def _freeze_pretrained_models(self):
        """冻结预训练模型参数"""
        for param in self.visual_prior.parameters():
            param.requires_grad = False
        
        if hasattr(self.language_prior, 'parameters'):
            for param in self.language_prior.parameters():
                param.requires_grad = False
    
    def forward(self, multimodal_features: torch.Tensor, text_tokens: Optional[torch.Tensor] = None):
        """
        前向传播
        Args:
            multimodal_features: [B, C, H, W] 多模态融合特征
            text_tokens: [B, seq_len] 文本token（可选）
        """
        # 1. 标准VAE前向传播
        recon_x, data, mu, logvar, z = self.vae_encoder(multimodal_features)
        
        # 2. 获取预训练模型的特征表示
        alignment_loss = self._compute_alignment_loss(z, recon_x, text_tokens)
        
        return recon_x, data, mu, logvar, z, alignment_loss
    
    def _compute_alignment_loss(self, latent_z: torch.Tensor, 
                               recon_x: torch.Tensor, 
                               text_tokens: Optional[torch.Tensor] = None):
        """计算对齐损失"""
        with torch.no_grad():
            # 视觉先验特征
            # 需要将recon_x调整为DINOv2期望的输入格式
            if recon_x.shape[1] != 3:  # 如果不是RGB格式
                # 转换为RGB格式或使用适配层
                visual_input = self._adapt_visual_input(recon_x)
            else:
                visual_input = recon_x
            
            visual_prior_features = self.visual_prior(visual_input)  # [B, 1024]
            
            # 语言先验特征
            language_prior_features = None
            if text_tokens is not None:
                language_prior_features = self.language_prior(text_tokens)  # [B, 512]
        
        # 多模态对齐
        alignment_loss = self.multimodal_aligner(
            latent_z, visual_prior_features, language_prior_features
        )
        
        return alignment_loss
    
    def _adapt_visual_input(self, x: torch.Tensor) -> torch.Tensor:
        """将非RGB输入适配为DINOv2可接受的格式"""
        B, C, H, W = x.shape
        if C != 3:
            # 简单的通道适配：使用1x1卷积或平均
            if not hasattr(self, 'visual_adapter'):
                self.visual_adapter = nn.Conv2d(C, 3, 1).to(x.device)
            x = self.visual_adapter(x)
        
        # 确保尺寸符合DINOv2要求
        if H != 224 or W != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        return x
    
    def compute_total_loss(self, recon_x, data, mu, logvar, z, alignment_loss, **kwargs):
        """计算总损失"""
        # 标准VAE损失
        vae_loss_dict = self.vae_encoder.loss_function(recon_x, data, mu, logvar, **kwargs)
        
        # 语义一致性损失
        semantic_loss = self._compute_semantic_consistency_loss(z, recon_x)
        
        # 总损失
        total_loss = (
            vae_loss_dict['VAEloss'] + 
            self.visual_weight * alignment_loss +
            self.semantic_weight * semantic_loss
        )
        
        return {
            'total_loss': total_loss,
            'vae_loss': vae_loss_dict['VAEloss'],
            'alignment_loss': alignment_loss,
            'semantic_loss': semantic_loss,
            **vae_loss_dict
        }
    
    def _compute_semantic_consistency_loss(self, latent_z, recon_x):
        """计算语义一致性损失"""
        with torch.no_grad():
            recon_visual_input = self._adapt_visual_input(recon_x)
            recon_semantic = self.visual_prior(recon_visual_input)
        
        # 将潜在向量投影到视觉空间
        latent_semantic = self.multimodal_aligner.visual_proj(latent_z)
        
        return F.mse_loss(latent_semantic, recon_semantic)


class MultimodalAligner(nn.Module):
    """多模态对齐模块"""
    def __init__(self, visual_dim: int, text_dim: int, latent_dim: int):
        super().__init__()
        
        # 特征投影层
        self.visual_proj = nn.Linear(latent_dim, visual_dim)
        self.text_proj = nn.Linear(latent_dim, text_dim) if text_dim > 0 else None
        
        # 交叉注意力机制（可选）
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_dim, 
            num_heads=8, 
            batch_first=True
        )
    
    def forward(self, latent_z, visual_features, text_features=None):
        """计算对齐损失"""
        losses = []
        
        # 视觉对齐损失 (VF Loss)
        visual_proj = self.visual_proj(latent_z)  # [B, visual_dim]
        visual_align_loss = F.mse_loss(visual_proj, visual_features.detach())
        losses.append(visual_align_loss)
        
        # 文本对齐损失 (LF Loss)
        if text_features is not None and self.text_proj is not None:
            text_proj = self.text_proj(latent_z)  # [B, text_dim]
            text_align_loss = F.mse_loss(text_proj, text_features.detach())
            losses.append(text_align_loss)
            
            # 跨模态一致性损失
            visual_proj_norm = F.normalize(visual_proj, dim=-1)
            text_proj_norm = F.normalize(text_proj, dim=-1)
            cross_modal_loss = 1 - F.cosine_similarity(
                visual_proj_norm, text_proj_norm, dim=-1
            ).mean()
            losses.append(cross_modal_loss)
        
        return sum(losses) / len(losses) if losses else torch.tensor(0.0)