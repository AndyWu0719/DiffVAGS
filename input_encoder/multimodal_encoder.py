import torch
import torch.nn as nn


class MultiModalEncoder(nn.Module):
    """
    MultiModalEncoder combines text, image, and point cloud encoders.
    Args:
        text_encoder (nn.Module): Text encoder module.
        image_encoder (nn.Module): Image encoder module.
        pointcloud_encoder (nn.Module): Point cloud encoder module.
        output_channels (int): Number of output channels after fusion.
        output_features (int): Number of output features after fusion.
    Input:
        data (dict): A dictionary containing the input data.
            - "text": Text features.
            - "image": Image features.
            - "pointcloud": Point cloud features.
    Output:
        out (torch.Tensor): Fused features of shape [B, H, W, C].
    """
    def __init__(self, 
                 text_encoder: nn.Module = None, 
                 image_encoder: nn.Module = None, 
                 output_channels: int = 128,    # x, y, z
                 output_features: int = 512,       # objects shape, same as DiffusionModel
                 output_resolution: int = 64,  # output resolution, default 64x64
                 freeze_encoders: bool = True  # 🔧 添加控制参数
                 ):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.output_channels = output_channels
        self.output_features = output_features
        self.output_resolution = output_resolution

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_features, num_heads=8, batch_first=True
        )

        # calculate input modalities' dimensions
        in_dim = 0
        if self.text_encoder is not None:
            in_dim += self.text_encoder.out_features
        if self.image_encoder is not None:
            in_dim += self.image_encoder.out_features

        self.fusion_linear = nn.Linear(in_dim, output_features)
        self.fusion_proj = nn.Linear(output_features, 
                                   output_channels * output_resolution * output_resolution)

        if freeze_encoders:
            if self.text_encoder is not None:
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
                print("🔒 Text encoder frozen")
            
            if self.image_encoder is not None:
                for param in self.image_encoder.parameters():
                    param.requires_grad = False
                print("🔒 Image encoder frozen")
            
            # 🔧 融合层保持可训练（用于适配不同下游任务）
            print("🔓 Fusion layers remain trainable")
        else:
            print("🔓 All parameters remain trainable")

    def forward(self, data: dict) -> torch.Tensor:
        features_list = []
        feat_text = None
        feat_image = None

        if "text" in data and self.text_encoder is not None:
            feat_text = self.text_encoder(data["text"])   # [batch_size, text_feat_dim], [B, 512]
            features_list.append(feat_text)
        if "image" in data and self.image_encoder is not None:
            feat_image = self.image_encoder(data["image"])  # [batch_size, image_feat_dim], [B, 512]
            features_list.append(feat_image)
        
        if len(features_list) == 0:
            raise ValueError("No input data provided for encoding.")
        
        combined_features = torch.cat(features_list, dim=-1)  # [B, in_dim]
        fused_features = self.fusion_linear(combined_features)  # [B, output_features]
        
        # 交叉注意力增强（如果有多模态）
        if len(features_list) >= 2:
            query = fused_features.unsqueeze(1)  # [B, 1, output_features]
            key_value = fused_features.unsqueeze(1)
            attn_output, _ = self.cross_attn(query=query, key=key_value, value=key_value)
            final_features = attn_output.squeeze(1)  # [B, output_features]
        else:
            final_features = fused_features

        proj = self.fusion_proj(final_features)  # [B, output_features] -> [B, output_channels * 64 * 64]
        out = proj.view(-1, self.output_channels, self.output_resolution, self.output_resolution)  # [B, output_channels, 64, 64]

        return out  # [B, output_channels, 64, 64]