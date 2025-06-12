import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.gaussian_vae.gaussian_decoder import TriplaneDecoder
from models.gaussian_vae.gaussian_encoder import GaussianEncoder
from dataloader.multimodal_loader import MultiModalDataset, MultiModalDataLoader
from input_encoder.text_encoder import TextEncoder
from input_encoder.image_encoder import ImageEncoder
from input_encoder.multimodal_encoder import MultiModalEncoder
from models.gaussian_vae.gaussian_model import GaussianModel

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 构造 specs 字典
    specs = {
        "GaussianModelSpecs": {
            "hidden_dims": [16, 24, 40],
            "latent_dim": 512,              # 潜变量维度定为 512
            "fusion_outfeatures": 512,      # 多模态编码器输出特征维度
            "fusion_outchannels": 3,        # 多模态编码器输出通道数（3 表示平面分成3组，每组通道数 = 512）
            "skip_connection": True,
            "tanh_act": False,
            "pn_hidden_dim": 256            # ConvPointnetLite 的隐藏维度
        }
    }
    
    # 实例化 GaussianModel
    model = GaussianModel(specs).to(device)
    model.eval()

    
    
    batch_size = 1
    # 根据 specs，MultiModalEncoder 的融合输出形状应为 [B, fusion_outchannels, H, W]
    # 但是 GaussianEncoder 内部期望输入形状为 [B, fusion_outfeatures * fusion_outchannels, 64, 64]
    # 此处 fused_features 的通道数 = 512 * 3 = 1536
    fused_channels = specs["GaussianModelSpecs"]["fusion_outfeatures"] * specs["GaussianModelSpecs"]["fusion_outchannels"]
    fused_feature = torch.randn(batch_size, fused_channels, 64, 64).to(device)
    
    # 构造 dummy plane_features
    # 根据 ConvPointnetLite 的参数 dim=59, plane_resolution=64，plane_features 形状可以设为 [B, 59, 64, 64]
    plane_features = torch.randn(batch_size, 59, 64, 64).to(device)
    
    #######################
    # 1. 测试 forward 方法
    #######################
    print("Test forward()")
    gau_pf, gau_cf, gau_tf = model(fused_feature)
    print("Forward outputs:")
    print("gau_pf shape:", gau_pf.shape)  # 预期: [B, 1, 64, 64]
    print("gau_cf shape:", gau_cf.shape)  # 预期: [B, 48, 64, 64]
    print("gau_tf shape:", gau_tf.shape)  # 预期: [B, 7, 64, 64]
    
    assert gau_pf.shape == (batch_size, 1, 64, 64), f"Forward: gau_pf shape mismatch: {gau_pf.shape}"
    assert gau_cf.shape == (batch_size, 48, 64, 64), f"Forward: gau_cf shape mismatch: {gau_cf.shape}"
    assert gau_tf.shape == (batch_size, 7, 64, 64), f"Forward: gau_tf shape mismatch: {gau_tf.shape}"
    
    ###############################
    # 2. 测试 forward_with_plane_features 方法
    ###############################
    print("Test forward_with_plane_features()")
    gau_cf2, gau_tf2 = model.forward_with_plane_features(plane_features, fused_feature)
    print("forward_with_plane_features outputs:")
    print("gau_cf shape:", gau_cf2.shape)   # 预期: [B, 48, 64, 64]
    print("gau_tf shape:", gau_tf2.shape)   # 预期: [B, 7, 64, 64]
    
    assert gau_cf2.shape == (batch_size, 48, 64, 64), f"Forward_with_plane_features: gau_cf shape mismatch: {gau_cf2.shape}"
    assert gau_tf2.shape == (batch_size, 7, 64, 64), f"Forward_with_plane_features: gau_tf shape mismatch: {gau_tf2.shape}"
    
    #################################
    # 3. 测试 forward_with_plane_features_pf 方法
    #################################
    print("Test forward_with_plane_features_pf()")
    gau_pf2 = model.forward_with_plane_features_pf(plane_features, fused_feature)
    print("forward_with_plane_features_pf output:")
    print("gau_pf shape:", gau_pf2.shape)   # 预期: [B, 1, 64, 64]
    
    assert gau_pf2.shape == (batch_size, 1, 64, 64), f"Forward_with_plane_features_pf: gau_pf shape mismatch: {gau_pf2.shape}"
    
    print("All tests passed for GaussianModel.")

if __name__ == '__main__':
    main()