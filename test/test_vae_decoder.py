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

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 配置参数
    in_channels = 768          # MultiModalEncoder 输出通道数
    latent_dim = 512           # 潜变量维度
    hidden_dims = [16, 24, 40] # 隐藏层设置

    # 实例化 MultiModalEncoder
    text_encoder = TextEncoder()
    image_encoder = ImageEncoder()
    mm_encoder = MultiModalEncoder(
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        fusion_outchannels=in_channels,
        fusion_outfeatures=512
    ).to(device)
    
    caption_path = "/home/andywu/Documents/dongjun/LightDiffGS/process_data/step1/texts/captions.txt"
    image_path = "/home/andywu/Documents/dongjun/LightDiffGS/process_data/step1/images"
    dataset = MultiModalDataset(caption_file=caption_path, image_dir=image_path)
    
    # 使用 MultiModalEncoder 对数据进行预编码
    fused_loader = MultiModalDataLoader(dataset, mm_encoder, device=device)
    print("Number of fused features:", len(fused_loader))
    
    # 取第一条样本，形状应为 [B, 768, 64, 64]
    fused_feature = fused_loader[0]
    print("Fused feature shape:", fused_feature.shape)
    
    if fused_feature.ndim == 3:
        fused_feature = fused_feature.unsqueeze(0)
    
    fused_feature = fused_feature.to(device)
    
    # 实例化 GaussianEncoder
    gsencoder = GaussianEncoder(
        in_channels=fused_feature.shape[0],
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        kl_std=1.0,
        beta=4,
        gamma=10.0,
        max_capacity=25,
        capacity_max_iteration=1e5,
        loss_type='B'
    ).to(device)
    
    outputs = gsencoder(fused_feature)
    recon_x, data, mu, logvar, z = outputs
    
    print("recon_x shape:", recon_x.shape)  # 应为 [B, in_channels, 64, 64]
    print("data shape:", data.shape)          # [B, in_channels, 64, 64]
    print("mu shape:", mu.shape)              # [B, latent_dim]
    print("logvar shape:", logvar.shape)      # [B, latent_dim]
    print("z shape:", z.shape)                # [B, latent_dim]
    
    decoder = TriplaneDecoder(latent_dim=latent_dim, grid_size=(64, 64), hidden_dim=40).to(device)
    
    # 调用 decoder 得到三个输出
    gau_pf, gau_cf, gau_tf = decoder(z)
    
    # 打印各输出的形状
    print("gau_pf shape:", gau_pf.shape)  # 期望 [batch_size, 1, 64, 64]
    print("gau_cf shape:", gau_cf.shape)  # 期望 [batch_size, 48, 64, 64]
    print("gau_tf shape:", gau_tf.shape)  # 期望 [batch_size, 7, 64, 64]
    
    # 断言输出形状是否符合预期
    assert gau_pf.shape == (1, 1, 64, 64), f"Shape mismatch for gau_pf: {gau_pf.shape}"
    assert gau_cf.shape == (1, 48, 64, 64), f"Shape mismatch for gau_cf: {gau_cf.shape}"
    assert gau_tf.shape == (1, 7, 64, 64), f"Shape mismatch for gau_tf: {gau_tf.shape}"
    
    print("Test passed: TriplaneDecoder outputs are correct.")

if __name__ == '__main__':
    main()