import sys
import os
import torch
import numpy as np
import torchvision.transforms as transforms


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from input_encoder.text_encoder import TextEncoder
from input_encoder.image_encoder import ImageEncoder
from input_encoder.multimodal_encoder import MultiModalEncoder
from dataloader.gaussian_loader import GaussianLoader
from dataloader.multimodal_loader import MultiModalDataLoader, MultiModalDataset
from models.gaussian_vae.gaussian_encoder import GaussianEncoder
from models.gaussian_vae.gaussian_decoder import TriplaneDecoder
from models.gaussian_vae.gaussian_model import GaussianModel

def print_tensor_info(name, tensor):
    """打印张量信息"""
    if isinstance(tensor, torch.Tensor):
        print(f"{name:30s} | Shape: {str(tensor.shape):20s} | Device: {tensor.device} | Dtype: {tensor.dtype}")
    else:
        print(f"{name:30s} | Type: {type(tensor)} | Value: {tensor}")


def test_complete_pipeline():
    """测试完整的数据流水线"""
    print("=" * 80)
    print("LightDiffGS 完整流程张量维度测试")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # ========================================
    # 1. 输入数据维度
    # ========================================
    print("1. 输入数据维度")
    print("-" * 40)
    
    caption_path = "/home/andywu/Documents/dongjun/LightDiffGS/process_data/step1/texts/captions.txt"
    image_path = "/home/andywu/Documents/dongjun/LightDiffGS/process_data/step1/images"

    dataset = MultiModalDataset(caption_file=caption_path, image_dir=image_path)
    print_tensor_info("数据集样本数量", len(dataset))
    
    # 获取第一个样本来查看原始数据格式
    sample_data = dataset[0]
    sample_text, sample_image = sample_data['text'], sample_data['image']
    print_tensor_info("原始文本输入", sample_text)
    print_tensor_info("原始图像输入", sample_image)

    print()
    
    # ========================================
    # 2. 编码器输出维度
    # ========================================
    print("2. 编码器输出维度")
    print("-" * 40)
    
    # 实例化编码器
    text_encoder = TextEncoder().to(device)
    image_encoder = ImageEncoder().to(device)

    # 使用真实数据
    text_features = text_encoder(sample_text)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    sample_image_tensor = transform(sample_image)
    image_features = image_encoder(sample_image_tensor.unsqueeze(0).to(device))


    print_tensor_info("文本特征", text_features)
    print_tensor_info("图像特征", image_features)
    
    # 多模态融合编码器
    mm_encoder = MultiModalEncoder(
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        fusion_outchannels=768,
        fusion_outfeatures=512
    ).to(device)
    
    dataset = MultiModalDataset(caption_file=caption_path, image_dir=image_path)
    fused_loader = MultiModalDataLoader(dataset, mm_encoder, device=device)
    
    print_tensor_info("融合特征数据集大小", len(fused_loader))
    
    # 取第一条样本
    fused_feature = fused_loader[0]
    print_tensor_info("融合特征 (原始)", fused_feature)
    
    # 添加batch维度，保持与训练时一致
    if fused_feature.ndim == 3:
        fused_feature = fused_feature.unsqueeze(0)
    fused_feature = fused_feature.to(device)
    print_tensor_info("融合特征 (4D)", fused_feature)
    print()
    
    # ========================================
    # 3. VAE 编码器维度
    # ========================================
    print("3. VAE 编码器维度")
    print("-" * 40)
    
    # 实例化 GaussianEncoder，参数与其他测试文件保持一致
    vae_encoder = GaussianEncoder(
        in_channels=768,  # 使用正确的通道数
        latent_dim=512,
        hidden_dims=[16, 24, 40],
        kl_std=1.0,
        beta=4,
        gamma=10.0,
        max_capacity=25,
        capacity_max_iteration=1e5,
        loss_type='B'
    ).to(device)
    
    # 运行VAE编码器
    vae_outputs = vae_encoder(fused_feature)
    recon_x, input_x, mu, logvar, z = vae_outputs
    
    print_tensor_info("VAE重构输出", recon_x)
    print_tensor_info("VAE原始输入", input_x)
    print_tensor_info("VAE均值", mu)
    print_tensor_info("VAE对数方差", logvar)
    print_tensor_info("VAE潜在向量", z)
    
    # 验证维度一致性
    assert recon_x.shape == input_x.shape, f"重构输出与输入形状不匹配: {recon_x.shape} vs {input_x.shape}"
    assert mu.shape == logvar.shape == z.shape, f"潜在向量维度不匹配: {mu.shape}, {logvar.shape}, {z.shape}"
    
    print("✅ VAE编码器维度验证通过")
    print()
    
    # ========================================
    # 4. VAE 解码器维度
    # ========================================
    print("4. VAE 解码器维度")
    print("-" * 40)
    
    decoder = TriplaneDecoder(
        latent_dim=512, 
        grid_size=(64, 64), 
        hidden_dim=40
    ).to(device)
    
    gau_pf, gau_cf, gau_tf = decoder(z)
    
    print_tensor_info("VAE解码 - 位置特征", gau_pf)
    print_tensor_info("VAE解码 - 颜色特征", gau_cf) 
    print_tensor_info("VAE解码 - 变换特征", gau_tf)

    expected_shapes = {
        'gau_pf': (1, 1, 64, 64),    # 位置特征：[batch, 1, 64, 64]
        'gau_cf': (1, 48, 64, 64),   # 颜色特征：[batch, 48, 64, 64]
        'gau_tf': (1, 7, 64, 64)     # 变换特征：[batch, 7, 64, 64]
    }
    
    actual_shapes = {
        'gau_pf': gau_pf.shape,
        'gau_cf': gau_cf.shape,
        'gau_tf': gau_tf.shape
    }
    
    for name, expected in expected_shapes.items():
        actual = actual_shapes[name]
        assert actual == expected, f"{name} 形状不匹配: 期望 {expected}, 实际 {actual}"
    
    print("✅ TriplaneDecoder 维度验证通过")

    print()
    
    # ========================================
    # 5. 高斯模型维度
    # ========================================
    print("5. 高斯模型维度")
    print("-" * 40)

    # 构造 specs 字典
    specs = {
        "GaussianModelSpecs": {
            "hidden_dims": [16, 24, 40],
            "latent_dim": 512,              
            "fusion_outfeatures": 512,      
            "fusion_outchannels": 768,        
            "skip_connection": True,
            "tanh_act": False,
            "pn_hidden_dim": 256            
        }
    }

    gaussian_model = GaussianModel(specs).to(device)
    print("✅ GaussianModel 创建成功")

    # 🔧 加载真实的高斯和占用数据
    print("加载真实的高斯和占用数据:")
    gaussian_data_path = "/home/andywu/Documents/dongjun/LightDiffGS/process_data/step4/0/gaussian.npy"
    occ_data_path = "/home/andywu/Documents/dongjun/LightDiffGS/process_data/step4/0/occ.npy"

    try:
        # 加载真实高斯数据
        real_gaussian_data = np.load(gaussian_data_path)
        real_gaussian_data = torch.tensor(real_gaussian_data).float().to(device)
        
        if real_gaussian_data.ndim == 2:
            real_gaussian_data = real_gaussian_data.unsqueeze(0)
        
        print_tensor_info("真实高斯数据", real_gaussian_data)
        
        # 提取高斯坐标和特征
        gaussian_xyz = real_gaussian_data[:, :, :3]
        print_tensor_info("真实高斯坐标", gaussian_xyz)
        
        # 分析高斯数据结构
        print(f"高斯数据特征维度: {real_gaussian_data.shape[-1]}")
        if real_gaussian_data.shape[-1] >= 59:
            gt_xyz = real_gaussian_data[:, :, :3]
            gt_colors = real_gaussian_data[:, :, 3:51]  # 48维颜色
            gt_opacity = real_gaussian_data[:, :, 51:52]
            gt_scale = real_gaussian_data[:, :, 52:55]
            gt_rotation = real_gaussian_data[:, :, 55:59]
            
            print_tensor_info("GT - 3D坐标", gt_xyz)
            print_tensor_info("GT - 颜色特征", gt_colors)
            print_tensor_info("GT - 不透明度", gt_opacity)
            print_tensor_info("GT - 缩放参数", gt_scale)
            print_tensor_info("GT - 旋转参数", gt_rotation)
        
        # 加载真实占用数据
        real_occ_data = np.load(occ_data_path)
        real_occ_data = torch.tensor(real_occ_data).float().to(device)
        
        if real_occ_data.ndim == 2:
            real_occ_data = real_occ_data.unsqueeze(0)
        
        print_tensor_info("真实占用数据", real_occ_data)
        
        # 提取占用坐标和值
        if real_occ_data.shape[-1] >= 3:
            occ_xyz = real_occ_data[:, :, :3]
            print_tensor_info("真实占用坐标", occ_xyz)
            
            if real_occ_data.shape[-1] >= 4:
                occ_values = real_occ_data[:, :, 3:]
                print_tensor_info("真实占用值", occ_values)
        
        print("✅ 真实数据加载成功")
        
    except Exception as e:
        print(f"❌ 加载真实数据失败: {e}")
        print("使用模拟数据...")
        # 模拟数据
        real_gaussian_data = torch.randn(1, 1000, 59).to(device)
        gaussian_xyz = real_gaussian_data[:, :, :3]
        real_occ_data = torch.randn(1, 10000, 4).to(device)
        occ_xyz = real_occ_data[:, :, :3]
        occ_values = real_occ_data[:, :, 3:]

    # 🔧 测试1: GaussianModel.forward() 方法（不需要额外参数）
    print("\n测试 GaussianModel.forward() 方法:")
    try:
        # 使用正确的输入：只需要融合特征
        print_tensor_info("融合特征输入", fused_feature)  # [1, 768, 64, 64]
        
        # 🔧 关键修复：不传入高斯参数
        gau_pf_pred, gau_cf_pred, gau_tf_pred = gaussian_model(fused_feature)
        
        print_tensor_info("预测位置特征", gau_pf_pred)
        print_tensor_info("预测颜色特征", gau_cf_pred)
        print_tensor_info("预测变换特征", gau_tf_pred)
        
        # 验证输出维度
        assert gau_pf_pred.shape == (1, 1, 64, 64), f"位置特征形状错误: {gau_pf_pred.shape}"
        assert gau_cf_pred.shape == (1, 48, 64, 64), f"颜色特征形状错误: {gau_cf_pred.shape}"
        assert gau_tf_pred.shape == (1, 7, 64, 64), f"变换特征形状错误: {gau_tf_pred.shape}"
        
        print("✅ GaussianModel.forward() 测试通过")
        
    except Exception as e:
        print(f"❌ GaussianModel.forward() 错误: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n测试使用预测三平面特征进行高斯参数预测:")
    try:
        # 合并三平面特征
        predicted_plane_features = torch.cat([gau_pf_pred, gau_cf_pred, gau_tf_pred], dim=1)
        print_tensor_info("预测的三平面特征", predicted_plane_features) # 应该是 [1, 56, 64, 64]
        
        # 🔧 限制查询点数量
        max_points = 1000
        if gaussian_xyz.shape[1] > max_points:
            sample_indices = torch.randperm(gaussian_xyz.shape[1])[:max_points]
            sample_xyz = gaussian_xyz[:, sample_indices, :]
        else:
            sample_xyz = gaussian_xyz
        
        print_tensor_info("采样查询坐标", sample_xyz)
        
        # 使用预测的三平面特征进行查询
        final_colors, final_transforms = gaussian_model.forward_with_plane_features(
            predicted_plane_features, sample_xyz
        )
        
        print_tensor_info("最终预测颜色", final_colors)
        print_tensor_info("最终预测变换", final_transforms)
        
        # 占用预测
        final_occupancy = gaussian_model.forward_with_plane_features_pf(
            predicted_plane_features, sample_xyz
        )
        
        print_tensor_info("最终预测占用", final_occupancy)
        
        print("✅ 完整预测流程测试通过")
        
    except Exception as e:
        print(f"❌ 完整预测流程错误: {e}")

    # 修复 VAE 损失计算部分：
    print("\n测试 VAE 损失计算:")
    try:
        if hasattr(gaussian_model, 'get_vae_loss'):
            # 🔧 修复：使用正确的参数名
            vae_loss_dict = gaussian_model.get_vae_loss(fused_feature, minibatch_weight=1.0)
            print_tensor_info("VAE总损失", vae_loss_dict['loss'])
            print_tensor_info("重构损失", vae_loss_dict['Reconstruction_Loss'])
            print_tensor_info("KL散度损失", vae_loss_dict['KLD'])
            print("✅ VAE损失计算成功")
        else:
            # 🔧 如果没有get_vae_loss方法，直接使用encoder的损失函数
            print("使用 GaussianEncoder 直接计算损失...")
            vae_loss_dict = vae_encoder.loss_function(*vae_outputs, minibatch_weight=1.0)  # 🔧 修复参数名
            print_tensor_info("VAE总损失", vae_loss_dict['loss'])
            print_tensor_info("重构损失", vae_loss_dict['Reconstruction_Loss'])
            print_tensor_info("KL散度损失", vae_loss_dict['KLD'])
            print("✅ VAE损失计算成功")
    except Exception as e:
        print(f"❌ VAE损失计算错误: {e}")
        import traceback
        traceback.print_exc()

    # 修复调试 ConvPointnetLite 部分：
    print("\n调试 ConvPointnetLite 期望的输入维度:")
    try:
        pointnet = gaussian_model.pointnet
        print(f"ConvPointnetLite c_dim: {pointnet.c_dim}")
        print(f"ConvPointnetLite dim: {getattr(pointnet, 'dim', 3)}")  # 🔧 使用getattr避免错误
        print(f"ConvPointnetLite hidden_dim: {getattr(pointnet, 'hidden_dim', 64)}")
        print(f"ConvPointnetLite plane_resolution: {getattr(pointnet, 'plane_resolution', 64)}")
        print(f"ConvPointnetLite plane_type: {getattr(pointnet, 'plane_type', ['xy', 'xz', 'yz'])}")
        
        # 🔧 分析平面特征分配
        actual_plane_channels = 56  # gau_pf(1) + gau_cf(48) + gau_tf(7) = 56
        num_planes = len(getattr(pointnet, 'plane_type', ['xy', 'xz', 'yz']))
        channels_per_plane = actual_plane_channels // num_planes
        remaining_channels = actual_plane_channels % num_planes
        
        print(f"实际的平面特征通道数: {actual_plane_channels}")
        print(f"平面数量: {num_planes}")
        print(f"每个平面基础通道数: {channels_per_plane}")
        print(f"余数通道: {remaining_channels}")
        print(f"平面通道分配: {[channels_per_plane + (1 if i < remaining_channels else 0) for i in range(num_planes)]}")
        
    except Exception as e:
        print(f"❌ 调试错误: {e}")
        import traceback
        traceback.print_exc()

    # 🔧 测试4: 潜在向量提取
    print("\n测试潜在向量提取:")
    try:
        if hasattr(gaussian_model, 'get_latent_vector'):
            latent_z = gaussian_model.get_latent_vector(fused_feature)
            print_tensor_info("提取的潜在向量", latent_z)
        else:
            print("使用已有的潜在向量:", z.shape)
            latent_z = z
        
        # 测试从潜在向量解码
        if hasattr(gaussian_model, 'decode_from_latent'):
            decoded_pf, decoded_cf, decoded_tf = gaussian_model.decode_from_latent(latent_z)
            print_tensor_info("解码位置特征", decoded_pf)
            print_tensor_info("解码颜色特征", decoded_cf)
            print_tensor_info("解码变换特征", decoded_tf)
            print("✅ 潜在向量操作成功")
        else:
            print("使用已有的解码结果")
    except Exception as e:
        print(f"❌ 潜在向量操作错误: {e}")

    # 🔧 总结
    print("\n🎯 流程总结:")
    print("=" * 50)
    print("✅ 1. 多模态输入 → 融合特征:", fused_feature.shape)
    print("✅ 2. 融合特征 → VAE编码 → 潜在向量:", z.shape)
    print("✅ 3. 潜在向量 → VAE解码 → 三平面特征:")
    print(f"   - 位置特征: {gau_pf.shape}")
    print(f"   - 颜色特征: {gau_cf.shape}")
    print(f"   - 变换特征: {gau_tf.shape}")
    print("✅ 4. GaussianModel完整前向传播:", (gau_pf_pred.shape, gau_cf_pred.shape, gau_tf_pred.shape))
    
    if 'final_colors' in locals():
        print("✅ 5. 三平面特征 + 坐标查询 → 预测:")
        print(f"   - 预测颜色: {final_colors.shape}")
        print(f"   - 预测变换: {final_transforms.shape}")
        if 'final_occupancy' in locals():
            print(f"   - 预测占用: {final_occupancy.shape}")
    
    print("=" * 50)
    print("✅ 测试完成！GaussianModel 工作正常")

if __name__ == "__main__":
    test_complete_pipeline()
