import os
import torch
import torch.nn as nn
from typing import Dict, Tuple

# 导入在 g2f_net 子目录中定义的模型结构
# 我们需要 EfficientFeaturePredictor 来加载权重
from .g2f_net.g2fnet import EfficientFeaturePredictor

class GaussianToFeature(nn.Module):
    """
    一个推理模块，用于加载预训练的G2FNet模型，
    并根据动态生成的高斯参数预测多视角特征。
    """
    
    def __init__(self, 
                 checkpoint_path: str, 
                 device: str = 'cuda',
                 spatial_dim: int = 768, 
                 feature_dim: int = 384):
        """
        Args:
            checkpoint_path (str): 预训练模型 'best_model.pth' 的路径。
            device (str): 模型运行的设备。
            spatial_dim (int): G2FNet中空间特征的维度。
            feature_dim (int): G2FNet中输出特征的维度。
        """
        super().__init__()
        self.device = torch.device(device)
        
        print(f"正在初始化 GaussianToFeature (G2F) 模块...")
        
        # 1. 初始化模型结构，与训练时保持一致
        self.model = EfficientFeaturePredictor(
            spatial_dim=spatial_dim,
            feature_dim=feature_dim
        ).to(self.device)
        
        # 2. 加载预训练的权重
        try:
            print(f"正在从 '{checkpoint_path}' 加载预训练权重...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 兼容不同保存格式
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            print(f"✅ 权重加载成功！模型来自 epoch {checkpoint.get('epoch', 'N/A')}, loss: {checkpoint.get('loss', 'N/A'):.4f}")
            
        except FileNotFoundError:
            print(f"❌ 错误: 找不到检查点文件 '{checkpoint_path}'。模型将使用随机初始化的权重。")
        except Exception as e:
            print(f"❌ 错误: 加载权重时发生错误: {e}。模型将使用随机初始化的权重。")
            
        # 3. 设置为评估模式
        self.model.eval()
        
        # 4. 冻结模型参数，因为它只用于推理
        for param in self.model.parameters():
            param.requires_grad = False
            
    def _reconstruct_gaussian_attributes(self, 
                                         pred_color: torch.Tensor, 
                                         pred_gs: torch.Tensor, 
                                         pred_occ: torch.Tensor) -> torch.Tensor:
        """
        将 combine_model.py 中分散的预测参数重组成 [B, N, 56] 的属性张量。
        
        Args:
            pred_color (torch.Tensor): [B, N, 48] 预测的SH系数。
            pred_gs (torch.Tensor): [B, N, 7] 预测的 scale(3) 和 rotation(4)。
            pred_occ (torch.Tensor): [B, N, 1] 预测的透明度。
            
        Returns:
            torch.Tensor: [B, N, 56] 重组后的高斯属性。
        """
        # 从 pred_gs 中分离 scale 和 rotation
        pred_scale = pred_gs[:, :, :3]
        pred_rotation = pred_gs[:, :, 3:7]
        
        # 按照 gt_gaussian 的格式 [3:56] 进行拼接
        # 格式: opacity(1), sh(48), scale(3), rotation(4)
        # 注意：这里的顺序需要和 gs_dataloader.py 中 gt_gaussian 的构造方式完全一致
        # 根据 gs_dataloader.py, gt_gaussian 的格式是 [:, 3:]
        # 假设原始 gaussian.npy 的格式是: xyz(3), opacity(1), sh(48), scale(3), rotation(4)
        # 那么 gt_gaussian [:, 3:] 就是 [opacity, sh, scale, rotation]
        
        # 确保 pred_occ 的维度正确
        if pred_occ.dim() == 2:
            pred_occ = pred_occ.unsqueeze(-1) # [B, N] -> [B, N, 1]

        attributes = torch.cat([
            pred_occ,       # [:, 3:4]
            pred_color,     # [:, 4:52]
            pred_scale,     # [:, 52:55]
            pred_rotation   # [:, 55:59]
        ], dim=-1)
        
        # 验证维度是否为 56
        expected_dim = 1 + 48 + 3 + 4
        assert attributes.shape[-1] == expected_dim, \
            f"重组后的属性维度错误！应为 {expected_dim}, 实际为 {attributes.shape[-1]}"
            
        return attributes

    def forward(self, 
                gaussian_xyz: torch.Tensor,
                pred_color: torch.Tensor, 
                pred_gs: torch.Tensor, 
                pred_occ: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        接收动态生成的高斯参数，并输出预测的多视角特征。

        Args:
            gaussian_xyz (torch.Tensor): [B, N, 3] 固定的高斯球中心点坐标。
            pred_color (torch.Tensor): [B, N, 48] 预测的SH系数。
            pred_gs (torch.Tensor): [B, N, 7] 预测的 scale(3) 和 rotation(4)。
            pred_occ (torch.Tensor): [B, N, 1] 预测的透明度。

        Returns:
            Dict[str, torch.Tensor]: 包含三个视角特征的字典。
                {'front_features': tensor, 'side_features': tensor, 'top_features': tensor}
        """
        # 将输入参数移动到正确的设备
        gaussian_xyz = gaussian_xyz.to(self.device)
        pred_color = pred_color.to(self.device)
        pred_gs = pred_gs.to(self.device)
        pred_occ = pred_occ.to(self.device)
        
        # 重组高斯属性
        reconstructed_attributes = self._reconstruct_gaussian_attributes(
            pred_color, pred_gs, pred_occ
        )
        
        # 构建模型所需的输入字典
        batch_data = {
            'gaussian_xyz': gaussian_xyz,
            'gt_gaussian': reconstructed_attributes
        }
        
        # 使用预训练模型进行推理
        # 注意：这里不再使用 torch.cuda.amp.autocast，因为它可能会影响梯度流
        # 训练循环中的主 autocast 会处理它
        predicted_features = self.model(batch_data)
            
        return predicted_features

# --- 使用示例 (用于测试和演示) ---
def test_g2f_module():
    print("🧪 测试 GaussianToFeature 模块...")
    
    # 假设的检查点路径
    # 请确保你有一个训练好的 'best_model.pth' 文件
    CHECKPOINT_PATH = 'best_model.pth' 
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"警告: 找不到测试用的检查点文件 '{CHECKPOINT_PATH}'。将无法完成测试。")
        # 创建一个假的权重文件用于测试代码结构
        dummy_model = EfficientFeaturePredictor()
        torch.save(dummy_model.state_dict(), CHECKPOINT_PATH)
        print("已创建一个临时的虚拟权重文件用于测试。")

    # 初始化模块
    g2f_module = GaussianToFeature(checkpoint_path=CHECKPOINT_PATH)

    # 模拟 combine_model.py 的输出
    batch_size = 4
    num_points = 16000
    
    mock_gaussian_xyz = torch.randn(batch_size, num_points, 3)
    mock_pred_color = torch.randn(batch_size, num_points, 48)
    mock_pred_gs = torch.randn(batch_size, num_points, 7)
    mock_pred_occ = torch.rand(batch_size, num_points, 1)

    print("\n模拟输入数据形状:")
    print(f"  gaussian_xyz: {mock_gaussian_xyz.shape}")
    print(f"  pred_color:   {mock_pred_color.shape}")
    print(f"  pred_gs:      {mock_pred_gs.shape}")
    print(f"  pred_occ:     {mock_pred_occ.shape}")

    # 调用 forward 方法
    output_features = g2f_module(
        mock_gaussian_xyz,
        mock_pred_color,
        mock_pred_gs,
        mock_pred_occ
    )

    print("\n✅ 模块成功执行！")
    print("输出特征字典的键:", output_features.keys())
    
    for view_name, feature_tensor in output_features.items():
        print(f"  - {view_name}:")
        print(f"    形状: {feature_tensor.shape}")
        print(f"    设备: {feature_tensor.device}")

    # 清理临时文件
    if "虚拟权重文件" in locals().get("CHECKPOINT_PATH_info", ""):
        os.remove(CHECKPOINT_PATH)

if __name__ == '__main__':
    # 这个测试脚本可以独立运行，以验证 G2F 模块是否能正确加载和执行
    # 需要将 g2f_net 文件夹放在同级目录
    test_g2f_module()