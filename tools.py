import torch
import os
from models.visual_alignment.g2fnet import EfficientFeaturePredictor # 确保可以导入模型结构

def extract_adapter_weights(checkpoint_path, output_path):
    """
    从 g2f 模型检查点中提取 feature_adapter 的权重并保存。

    Args:
        checkpoint_path (str): 训练好的 g2f 模型检查点路径 (best_model.pth)。
        output_path (str): 保存 adapter 权重的目标路径。
    """
    print(f"正在从 '{checkpoint_path}' 加载模型...")
    
    # 1. 初始化模型结构以加载权重
    #    这里的参数需要与你训练时使用的参数一致
    model = EfficientFeaturePredictor(feature_dim=384) # 假设 feature_dim 是 384
    
    # 2. 加载完整的模型 state_dict
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    print("模型加载成功。")

    # 3. 提取 adapter 的 state_dict
    #    路径是: model -> feature_extractor -> feature_adapter
    adapter_state_dict = model.feature_extractor.feature_adapter.state_dict()
    
    # 4. 保存 adapter 的权重
    torch.save(adapter_state_dict, output_path)
    print(f"✅ Adapter 权重已成功提取并保存到 '{output_path}'")

if __name__ == '__main__':
    # --- 配置 ---
    G2F_CHECKPOINT = 'best_model.pth' # 你训练好的g2f模型文件名
    ADAPTER_OUTPUT = 'adapter_weights.pth' # 输出的adapter权重文件名
    
    if not os.path.exists(G2F_CHECKPOINT):
        print(f"错误: 找不到 g2f 检查点文件 '{G2F_CHECKPOINT}'")
    else:
        extract_adapter_weights(G2F_CHECKPOINT, ADAPTER_OUTPUT)