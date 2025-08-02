import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import argparse
import os

# 从同级目录的 foundation_models.py 文件中导入 aux_foundation_model 类
from foundation_models import aux_foundation_model

def extract_and_save_multiview_features(
    data_root: str, 
    output_path: str,
    model_type: str = 'dinov2', 
    device: str = 'cuda',
    batch_size: int = 16
) -> None:
    """
    遍历指定数据根目录下的所有模型文件夹，为每个模型提取三个标准视角的全局特征向量，
    并将结果保存到指定的 .pt 文件中。

    Args:
        data_root (str): 数据集的根目录，例如 '/media/.../training_data'。
        output_path (str): 保存提取特征的 .pt 文件路径。
        model_type (str): 要使用的基础模型类型，'dinov2' 或 'mae'。
        device (str): 运行模型的设备，'cuda' 或 'cpu'。
        batch_size (int): 批量处理图像的大小，以提高效率。
    """
    
    print(f"Initializing foundation model: {model_type} on {device}...")
    # 1. 初始化基础模型
    foundation_model = aux_foundation_model(type=model_type).to(device)
    foundation_model.eval()

    # 2. 定义要查找的视图和对应的文件名
    view_definitions = {
        'front': 'images_r2_000.png',
        'side': 'images_r0_000.png',
        'top': 'images_r1_000.png'
    }

    # 3. 准备图像加载和转换
    image_loader = transforms.Compose([
        transforms.ToTensor()
    ])

    # 4. 收集所有需要处理的图像路径
    all_image_paths = []
    model_id_map = [] # 用于记录每个图像对应的模型ID和视图名称
    data_path = Path(data_root)
    model_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()]) # 排序以保证一致性
    
    print(f"Scanning {len(model_dirs)} model directories to find images...")
    for model_dir in tqdm(model_dirs, desc="Scanning directories"):
        model_id = model_dir.name
        images_path = model_dir / 'images'

        if not images_path.exists():
            continue

        # 检查该模型是否包含所有三个视图
        view_paths = {view: images_path / fname for view, fname in view_definitions.items()}
        if all(p.exists() for p in view_paths.values()):
            for view_name, image_path in view_paths.items():
                all_image_paths.append(image_path)
                model_id_map.append({'model_id': model_id, 'view_name': view_name})

    if not all_image_paths:
        print("Error: No valid image sets found. Please check directory structure and filenames.")
        return

    print(f"Found {len(all_image_paths) // 3} complete model view sets. Starting feature extraction with batch size {batch_size}...")

    # 5. 批量提取特征
    all_features_flat = []
    with torch.no_grad():
        for i in tqdm(range(0, len(all_image_paths), batch_size), desc="Extracting Features"):
            batch_paths = all_image_paths[i:i+batch_size]
            batch_images = []
            
            for image_path in batch_paths:
                try:
                    image = Image.open(image_path).convert('RGB')
                    # foundation_model内部会进行预处理，这里只需加载
                    batch_images.append(image)
                except Exception as e:
                    print(f"Warning: Could not load image {image_path}, skipping. Error: {e}")
                    # 添加一个None占位符，后面会处理
                    batch_images.append(None)
            
            # 过滤掉加载失败的图像
            valid_images = [img for img in batch_images if img is not None]
            if not valid_images:
                continue

            # foundation_model的preprocess_image期望一个图像列表
            processed_batch = foundation_model.preprocess_image(valid_images).to(device)
            
            # 提取全局特征
            global_features_batch = foundation_model.extract_global_features(processed_batch)
            all_features_flat.extend(list(global_features_batch.cpu()))

    # 6. 重组特征字典
    final_features_dict = {}
    feature_idx = 0
    for mapping in model_id_map:
        model_id = mapping['model_id']
        view_name = mapping['view_name']
        
        if model_id not in final_features_dict:
            final_features_dict[model_id] = {}
            
        final_features_dict[model_id][view_name] = all_features_flat[feature_idx]
        feature_idx += 1

    # 7. 保存特征字典到文件
    print(f"\nFeature extraction complete. Successfully processed {len(final_features_dict)} models.")
    print(f"Saving features to '{output_path}'...")
    
    try:
        torch.save(final_features_dict, output_path)
        print(f"✅ Features successfully saved!")
    except Exception as e:
        print(f"❌ Error saving features: {e}")

# --- 使用示例 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract and save multi-view features from a dataset.")
    parser.add_argument(
        "--data_root", 
        type=str, 
        default='/media/guest1/WD6TB/WD6TB/Andy/Datasets/lightdiffgsdata/03001627/training_data',
        help="Path to the root directory of the training data."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default='multiview_features_dinov2.pt',
        help="Path to save the output .pt file."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default='dinov2', 
        choices=['dinov2', 'mae'],
        help="Foundation model to use for feature extraction."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device to run the model on ('cuda' or 'cpu')."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Batch size for feature extraction to improve performance."
    )
    
    args = parser.parse_args()

    # 调用主函数
    extract_and_save_multiview_features(
        data_root=args.data_root,
        output_path=args.output_file,
        model_type=args.model,
        device=args.device,
        batch_size=args.batch_size
    )

    # --- 验证保存的文件 ---
    if os.path.exists(args.output_file):
        print("\n--- Verifying saved file ---")
        try:
            loaded_features = torch.load(args.output_file)
            print(f"Successfully loaded '{args.output_file}'.")
            print(f"Total models in file: {len(loaded_features)}")
            
            if loaded_features:
                first_model_id = list(loaded_features.keys())[0]
                print(f"\nExample data for model '{first_model_id}':")
                for view_name, feature_tensor in loaded_features[first_model_id].items():
                    print(f"  - View: {view_name}, Feature Shape: {feature_tensor.shape}, DType: {feature_tensor.dtype}")
        except Exception as e:
            print(f"Error verifying saved file: {e}")