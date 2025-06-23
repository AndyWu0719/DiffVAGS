#!/usr/bin/env python3

import os
import shutil
from pathlib import Path

def reorganize_multiview_images():
    
    base_path = Path("/media/guest1/WD6TB/WD6TB/Andy/Datasets/lightdiffgsdata/4")
    training_data_path = base_path / "training_data"
    
    if not training_data_path.exists():
        print(f"训练数据目录不存在: {training_data_path}")
        return
    
    moved_count = 0
    failed_count = 0
    
    # 遍历训练数据目录中的每个模型
    for model_dir in training_data_path.iterdir():
        if not model_dir.is_dir() or not model_dir.name.isdigit():
            continue
            
        model_idx = model_dir.name
        
        # 检查是否已有图像目录
        target_images_dir = model_dir / "images"
        if target_images_dir.exists():
            print(f"⏭️  模型 {model_idx}: 图像目录已存在，跳过")
            continue
        
        # 查找原始图像位置
        source_images_dir = base_path / model_idx / "1" / "images"
        
        if source_images_dir.exists():
            try:
                # 复制图像目录到训练数据目录
                shutil.copytree(source_images_dir, target_images_dir)
                
                # 也复制transforms文件
                source_transforms = base_path / model_idx / "1" / "transforms_train.json"
                if source_transforms.exists():
                    target_transforms = model_dir / "transforms_train.json"
                    shutil.copy2(source_transforms, target_transforms)
                
                print(f"✅ 模型 {model_idx}: 成功移动 {len(list(target_images_dir.glob('*.png')))} 张图像")
                moved_count += 1
                
            except Exception as e:
                print(f"❌ 模型 {model_idx}: 移动失败 - {e}")
                failed_count += 1
        else:
            print(f"⚠️  模型 {model_idx}: 未找到原始图像 {source_images_dir}")
            failed_count += 1
    
    print(f"\n📊 重组完成:")
    print(f"  成功移动: {moved_count} 个模型")
    print(f"  失败/跳过: {failed_count} 个模型")

if __name__ == "__main__":
    reorganize_multiview_images()