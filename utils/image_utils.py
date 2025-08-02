#!/usr/bin/env python3

import os
import random
import json
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from typing import List, Dict, Optional

class MultiViewImageHandler:
    """
    一个经过简化的、健壮的多视角图像处理器。
    它的职责是：从一个给定的图像文件夹路径中，加载一个指定的视图列表。
    """
    
    def __init__(self, multiview_specs: dict):
        """
        Args:
            multiview_specs (dict): 视觉对齐的配置。
                - "views_to_load" (List[str]): 需要加载的视图文件名列表。例如: ["front.png", "side.png", "top.png"]
                - "image_size" (int): 图像的目标尺寸。
                - "transforms" (dict, optional): 自定义图像变换。
        """
        self.multiview_specs = multiview_specs
        
        # 1. 从配置中获取要加载的特定视图列表
        self.views_to_load = self.multiview_specs.get("views_to_load")
        if not self.views_to_load:
            raise ValueError("'views_to_load' must be specified in VisualAlignmentSpecs.")
            
        self.image_size = self.multiview_specs.get("image_size", 224)
        
        # 2. 创建图像变换流水线
        self.image_transform = self._create_transforms()
        
        print(f"ImageHandler initialized to load {len(self.views_to_load)} specific views: {self.views_to_load}")

    @property
    def num_views(self) -> int:
        """动态返回视图数量"""
        return len(self.views_to_load)

    def _create_transforms(self) -> transforms.Compose:
        """
        创建图像变换。这里的默认归一化参数与DINOv2等模型一致，是正确的。
        """
        transforms_list = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ]
        
        # 检查是否有自定义的归一化配置
        norm_config = self.multiview_specs.get("transforms", {}).get("normalize")
        if norm_config:
            transforms_list.append(transforms.Normalize(mean=norm_config["mean"], std=norm_config["std"]))
        else:
            # 使用标准的ImageNet归一化，这对于DINOv2是正确的
            transforms_list.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ))
        
        return transforms.Compose(transforms_list)
    
    def check_views_exist(self, images_dir_path: Path) -> bool:
        """
        检查指定的图像文件夹中是否存在所有必需的视图文件。
        这是给 multiview_loader 使用的辅助函数，以确保数据的完整性。
        """
        if not images_dir_path.is_dir():
            return False
        
        for view_filename in self.views_to_load:
            if not (images_dir_path / view_filename).exists():
                return False # 只要有一个视图不存在，就返回False
        
        return True # 所有视图都存在

    def load_multiview_images(self, images_dir_path: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        从给定的文件夹路径加载、变换并返回一个包含多视角图像的字典。
        
        Args:
            images_dir_path (str): 包含图像的文件夹的**确切**路径。
        
        Returns:
            一个字典，键是视图名（不带扩展名），值是图像张量。
            如果任何图像加载失败，则返回 None。
        """
        try:
            images_dir = Path(images_dir_path)
            loaded_images = {}

            for view_filename in self.views_to_load:
                img_path = images_dir / view_filename
                
                if not img_path.exists():
                    print(f"Warning: Required image file not found: {img_path}")
                    return None

                img = Image.open(img_path).convert('RGB')
                img_tensor = self.image_transform(img)
                
                # 使用视图的文件名（不含扩展名）作为键
                view_key = Path(view_filename).stem
                loaded_images[view_key] = img_tensor
            
            # 确保我们成功加载了所有期望的视图
            if len(loaded_images) == self.num_views:
                return loaded_images
            else:
                return None

        except Exception as e:
            print(f"Warning: An error occurred while loading multiview images from {images_dir_path}: {e}")
            return None
