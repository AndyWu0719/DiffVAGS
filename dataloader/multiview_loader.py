#!/usr/bin/env python3

import os
import torch
import torch.utils.data
import numpy as np
from pathlib import Path

from utils.image_utils import MultiViewImageHandler

class MultiViewGaussianLoader(torch.utils.data.Dataset):
    """
    一个专用的数据加载器，用于在启用视觉对齐(VA)模式时，
    同时加载3D高斯数据和对应的多视角2D图像。
    它借鉴了 GaussianLoader 的简洁性，并消除了所有硬编码路径。
    """
    
    def __init__(self, data_root: str, multiview_specs: dict):
        """
        初始化加载器。
        Args:
            data_root (str): 数据集的根目录。
            multiview_specs (dict): 多视角图像处理的配置。这是必需的。
        """
        if multiview_specs is None:
            raise ValueError("MultiViewGaussianLoader requires 'multiview_specs' to be provided.")

        self.data_root = Path(data_root)
        self.data_files = []
        
        # 1. 直接从配置中初始化图像处理器
        self.image_handler = MultiViewImageHandler(multiview_specs)
        # 从配置中获取图像文件夹的名称，默认为 "images"
        self.images_folder_name = multiview_specs.get("images_folder", "images")
        
        print(f"✅ MultiViewLoader initialized for {self.image_handler.num_views} views from folder '{self.images_folder_name}'.")
        
        self._scan_data_directory()
        
        if not self.data_files:
            raise RuntimeError(f"No valid data (gaussian.npy, occ.npy, and images folder) found in {self.data_root}")
        
        print(f"Found {len(self.data_files)} complete models with multiview data.")

    def _scan_data_directory(self):
        """
        扫描数据目录，只添加包含所有必需文件（高斯、遮挡、图像）的模型。
        """
        for model_dir in self.data_root.iterdir():
            if not model_dir.is_dir():
                continue
                
            gaussian_file = model_dir / "gaussian.npy"
            occ_file = model_dir / "occ.npy"
            images_dir = model_dir / self.images_folder_name
            
            # 2. 确保所有必需的文件和文件夹都存在
            if not (gaussian_file.exists() and occ_file.exists() and images_dir.exists()):
                continue
            
            # (可选但推荐) 进一步检查是否包含必需的视图文件
            if not self.image_handler.check_views_exist(images_dir):
                print(f"Warning: Skipping {model_dir.name}, images folder exists but missing required views.")
                continue

            data_info = {
                'model_idx': int(model_dir.name),
                'gaussian_file': str(gaussian_file),
                'occ_file': str(occ_file),
                'images_path': str(images_dir) # 路径现在是保证存在的
            }
            self.data_files.append(data_info)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_info = self.data_files[idx]
        
        try:
            # --- 加载和处理 3D 高斯数据 (与 gaussian_loader.py 保持一致) ---
            gaussian_data = np.load(data_info['gaussian_file'])
            occ_data = np.load(data_info['occ_file'])
            
            occ_indices = np.random.choice(occ_data.shape[0], min(80000, occ_data.shape[0]), replace=False)
            occ_sampled = occ_data[occ_indices, :]
            
            gaussian_indices = np.random.choice(gaussian_data.shape[0], min(16000, gaussian_data.shape[0]), replace=False)
            gaussian_sampled = gaussian_data[gaussian_indices, :]
            
            gaussian_processed = gaussian_sampled.copy()
            gaussian_processed[:, 52:55] = np.exp(gaussian_processed[:, 52:55])
            norm = np.linalg.norm(gaussian_processed[:, 55:59], ord=2, axis=-1, keepdims=True)
            gaussian_processed[:, 55:59] = gaussian_processed[:, 55:59] / norm
            
            gs = torch.from_numpy(gaussian_data).float()
            gaussian_xyz = torch.from_numpy(gaussian_processed[:, :3]).float()
            gaussian_gt = torch.from_numpy(gaussian_processed[:, 3:]).float()
            occ_xyz = torch.from_numpy(occ_sampled[:, :3]).float()
            occ = torch.from_numpy(occ_sampled[:, 3:]).float()
            
            data_dict = {
                'gaussians': gs,
                'gaussian_xyz': gaussian_xyz,
                'gt_gaussian': gaussian_gt,
                'occ_xyz': occ_xyz,
                'occ': occ,
                'model_id': f"class_{data_info.get('class_idx', 'N/A')}_model_{data_info['model_idx']}"
            }
            
            # --- 3. 加载多视角图像 ---
            # 因为 _scan_data_directory 已经确保了路径存在，所以这里可以直接加载
            multiview_images = self.image_handler.load_multiview_images(data_info['images_path'])
            if multiview_images:
                data_dict['multiview_images'] = multiview_images
            else:
                # 如果由于某种原因（如文件损坏）加载失败，则跳过此样本
                print(f"Warning: Failed to load images for model {data_info['model_idx']}, skipping.")
                return self.__getitem__((idx + 1) % len(self))

            return data_dict
            
        except Exception as e:
            print(f"Error loading data at index {idx} (model_idx: {data_info.get('model_idx', 'N/A')}): {e}")
            # 跳到下一个样本，避免因单个损坏文件导致训练崩溃
            return self.__getitem__((idx + 1) % len(self))