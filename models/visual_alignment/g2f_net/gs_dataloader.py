import os
import torch
import json
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, List, Tuple
import numpy as np
import time
import pickle
from tqdm import tqdm
import sys

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

class MultiViewGaussianDataset(Dataset):
    """高效加载高斯数据和多视角图像的数据集"""
    
    def __init__(self, 
                 gaussian_data_path: str,
                 image_data_path: str,
                 view_mapping: Dict[str, str] = None,
                 cache_path: str = "dataset_cache.pkl",
                 max_samples: int = None):
        """
        Args:
            gaussian_data_path: 高斯数据路径
            image_data_path: 图像数据路径
            view_mapping: 视角映射
            cache_path: 预加载缓存路径
            max_samples: 最大样本数量（用于测试）
        """
        self.gaussian_data_path = gaussian_data_path
        self.image_data_path = image_data_path
        
        # 默认视角映射
        self.view_mapping = view_mapping or {
            'front': 'images_r0_',  # 正视图
            'side': 'images_r1_',   # 侧视图
            'top': 'images_r2_'     # 俯视图
        }
        
        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((518, 518)),  # 减小尺寸加速处理
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 检查缓存
        if os.path.exists(cache_path):
            print(f"加载数据集缓存: {cache_path}")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.valid_sample_ids = cache_data['sample_ids']
                self.metadata = cache_data['metadata']
        else:
            print(f"创建数据集缓存: {cache_path}")
            self.valid_sample_ids = self._get_valid_samples()
            self.metadata = self._preload_metadata()
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'sample_ids': self.valid_sample_ids,
                    'metadata': self.metadata
                }, f)
        
        # 限制样本数量（用于测试）
        if max_samples:
            self.valid_sample_ids = self.valid_sample_ids[:max_samples]
            self.metadata = self.metadata[:max_samples]
        
        print(f"找到 {len(self.valid_sample_ids)} 个有效样本")
    
    def _get_valid_samples(self) -> List[str]:
        """获取同时存在高斯数据和图像数据的有效样本"""
        valid_samples = []
        
        # 遍历高斯数据路径下的所有子目录
        for sample_dir in os.listdir(self.gaussian_data_path):
            sample_path = os.path.join(self.gaussian_data_path, sample_dir)
            
            # 确保是目录
            if not os.path.isdir(sample_path):
                continue
                
            # 检查高斯数据文件是否存在
            gaussian_exists = (
                os.path.exists(os.path.join(sample_path, 'gaussian.npy')) and
                os.path.exists(os.path.join(sample_path, 'occ.npy'))
            )
            
            # 检查对应的图像数据目录
            image_sample_path = os.path.join(self.image_data_path, sample_dir, 'images')
            image_exists = os.path.exists(image_sample_path) and len(os.listdir(image_sample_path)) > 0
            
            # 如果两者都存在，则添加到有效样本
            if gaussian_exists and image_exists:
                valid_samples.append(sample_dir)
                
        return sorted(valid_samples, key=lambda x: int(x))
    
    def _preload_metadata(self) -> List[dict]:
        """预加载所有样本的元数据"""
        metadata = []
        for sample_id in tqdm(self.valid_sample_ids, desc="预加载元数据"):
            # 高斯数据路径
            gaussian_dir = os.path.join(self.gaussian_data_path, sample_id)
            
            # 图像数据目录
            image_dir = os.path.join(self.image_data_path, sample_id, 'images')
            
            # 获取所有图像文件
            if os.path.exists(image_dir):
                image_files = sorted([
                    f for f in os.listdir(image_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
            else:
                image_files = []
            
            # 确定视角图像路径
            view_paths = {}
            for view_name, prefix in self.view_mapping.items():
                # 找到匹配前缀的第一个图像
                match = next((f for f in image_files if f.startswith(prefix)), None)
                view_paths[view_name] = os.path.join(image_dir, match) if match else None
            
            metadata.append({
                'gaussian_dir': gaussian_dir,
                'image_dir': image_dir,
                'view_paths': view_paths
            })
        return metadata
    
    def _load_gaussian_data(self, meta: dict) -> Dict[str, torch.Tensor]:
        """加载并预处理高斯数据"""
        gaussian_file = os.path.join(meta['gaussian_dir'], 'gaussian.npy')
        occ_file = os.path.join(meta['gaussian_dir'], 'occ.npy')
        
        try:
            # 加载原始数据 (numpy)
            gaussian = np.load(gaussian_file)  # [N, 59]
            occ = np.load(occ_file)  # [M, K]
            
            # 转换为tensor
            gaussian = torch.from_numpy(gaussian).float()
            occ = torch.from_numpy(occ).float()
            
            # 采样高斯点 (16000个)
            if gaussian.shape[0] > 16000:
                indices = torch.randperm(gaussian.shape[0])[:16000]
            else:
                indices = torch.randint(0, gaussian.shape[0], (16000,))
            sampled_gaussian = gaussian[indices]
            
            # 分离xyz坐标和其他属性
            gaussian_xyz = sampled_gaussian[:, :3]  # [16000, 3]
            gaussian_attrs = sampled_gaussian[:, 3:]  # [16000, 56]
            
            # 处理缩放因子和旋转四元数
            gaussian_attrs[:, 49:52] = torch.exp(gaussian_attrs[:, 49:52])  # 缩放因子
            norm = torch.norm(gaussian_attrs[:, 52:56], p=2, dim=-1, keepdim=True)
            norm = torch.where(norm == 0, torch.ones_like(norm), norm)
            gaussian_attrs[:, 52:56] = gaussian_attrs[:, 52:56] / norm
            
            # 采样占用点 (80000个)
            if occ.shape[0] > 80000:
                occ_indices = torch.randperm(occ.shape[0])[:80000]
            else:
                occ_indices = torch.randint(0, occ.shape[0], (80000,))
            sampled_occ = occ[occ_indices]
            
            occ_xyz = sampled_occ[:, :3]  # [80000, 3]
            occ_attrs = sampled_occ[:, 3:]  # [80000, K-3]
            
            return {
                'gaussian_xyz': gaussian_xyz,
                'gt_gaussian': gaussian_attrs,
                'occ_xyz': occ_xyz,
                'occ': occ_attrs
            }
            
        except Exception as e:
            print(f"加载高斯数据 {meta['gaussian_dir']} 时出错: {e}")
            # 返回空数据
            return {
                'gaussian_xyz': torch.zeros(16000, 3),
                'gt_gaussian': torch.zeros(16000, 56),
                'occ_xyz': torch.zeros(80000, 3),
                'occ': torch.zeros(80000, 1)
            }
    
    def _load_view_images(self, view_paths: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """加载多视角图像"""
        view_images = {}
        for view_name, path in view_paths.items():
            if not path or not os.path.exists(path):
                # 如果路径不存在，使用占位符
                view_images[f'{view_name}_image'] = torch.zeros(3, 518, 518)
                continue
            
            try:
                image = Image.open(path).convert('RGB')
                image_tensor = self.image_transform(image)
                view_images[f'{view_name}_image'] = image_tensor
            except Exception as e:
                print(f"加载图像 {path} 时出错: {e}")
                view_images[f'{view_name}_image'] = torch.zeros(3, 518, 518)
        
        return view_images
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取一个训练样本"""
        sample_id = self.valid_sample_ids[idx]
        meta = self.metadata[idx]
        
        # 获取高斯数据
        gaussian_data = self._load_gaussian_data(meta)
        
        # 获取多视角图像
        view_images = self._load_view_images(meta['view_paths'])
        
        # 合并数据
        combined_data = {
            'sample_id': sample_id,
            **gaussian_data,
            **view_images
        }
        
        return combined_data
    
    def __len__(self) -> int:
        return len(self.valid_sample_ids)

# 测试数据集
def test_multiview_dataset():
    """测试数据集加载"""
    print("🧪 测试MultiViewGaussianDataset...")
    
    # 使用您提供的实际路径
    dataset = MultiViewGaussianDataset(
        gaussian_data_path="/media/andywu/WD6TB/WD6TB/Andy/Datasets/lightdiffgsdata/03001627/convert_data",
        image_data_path="/media/andywu/WD6TB/WD6TB/Andy/Datasets/lightdiffgsdata/03001627/training_data",
        max_samples=5  # 只测试前5个样本
    )
    
    print(f"✅ 数据集大小: {len(dataset)}")
    
    if len(dataset) > 0:
        # 测试第一个样本
        print(f"\n🔍 测试样本...")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\n样本ID: {sample['sample_id']}")
            
            print(f"高斯数据形状:")
            print(f"  gaussian_xyz: {sample['gaussian_xyz'].shape}")
            print(f"  gt_gaussian: {sample['gt_gaussian'].shape}")
            
            print(f"图像数据:")
            for view in ['front', 'side', 'top']:
                image_key = f'{view}_image'
                image_tensor = sample[image_key]
                is_blank = torch.all(image_tensor == 0)
                print(f"  {view}: {'❌ 空白图像' if is_blank else '✅ 有效图像'} ({image_tensor.shape})")

if __name__ == "__main__":
    test_multiview_dataset()