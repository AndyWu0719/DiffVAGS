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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

class MultiViewGaussianDataset(Dataset):
    
    def __init__(self, 
                 gaussian_data_path: str,
                 image_data_path: str,
                 view_mapping: Dict[str, str] = None,
                 cache_path: str = "/home/dwubf/workplace/DiffVAGS/experiments/g2f/dataset_cache.pkl",
                 max_samples: int = None):
        
        self.gaussian_data_path = gaussian_data_path
        self.image_data_path = image_data_path
        
        self.view_mapping = view_mapping or {
            'front': 'images_r0_',
            'side': 'images_r1_',
            'top': 'images_r2_'
        }
        
        self.image_transform = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        if os.path.exists(cache_path):
            print(f"Load cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.valid_sample_ids = cache_data['sample_ids']
                self.metadata = cache_data['metadata']
        else:
            print(f"Build cache: {cache_path}")
            self.valid_sample_ids = self._get_valid_samples()
            self.metadata = self._preload_metadata()
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'sample_ids': self.valid_sample_ids,
                    'metadata': self.metadata
                }, f)
        
        if max_samples:
            self.valid_sample_ids = self.valid_sample_ids[:max_samples]
            self.metadata = self.metadata[:max_samples]

        print(f"Found {len(self.valid_sample_ids)} valid samples")

    def _get_valid_samples(self) -> List[str]:
        valid_samples = []
        
        for sample_dir in os.listdir(self.gaussian_data_path):
            sample_path = os.path.join(self.gaussian_data_path, sample_dir)
            
            if not os.path.isdir(sample_path):
                continue
                
            gaussian_exists = (
                os.path.exists(os.path.join(sample_path, 'gaussian.npy')) and
                os.path.exists(os.path.join(sample_path, 'occ.npy'))
            )
            
            image_sample_path = os.path.join(self.image_data_path, sample_dir, 'images')
            image_exists = os.path.exists(image_sample_path) and len(os.listdir(image_sample_path)) > 0
            
            if gaussian_exists and image_exists:
                valid_samples.append(sample_dir)
                
        return sorted(valid_samples, key=lambda x: int(x))
    
    def _preload_metadata(self) -> List[dict]:
        metadata = []
        for sample_id in tqdm(self.valid_sample_ids, desc="Preloading metadata"):
            gaussian_dir = os.path.join(self.gaussian_data_path, sample_id)
            
            image_dir = os.path.join(self.image_data_path, sample_id, 'images')
            
            if os.path.exists(image_dir):
                image_files = sorted([
                    f for f in os.listdir(image_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
            else:
                image_files = []
            
            view_paths = {}
            for view_name, prefix in self.view_mapping.items():
                match = next((f for f in image_files if f.startswith(prefix)), None)
                view_paths[view_name] = os.path.join(image_dir, match) if match else None
            
            metadata.append({
                'gaussian_dir': gaussian_dir,
                'image_dir': image_dir,
                'view_paths': view_paths
            })
        return metadata
    
    def _load_gaussian_data(self, meta: dict) -> Dict[str, torch.Tensor]:
        gaussian_file = os.path.join(meta['gaussian_dir'], 'gaussian.npy')
        occ_file = os.path.join(meta['gaussian_dir'], 'occ.npy')
        
        try:
            gaussian = np.load(gaussian_file)  # [N, 59]
            occ = np.load(occ_file)  # [M, K]
            
            gaussian = torch.from_numpy(gaussian).float()
            occ = torch.from_numpy(occ).float()
            
            if gaussian.shape[0] > 16000:
                indices = torch.randperm(gaussian.shape[0])[:16000]
            else:
                indices = torch.randint(0, gaussian.shape[0], (16000,))
            sampled_gaussian = gaussian[indices]
            
            gaussian_xyz = sampled_gaussian[:, :3]  # [16000, 3]
            gaussian_attrs = sampled_gaussian[:, 3:]  # [16000, 56]
            
            gaussian_attrs[:, 49:52] = torch.exp(gaussian_attrs[:, 49:52])
            norm = torch.norm(gaussian_attrs[:, 52:56], p=2, dim=-1, keepdim=True)
            norm = torch.where(norm == 0, torch.ones_like(norm), norm)
            gaussian_attrs[:, 52:56] = gaussian_attrs[:, 52:56] / norm
            
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
            print(f"Load gaussian data {meta['gaussian_dir']} error: {e}")
            return {
                'gaussian_xyz': torch.zeros(16000, 3),
                'gt_gaussian': torch.zeros(16000, 56),
                'occ_xyz': torch.zeros(80000, 3),
                'occ': torch.zeros(80000, 1)
            }
    
    def _load_view_images(self, view_paths: Dict[str, str]) -> Dict[str, torch.Tensor]:
        view_images = {}
        for view_name, path in view_paths.items():
            if not path or not os.path.exists(path):
                view_images[f'{view_name}_image'] = torch.zeros(3, 518, 518)
                continue
            
            try:
                image = Image.open(path).convert('RGB')
                image_tensor = self.image_transform(image)
                view_images[f'{view_name}_image'] = image_tensor
            except Exception as e:
                print(f"Load image {path} error: {e}")
                view_images[f'{view_name}_image'] = torch.zeros(3, 518, 518)
        
        return view_images
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_id = self.valid_sample_ids[idx]
        meta = self.metadata[idx]
        
        gaussian_data = self._load_gaussian_data(meta)
        
        view_images = self._load_view_images(meta['view_paths'])
        
        combined_data = {
            'sample_id': sample_id,
            **gaussian_data,
            **view_images
        }
        
        return combined_data
    
    def __len__(self) -> int:
        return len(self.valid_sample_ids)