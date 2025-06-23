#!/usr/bin/env python3

import torch
import torch.utils.data 
from torch.nn import functional as F
import pytorch_lightning as pl
import os
import json, csv
import time
from tqdm.auto import tqdm
from einops import rearrange, reduce
import numpy as np
import warnings
from pathlib import Path

from models.combine_model import CombinedModel
from dataloader.gaussian_loader import GaussianLoader
from dataloader.multiview_loader import MultiViewGaussianLoader
from utils.evaluate_utils import evaluate, pointcloud
from convert import convert
from utils.image_utils import MultiViewImageHandler


class TestDataLoader:
    def __init__(self, data_root, enable_vavae=False, multiview_specs=None):
        if enable_vavae:
            self.dataset = MultiViewGaussianLoader(
                data_root=data_root,
                multiview_specs=multiview_specs,
                enable_multiview=True
            )
        else:
            self.dataset = GaussianLoader(data_root)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


@torch.no_grad()
def test_modulations():
    print("=== Testing Modulations ===")
    
    vavae_specs = specs.get("VAVAESpecs", {})
    enable_vavae = vavae_specs.get("enable", False)
    
    test_dataset = TestDataLoader(
        data_root=specs["Data_path"], 
        enable_vavae=enable_vavae,
        multiview_specs=vavae_specs.get("multiview", {}) if enable_vavae else None
    )
    
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=4)
    
    ckpt_name = f"{args.resume}.ckpt" if args.resume == 'last' else f"epoch={args.resume}.ckpt"
    resume_path = os.path.join(args.exp_dir, ckpt_name)
    
    model = CombinedModel.load_from_checkpoint(resume_path, specs=specs).cuda().eval()
    
    output_suffix = "vavae" if enable_vavae else "standard_vae"
    latent_dir = os.path.join(args.exp_dir, output_suffix, "modulations")
    os.makedirs(latent_dir, exist_ok=True)
    
    if enable_vavae:
        image_handler = MultiViewImageHandler(vavae_specs.get("multiview", {}))
    
    with tqdm(test_dataloader) as pbar:
        for idx, data in enumerate(pbar):
            pbar.set_description(f"Extracting latents: {idx}/{len(test_dataloader)}")
            
            gs = data['gaussians'].cuda()
            plane_features = model.gaussian_model.pointnet.get_plane_features(gs)
            original_features = torch.cat(plane_features, dim=1)
            
            if enable_vavae:
                if data.get('has_multiview_data', False):
                    multiview_images = data['multiview_images'].cuda()
                    out = model.vae_model(original_features, multiview_images)
                    latent = torch.cat([out[6], out[7]], dim=-1)  # z_geo + z_sem
                else:
                    data_path = data.get('data_path', [None])[0]
                    if data_path:
                        multiview_images = image_handler.load_multiview_images(data_path)
                        if multiview_images is not None:
                            multiview_images = multiview_images.unsqueeze(0).cuda()
                            out = model.vae_model(original_features, multiview_images)
                            latent = torch.cat([out[6], out[7]], dim=-1)
                        else:
                            print(f"Skipping sample {idx}: no multiview images")
                            continue
                    else:
                        continue
            else:
                out = model.vae_model(original_features)
                latent = out[4]  # z
            
            outdir = os.path.join(latent_dir, str(idx))
            os.makedirs(outdir, exist_ok=True)
            np.savetxt(os.path.join(outdir, "latent.txt"), latent.cpu().numpy())
    
    print(f"✅ Completed! Saved to {latent_dir}")


def test_generation():
    print("=== Testing Generation (Diffusion) ===")
    
    vavae_specs = specs.get("VAVAESpecs", {})
    enable_vavae = vavae_specs.get("enable", False)
    vavae_suffix = "vavae" if enable_vavae else "standard_vae"
    model_subdir = os.path.join(args.exp_dir, vavae_suffix)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CombinedModel.load_from_checkpoint(specs["modulation_ckpt_path"], specs=specs, strict=False) 
        ckpt = torch.load(specs["diffusion_ckpt_path"])
        new_state_dict = {}
        for k,v in ckpt['state_dict'].items():
            new_key = k.replace("diffusion_model.", "")
            new_state_dict[new_key] = v
        
        model.diffusion_model.load_state_dict(new_state_dict)
        model = model.cuda().eval()

    idx = 0
    for e in range(args.epoches):
        samples = model.diffusion_model.generate_unconditional(args.num_samples)
        plane_features = model.vae_model.decode(samples)
        output_dir = "/media/guest1/WD6TB/WD6TB/Andy/Datasets/lightdiffgsdata/4/0/2/generate"
        os.makedirs(output_dir, exist_ok=True) 
        for i in range(len(plane_features)):
            plane_feature = plane_features[i].unsqueeze(0)
            with torch.no_grad():
                print('create points fast')
                new_pc = pointcloud.create_pc_fast(model.gaussian_model, plane_feature, N=1024, max_batch=2**20, from_plane_features=True)
            new_pc_optimizer = pointcloud.pc_optimizer(model.gaussian_model, plane_feature.detach(), new_pc.clone().detach().cuda())            
            with torch.no_grad():
                new_pc = torch.cat([new_pc, new_pc_optimizer], dim=1)
                new_pc = new_pc.reshape(1, -1, 3).float()
                pred_color, pred_gs = model.gaussian_model.forward_with_plane_features(plane_feature, new_pc)
                gaussian = torch.zeros(new_pc.shape[1], 59).cpu()
                gaussian[:,:3] = new_pc[0]
                gaussian[:,3:51]
                gaussian[:,3:51] = pred_color[0]
                gaussian[:,51] = 2.9444
                gaussian[:,52:55] = 0.9 * torch.log(pred_gs[0,:,0:3])
                gaussian[:,55:59] = pred_gs[0,:,3:7]
                convert(gaussian.detach().cpu().numpy(), f"./generate/gaussian_{idx}.ply")
                idx = idx + 1


def main():
    print("=" * 80)
    print("🧪 LightDiffGS Testing Pipeline")
    print("=" * 80)
    print(f"实验描述: {specs.get('Description', 'No Description')}")
    print(f"训练任务: {specs.get('training_task', 'Unknown')}")
    print(f"数据路径: {specs.get('Data_path', 'Not specified')}")
    print(f"实验目录: {args.exp_dir}")
    print(f"恢复节点: {args.resume}")
    
    vavae_specs = specs.get("VAVAESpecs", {})
    enable_vavae = vavae_specs.get("enable", False)
    print(f"VAVAE 启用状态: {enable_vavae}")
    
    if enable_vavae:
        multiview_specs = vavae_specs.get("multiview", {})
        print(f"多视角视图数: {multiview_specs.get('num_views', 12)}")
        print(f"选择策略: {multiview_specs.get('selection_strategy', 'random')}")
    
    print("=" * 80)
    
    task = specs.get('training_task', 'modulation')
    
    if task == 'modulation':
        print("🔧 Running modulation testing (latent extraction)")
        test_modulations()
        
    elif task == 'combined':
        print("🎨 Running combined testing (generation)")
        test_generation()
        
    elif task == 'diffusion':
        print("🌊 Running diffusion testing (generation only)")
        test_generation()
        
    else:
        raise ValueError(f"Unknown training task: {task}")
    
    print("\n🎉 Testing completed!")


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="LightDiffGS Testing Script")
    
    arg_parser.add_argument(
        "--exp_dir", "-e", required=True,
        help="实验目录，应包含specs.json配置文件"
    )
    
    arg_parser.add_argument(
        "--resume", "-r", default="last", 
        help="要测试的检查点: 整数值、'last'等 (默认: 'last')"
    )
    
    arg_parser.add_argument(
        "--num_samples", "-n", default=5, type=int, 
        help="生成和重建的样本数 (默认: 5)"
    )
    
    arg_parser.add_argument(
        "--epoches", default=100, type=int, 
        help="生成和重建的轮数 (默认: 100)"
    )
    
    arg_parser.add_argument(
        "--filter", default=False, action="store_true",
        help="条件采样时是否过滤"
    )

    args = arg_parser.parse_args()
    
    if not os.path.exists(args.exp_dir):
        raise FileNotFoundError(f"❌ 实验目录不存在: {args.exp_dir}")
    
    specs_file = os.path.join(args.exp_dir, "specs.json")
    if not os.path.exists(specs_file):
        raise FileNotFoundError(f"❌ 配置文件不存在: {specs_file}")
    
    try:
        with open(specs_file, 'r') as f:
            specs = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ 配置文件格式错误: {e}")
    
    warnings.simplefilter("ignore", category=UserWarning)
    warnings.simplefilter("ignore", category=FutureWarning)
    
    main()