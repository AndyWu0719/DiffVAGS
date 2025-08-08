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
from dataloader.gs_dataloader import MultiViewGaussianDataset
from utils.evaluate_utils import evaluate, pointcloud
from convert import convert


@torch.no_grad()
def test_modulations():
    va_specs = specs.get("VisualAlignmentSpecs", {})
    enable_va = va_specs.get("enable", False)
    
    if enable_va:
        print("Visual Alignment mode enabled. Using MultiViewGaussianDataset for testing.")
        data_path = specs["Data_path"]
        test_dataset = MultiViewGaussianDataset(
            gaussian_data_path=data_path["Data_path"],
            image_data_path=data_path["Data_path"],
            cache_path=os.path.join(args.exp_dir, "test_dataset_cache.pkl")
        )
    else:
        print("Standard mode enabled. Using GaussianLoader for testing.")
        test_dataset = GaussianLoader(specs["Data_path"])
    
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=4)

    ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
    model_dir = os.path.join(args.exp_dir, 'visual_alignment' if enable_va == 'modulation' else 'standard_vae')
    resume = os.path.join(model_dir, ckpt)

    print(f"Loading modulation checkpoint from: {resume}")
    model = CombinedModel.load_from_checkpoint(resume, specs=specs).cuda().eval()
    
    output_suffix = "visual_alignment" if enable_va else "standard_vae"
    latent_dir = os.path.join(args.exp_dir, output_suffix, "modulations")
    os.makedirs(latent_dir, exist_ok=True)

    with tqdm(test_dataloader) as pbar:
        for idx, data in enumerate(pbar):
            pbar.set_description("Files evaluated: {}/{}".format(idx, len(test_dataloader)))
            gs = data['gaussians'].cuda()
            plane_features = model.gaussian_model.pointnet.get_plane_features(gs)
            original_features = torch.cat(plane_features, dim=1)
            
            latent = model.vae_model.get_latent(original_features)
            
            outdir = os.path.join(latent_dir, "{}".format(idx))
            os.makedirs(outdir, exist_ok=True)
            np.savetxt(os.path.join(outdir, "latent.txt"), latent.cpu().numpy())
    
    print(f"Completed! Saved to {latent_dir}")


def test_generation():
    print("=== Testing Generation (Diffusion) ===")

    modulation_ckpt_path = specs.get("modulation_ckpt_path")
    diffusion_ckpt_path = specs.get("diffusion_ckpt_path")

    if not modulation_ckpt_path or not diffusion_ckpt_path:
        raise ValueError("specs.json must contain 'modulation_ckpt_path' and 'diffusion_ckpt_path' for generation testing.")
    
    print(f"Loading VAE from: {modulation_ckpt_path}")
    print(f"Loading Diffusion from: {diffusion_ckpt_path}")

    va_specs = specs.get("VisualAlignmentSpecs", {})
    enable_va = va_specs.get("enable", False)
    va_suffix = "va" if enable_va else "standard_vae"
    model_subdir = os.path.join(args.exp_dir, va_suffix)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CombinedModel.load_from_checkpoint(modulation_ckpt_path, specs=specs, strict=False) 
        ckpt = torch.load(diffusion_ckpt_path)
        new_state_dict = {}
        for k,v in ckpt['state_dict'].items():
            new_key = k.replace("diffusion_model.", "")
            new_state_dict[new_key] = v
        
        model.diffusion_model.load_state_dict(new_state_dict)
        model = model.cuda().eval()

    output_dir = os.path.join(args.exp_dir, "generated_samples")
    os.makedirs(output_dir, exist_ok=True) 
    print(f"Generated samples will be saved to: {output_dir}")

    idx = 0
    for e in range(args.epoches):
        samples = model.diffusion_model.generate_unconditional(args.num_samples)
        plane_features = model.vae_model.decode(samples)
        
        for i in range(len(plane_features)):
            plane_feature = plane_features[i].unsqueeze(0)
            with torch.no_grad():
                print(f'Generating sample {idx+1}...')
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
                output_ply_path = os.path.join(output_dir, f"gaussian_{idx}.ply")
                convert(gaussian.detach().cpu().numpy(), output_ply_path)
                idx = idx + 1


def main():
    print("=" * 80)
    print("🧪 DiffVAGS Testing Pipeline")
    print("=" * 80)
    print(f"Experiment description: {specs.get('Description', 'No Description')}")
    print(f"Training task: {specs.get('training_task', 'Unknown')}")
    print(f"Data path: {specs.get('Data_path', 'Not specified')}")
    print(f"Experiment directory: {args.exp_dir}")
    print(f"Resume checkpoint: {args.resume}")

    va_specs = specs.get("VisualAlignmentSpecs", {})
    enable_va = va_specs.get("enable", False)
    print(f"Visual Alignment status: {enable_va}")
    
    if enable_va:
        multiview_specs = va_specs
        print(f"Number of views: {multiview_specs.get('num_views', 'N/A')}")

    print("=" * 80)
    
    task = specs.get('training_task', 'modulation')
    
    if task == 'modulation':
        print("Running modulation testing (latent extraction)")
        test_modulations()
        
    elif task == 'combined':
        print("Running combined testing (generation)")
        test_generation()
        
    elif task == 'diffusion':
        print("Running diffusion testing (generation only)")
        test_generation()
        
    else:
        raise ValueError(f"Unknown training task: {task}")
    
    print("\nTesting completed!")


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="DiffVAGS Testing Script")
    
    arg_parser.add_argument(
        "--exp_dir", "-e", required=True,
        help="Experiment directory containing specs.json and checkpoints (required)"
    )
    
    arg_parser.add_argument(
        "--resume", "-r", default="last", 
        help="Checkpoint to resume from: integer value, 'last', etc. (default: 'last')"
    )
    
    arg_parser.add_argument(
        "--num_samples", "-n", default=5, type=int, 
        help="Number of samples to generate and reconstruct (default: 5)"
    )
    
    arg_parser.add_argument(
        "--epoches", default=100, type=int, 
        help="Number of epochs to run for generation and reconstruction testing (default: 100)"
    )
    
    arg_parser.add_argument(
        "--filter", default=False, action="store_true",
        help="Whether to filter during conditional sampling"
    )

    args = arg_parser.parse_args()
    
    if not os.path.exists(args.exp_dir):
        raise FileNotFoundError(f"No experiment path: {args.exp_dir}")
    
    specs_file = os.path.join(args.exp_dir, "specs.json")
    if not os.path.exists(specs_file):
        raise FileNotFoundError(f"No specs file: {specs_file}")

    try:
        with open(specs_file, 'r') as f:
            specs = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Config file format error: {e}")
    
    warnings.simplefilter("ignore", category=UserWarning)
    warnings.simplefilter("ignore", category=FutureWarning)
    
    main()