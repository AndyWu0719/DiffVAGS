#!/usr/bin/env python3

import torch
import torch.utils.data 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os
import json
import warnings

from models.combine_model import CombinedModel
from dataloader.modulation_loader import ModulationLoader
from dataloader.gaussian_loader import GaussianLoader
from dataloader.multiview_loader import MultiViewGaussianLoader
from utils.diff_utils import save_code_to_conf

def train():
    
    task = specs['training_task']
    data_path = specs["Data_path"]
    
    if task == 'diffusion':
        train_dataset = ModulationLoader(data_path, gs_path=specs.get("gs_path", None))
        enable_vavae = specs.get("enable_vavae", False)
    else:
        gaussian_specs = specs.get("GaussianModelSpecs", {})
        enable_vavae = gaussian_specs.get("enable_vavae", False)
        
        if enable_vavae:
            multiview_specs = gaussian_specs.get("multiview", {})
            train_dataset = MultiViewGaussianLoader(
                data_root=data_path,
                multiview_specs=multiview_specs,
                enable_multiview=True
            )
        else:
            train_dataset = GaussianLoader(data_path)
    
    print(f"Dataset size: {len(train_dataset)}")
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, 
        num_workers=args.workers,
        drop_last=True, 
        shuffle=True, 
        pin_memory=True, 
        persistent_workers=True if args.workers > 0 else False
    )

    model_type = "vavae" if enable_vavae else "standard_vae"
    save_dir = os.path.join(args.exp_dir, model_type)
    os.makedirs(save_dir, exist_ok=True)

    save_code_to_conf(save_dir) 
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir, 
        filename='{epoch}', 
        save_top_k=-1, 
        save_last=True, 
        every_n_epochs=specs.get("log_freq", 100)
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback, lr_monitor]

    model = CombinedModel(specs)
    
    resume = None
    if args.resume == 'finetune':
        print("Fine-tuning mode: loading separate checkpoints...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if "modulation_ckpt_path" in specs:
                model = model.load_from_checkpoint(
                    specs["modulation_ckpt_path"], 
                    specs=specs, 
                    strict=False
                )
                print(f"Loaded modulation checkpoint: {specs['modulation_ckpt_path']}")
            
            if "diffusion_ckpt_path" in specs:
                ckpt = torch.load(specs["diffusion_ckpt_path"])
                new_state_dict = {}
                for k, v in ckpt['state_dict'].items():
                    new_key = k.replace("diffusion_model.", "")
                    new_state_dict[new_key] = v
                model.diffusion_model.load_state_dict(new_state_dict)
                print(f"Loaded diffusion checkpoint: {specs['diffusion_ckpt_path']}")
                
    elif args.resume is not None:
        ckpt_name = "last.ckpt" if args.resume == 'last' else f"epoch={args.resume}.ckpt"
        resume = os.path.join(save_dir, ckpt_name)
        if os.path.exists(resume):
            print(f"Resuming from: {resume}")
        else:
            print(f"Checkpoint not found: {resume}")
            resume = None

    print("Starting training...")
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=-1, 
        precision=32,
        max_epochs=specs["num_epochs"], 
        callbacks=callbacks, 
        log_every_n_steps=1,
        default_root_dir=os.path.join("tensorboard_logs", os.path.basename(args.exp_dir), model_type)
    )
    
    trainer.fit(model=model, train_dataloaders=train_dataloader, ckpt_path=resume)
    print("Training completed!")

if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="LightDiffGS Training")
    arg_parser.add_argument(
        "--exp_dir", "-e", required=True,
        help="Experiment dir, include specs.json"
    )
    arg_parser.add_argument(
        "--resume", "-r", default=None,
        help="Resume training: Int, 'last' or 'finetune'"
    )
    arg_parser.add_argument("--batch_size", "-b", default=1, type=int)
    arg_parser.add_argument("--workers", "-w", default=1, type=int)

    args = arg_parser.parse_args()
    
    specs_file = os.path.join(args.exp_dir, "specs.json")
    if not os.path.exists(specs_file):
        raise FileNotFoundError(f"No config file: {specs_file}")
        
    with open(specs_file, 'r') as f:
        specs = json.load(f)
    
    print("=" * 50)
    print(f"{specs.get('Description', 'LightDiffGS Training')}")
    print(f"Task: {specs.get('training_task', 'unknown')}")
    print(f"Data: {specs.get('Data_path', 'not specified')}")
    print(f"Experiment dir: {args.exp_dir}")
    print("=" * 50)
    
    warnings.simplefilter("ignore", category=UserWarning)
    warnings.simplefilter("ignore", category=FutureWarning)
    
    train()