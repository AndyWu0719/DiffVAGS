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
        model_type = "diffusion"
    else:
        va_specs = specs.get("VisualAlignmentSpecs", {})
        enable_visual_alignment = va_specs.get("enable", False)
        
        if enable_visual_alignment:
            print("Visual Alignment mode enabled. Using MultiViewGaussianLoader.")
            train_dataset = MultiViewGaussianLoader(
                data_root=data_path,
                multiview_specs=va_specs # 直接传递整个VA配置节
            )
            model_type = "visual_alignment"
        else:
            print("Standard mode enabled. Using GaussianLoader.")
            train_dataset = GaussianLoader(data_path)
            model_type = "standard_vae"
    
    print(f"Dataset size: {len(train_dataset)}")

    trainer_specs = specs.get("TrainerSpecs", {})

    batch_size = args.batch_size if args.batch_size is not None else trainer_specs.get("batch_size", 1)
    num_workers = args.workers if args.workers is not None else trainer_specs.get("num_workers", 1)
    precision = args.precision if args.precision is not None else trainer_specs.get("precision", 32)

    print(f"Effective training params: batch_size={batch_size}, workers={num_workers}, precision={precision}")
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, 
        num_workers=num_workers,
        drop_last=True, 
        shuffle=True, 
        pin_memory=True, 
        persistent_workers=True if num_workers > 0 else False
    )

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
        strategy='ddp',
        precision=precision,
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
    arg_parser.add_argument("--batch_size", "-b", default=None, type=int, help="Override batch_size in specs.json")
    arg_parser.add_argument("--workers", "-w", default=None, type=int, help="Override num_workers in specs.json")
    arg_parser.add_argument("--precision", "-p", default=None, type=int, help="Override precision in specs.json")

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