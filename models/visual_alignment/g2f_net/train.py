#!/usr/bin/env python3
import argparse
import torch
import torch.distributed as dist
import os
from g2fnet import train_model

def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def main():
    parser = argparse.ArgumentParser(description='G2FNet Training Script')
    parser.add_argument('--gaussian_path', type=str, required=False, help='Path to the Gaussian data', default='/media/andywu/WD6TB/WD6TB/Andy/Datasets/lightdiffgsdata/03001627/convert_data')
    parser.add_argument('--image_path', type=str, required=False, help='Path to the image data', default='/media/andywu/WD6TB/WD6TB/Andy/Datasets/lightdiffgsdata/03001627/training_data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Resume training from the specified best_model.pth file')

    args = parser.parse_args()
    
    local_rank = setup_distributed()
    
    if local_rank == 0:
        print(f"Start distributed training: Gaussian data={args.gaussian_path}, Image data={args.image_path}")
        print(f"Parameters: epochs={args.epochs}, batch_size(per GPU)={args.batch_size}, lr={args.lr}")
        if args.resume_checkpoint:
            print(f"Resume training from checkpoint: {args.resume_checkpoint}")

    train_model(
        local_rank=local_rank,
        gaussian_path=args.gaussian_path,
        image_path=args.image_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        resume_checkpoint=args.resume_checkpoint
    )

    dist.destroy_process_group()

if __name__ == "__main__":
    main()