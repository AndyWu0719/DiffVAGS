#!/usr/bin/env python3
import os
import json
import torch
import warnings
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser

from models.combine_model import CombinedModel
from dataloader.modulation_loader import ModulationLoader
from dataloader.multimodal_loader import MultiModalDataset 
from input_encoder.multimodal_encoder import MultiModalEncoder
from input_encoder.text_encoder import TextEncoder
from input_encoder.image_encoder import ImageEncoder
from dataloader.multimodal_loader import MultiModalDataLoader


def print_config_summary(specs):
    """打印配置摘要"""
    print(f"🚀 开始训练任务: {specs['training_task']}")
    print(f"📊 配置摘要:")
    
    # 🔧 安全访问配置参数
    try:
        print(f"  - VAE潜在维度: {specs['VAESpecs']['encoder']['latent_dim']}")
    except KeyError:
        print(f"  - VAE潜在维度: 配置缺失")
    
    try:
        print(f"  - 多模态输出通道: {specs['MultiModalEncoderSpecs']['fusion']['output_channels']}")
    except KeyError:
        print(f"  - 多模态输出通道: 配置缺失")
    
    try:
        print(f"  - 训练轮数: {specs['TrainingSpecs']['num_epochs']}")
    except KeyError:
        # 兼容旧配置
        print(f"  - 训练轮数: {specs.get('num_epochs', '配置缺失')}")
    
    # VA-VAE配置检查
    if specs.get('VAVAESpecs', {}).get('enable', False):
        try:
            visual_type = specs['VAVAESpecs']['visual_model']['type']
            language_type = specs['VAVAESpecs']['language_model']['type']
            print(f"  - 使用VA-VAE约束: {visual_type} + {language_type}")
        except KeyError:
            print(f"  - VA-VAE配置不完整")


def create_multimodal_dataloader(specs):
    """创建多模态数据加载器"""
    # 🔧 从配置中获取路径，提供默认值
    data_path = specs.get("Text_Image_Data_path", "/home/andywu/Documents/dongjun/LightDiffGS/process_data/step1")
    caption_path = os.path.join(data_path, "texts", "captions.txt")
    image_path = os.path.join(data_path, "images")
    
    # 检查路径是否存在
    if not os.path.exists(caption_path):
        print(f"⚠️  警告: 文本文件不存在: {caption_path}")
        caption_path = "/home/andywu/Documents/dongjun/LightDiffGS/process_data/step1/texts/captions.txt"
    
    if not os.path.exists(image_path):
        print(f"⚠️  警告: 图像目录不存在: {image_path}")
        image_path = "/home/andywu/Documents/dongjun/LightDiffGS/process_data/step1/images"
    
    print(f"📁 数据路径:")
    print(f"  - 文本: {caption_path}")
    print(f"  - 图像: {image_path}")
    
    # 创建数据集
    train_dataset = MultiModalDataset(caption_file=caption_path, image_dir=image_path)
    
    # 🔧 从统一配置获取参数
    try:
        multimodal_specs = specs['MultiModalEncoderSpecs']
        fusion_specs = multimodal_specs['fusion']
        
        output_channels = fusion_specs['output_channels']
        output_features = fusion_specs['output_features'] 
        output_resolution = fusion_specs['output_resolution']
        
    except KeyError as e:
        print(f"❌ 多模态编码器配置缺失: {e}")
        # 🔧 使用兼容的旧配置
        output_channels = specs.get("GaussianModelSpecs1", {}).get("pn_hidden_dim", 128)
        output_features = 512
        output_resolution = 64
        print(f"🔄 使用兼容配置: channels={output_channels}, features={output_features}")
    
    # 创建编码器
    text_encoder = TextEncoder()
    image_encoder = ImageEncoder()
    
    encoder = MultiModalEncoder(
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        output_channels=output_channels,
        output_features=output_features,
        output_resolution=output_resolution,
        freeze_encoders=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fused_dataset = MultiModalDataLoader(train_dataset, encoder, device)
    
    return fused_dataset


def create_trainer(specs, args, callbacks):
    """创建训练器"""
    # 🔧 安全获取训练参数
    try:
        num_epochs = specs['TrainingSpecs']['num_epochs']
    except KeyError:
        num_epochs = specs.get('num_epochs', 200)  # 兼容旧配置
    
    try:
        log_freq = specs['TrainingSpecs']['log_freq'] 
    except KeyError:
        log_freq = specs.get('log_freq', 10)  # 兼容旧配置
    
    print(f"🎯 训练器配置: epochs={num_epochs}, log_freq={log_freq}")
    
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=1, 
        precision='16-mixed',
        max_epochs=num_epochs,
        callbacks=callbacks,
        log_every_n_steps=10,
        default_root_dir=os.path.join("tensorboard_logs", args.exp_dir),
        # 内存优化配置
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        enable_progress_bar=True,
        enable_model_summary=False,
        enable_checkpointing=True,
        num_sanity_val_steps=0,
        limit_train_batches=1.0,
        reload_dataloaders_every_n_epochs=0,
    )
    
    return trainer


def train(specs, args):
    """统一的训练函数"""
    print_config_summary(specs)
    
    # 🔧 根据任务类型选择数据加载器
    if specs['training_task'] == 'diffusion':
        print("📊 使用扩散任务数据加载器")
        train_dataset = ModulationLoader(specs["Data_path"], gs_path=specs.get("gs_path", None))
        fused_dataset = train_dataset
    else:
        print("📊 使用多模态数据加载器")
        fused_dataset = create_multimodal_dataloader(specs)
    
    print("🔄 Loading dataset...")
    
    # 🔧 从配置获取数据加载器参数
    try:
        dataloader_specs = specs.get('DataLoaderSpecs', {})
        batch_size = dataloader_specs.get('batch_size', 1)
        num_workers = dataloader_specs.get('num_workers', 0)
        shuffle = dataloader_specs.get('shuffle', True)
        pin_memory = dataloader_specs.get('pin_memory', False)
    except:
        # 默认配置
        batch_size, num_workers, shuffle, pin_memory = 1, 0, True, False
    
    train_dataloader = DataLoader(
        fused_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=shuffle,
        pin_memory=pin_memory,
        persistent_workers=False,
    )
    
    # 🔧 设置回调 - 兼容新旧配置
    try:
        log_freq = specs['TrainingSpecs']['log_freq']
    except KeyError:
        log_freq = specs.get('log_freq', 50)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.exp_dir, 
        filename='{epoch}', 
        save_top_k=3, 
        save_last=False, 
        every_n_epochs=log_freq,
        monitor='train/total_loss',      # 🔧 修正监控指标
        mode='min',
        auto_insert_metric_name=False,
        save_on_train_epoch_end=True,
        save_weights_only=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks = [checkpoint_callback, lr_monitor]
    
    # 构造模型
    print("🔧 初始化模型...")
    model = CombinedModel(specs)
    
    # Resume 处理逻辑
    resume_ckpt = None
    if args.resume == 'finetune':
        print("🔄 微调模式：加载预训练检查点...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = model.load_from_checkpoint(
                    specs["modulation_ckpt_path"], 
                    specs=specs, 
                    strict=False
                )
                ckpt = torch.load(specs["diffusion_ckpt_path"], map_location="cpu")
                new_state_dict = {k.replace("diffusion_model.", ""): v for k, v in ckpt['state_dict'].items()}
                model.diffusion_model.load_state_dict(new_state_dict)
            resume_ckpt = None
        except Exception as e:
            print(f"⚠️  微调加载失败: {e}")
            
    elif args.resume is not None:
        ckpt_name = "last.ckpt" if args.resume == "last" else f"epoch={args.resume}.ckpt"
        resume_ckpt = os.path.join(args.exp_dir, ckpt_name)
        if os.path.exists(resume_ckpt):
            print(f"🔄 从检查点恢复: {resume_ckpt}")
        else:
            print(f"⚠️  检查点不存在: {resume_ckpt}")
            resume_ckpt = None
    
    # 创建训练器
    trainer = create_trainer(specs, args, callbacks)
    
    print("🚀 开始训练...")
    try:
        trainer.fit(model=model, train_dataloaders=train_dataloader, ckpt_path=resume_ckpt)
    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        raise


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser(description="Training script for LightDiffGS")
    parser.add_argument("--exp_dir", "-e", required=True,
                        help="Path to experiment directory containing specs.json")
    parser.add_argument("--resume", "-r", default=None,
                        help="Resume training: specify epoch, 'last', or 'finetune'")
    parser.add_argument("--batch_size", "-b", type=int, default=None,
                        help="Override batch size (optional)")
    parser.add_argument("--workers", "-w", type=int, default=None,
                        help="Override num workers (optional)")
    
    args = parser.parse_args()
    
    # 🔧 检查配置文件
    specs_path = os.path.join(args.exp_dir, "specs.json")
    if not os.path.exists(specs_path):
        raise FileNotFoundError(f"配置文件不存在: {specs_path}")
    
    with open(specs_path, "r") as f:
        specs = json.load(f)
    
    # 🔧 命令行参数覆盖配置
    if args.batch_size is not None:
        specs.setdefault('DataLoaderSpecs', {})['batch_size'] = args.batch_size
    if args.workers is not None:
        specs.setdefault('DataLoaderSpecs', {})['num_workers'] = args.workers
    
    print("=" * 60)
    print("🎯 LightDiffGS Training")
    print("=" * 60)
    print("Experiment Description:", specs.get("Description", "No Description"))
    print("=" * 60)
    
    warnings.simplefilter("ignore")
    train(specs, args)