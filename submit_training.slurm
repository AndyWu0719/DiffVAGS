#!/bin/bash

# --- SLURM 资源申请 ---
#SBATCH --job-name=diffvags_train   # 作业名称，方便你查看
#SBATCH --partition=gpu-a30         # 申请使用 GPU 分区 (具体名称请咨询你的HPC管理员)
#SBATCH --gres=gpu:4                # 申请 4 张 GPU
#SBATCH --nodes=1                   # 在 1 个节点上申请
#SBATCH --ntasks-per-node=4         # 每个节点上运行 4 个任务 (通常与GPU数量一致)
#SBATCH --cpus-per-task=8           # 为每个任务(GPU)分配 8 个 CPU 核心
#SBATCH --mem=64G                   # 申请总内存
#SBATCH --time=72:00:00             # 预计运行时间 (2天)
#SBATCH --output=slurm_logs/%x_%j.out # 将标准输出重定向到文件

# --- 软件环境设置 ---
echo "Setting up environment..."
# 加载你需要的模块，例如 anaconda, cuda, cudnn (具体命令请咨询HPC管理员)
module load anaconda/2023a
module load cuda/11.8

# 激活你的 conda 环境
source activate your_conda_env_name  # <-- 替换成你的 conda 环境名

# --- 运行训练脚本 ---
echo "Starting training script..."
# srun 会自动处理分布式环境的设置
# PyTorch Lightning 会自动识别 srun 设置的环境变量
srun python train.py \
    --exp_dir /path/to/your/experiments/stage1_modulation \
    --batch_size 4 \
    --workers 8
    # 注意：这里的 batch_size 是每个 GPU 的批次大小。
    # 如果你设置了 batch_size=4 并使用了4张卡，那么你的全局批次大小 (global batch size) 就是 4 * 4 = 16。

echo "Job finished."