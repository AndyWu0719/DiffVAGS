#!/bin/bash
#SBATCH --job-name=diffvags_stage2_diffusion          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=4      # number of tasks per node (adjust when using MPI)
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks, adjust when using OMP)
#SBATCH --time=72:00:00          # total run time limit (D-HH:MM:SS)
#SBATCH --gres=gpu:4                # 申请 4 张 GPU
#SBATCH --output=slurm_logs_%x_%j.out # 将标准输出重定向到文件
#SBATCH --error=slurm_logs_%A_%a.err
#SBATCH --partition=gpu-a30        # partition(queue) where you submit (amd/intel/gpu-a30/gpu-l20)
#SBATCH --account=owlicdc        # slurm account name
#SBATCH --mem=48G                # memory per node (adjust based on your job requirements)

# --- 软件环境设置 ---
echo "Setting up environment..."
# 加载你需要的模块，例如 anaconda, cuda, cudnn (具体命令请咨询HPC管理员)
module load anaconda3
module load cuda/12.1.0-a4oqlhx

__conda_setup="$('/opt/shared/spack/local/linux-rocky9-x86_64_v4/gcc-11.4.1/anaconda3-2023.09-0-biybti3xwtszqc65dlpamjnvrcfyrd7o/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/shared/spack/local/linux-rocky9-x86_64_v4/gcc-11.4.1/anaconda3-2023.09-0-biybti3xwtszqc65dlpamjnvrcfyrd7o/etc/profile.d/conda.sh" ]; then
        . "/opt/shared/spack/local/linux-rocky9-x86_64_v4/gcc-11.4.1/anaconda3-2023.09-0-biybti3xwtszqc65dlpamjnvrcfyrd7o/etc/profile.d/conda.sh"
    else
        export PATH="/opt/shared/spack/local/linux-rocky9-x86_64_v4/gcc-11.4.1/anaconda3-2023.09-0-biybti3xwtszqc65dlpamjnvrcfyrd7o/bin:$PATH"
    fi
fi
unset __conda_setup
# 激活你的 conda 环境
conda activate diffvags  # <-- 替换成你的 conda 环境名

# --- 运行训练脚本 ---
echo "Starting training script..."
# srun 会自动处理分布式环境的设置
# PyTorch Lightning 会自动识别 srun 设置的环境变量
python /home/dwubf/workplace/DiffVAGS/train.py \
    --exp_dir /home/dwubf/workplace/DiffVAGS/experiments/stage2_diffusion \
    --batch_size 4 \
    --workers 8

echo "Job finished."