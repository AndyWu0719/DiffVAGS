#!/bin/bash
#SBATCH --job-name=g2fnet_train          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=4      # number of tasks per node (adjust when using MPI)
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks, adjust when using OMP)
#SBATCH --time=72:00:00          # total run time limit (D-HH:MM:SS)
#SBATCH --gres=gpu:4             
#SBATCH --output=slurm_logs_%x_%j.out
#SBATCH --error=slurm_logs_%A_%a.err
#SBATCH --partition=gpu-a30      # partition(queue) where you submit (amd/intel/gpu-a30/gpu-l20)
#SBATCH --account=owlicdc        # slurm account name
#SBATCH --mem=48G                # memory per node (adjust based on your job requirements)

echo "Setting up environment..."
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
conda activate diffvags

echo "Starting training script..."
torchrun --nproc_per_node=$SLURM_NTASKS_PER_NODE /home/dwubf/workplace/DiffVAGS/models/visual_alignment/g2f_net/train.py \
    --gaussian_path /project/owlicdc/andy/dataset/03001627/media/andywu/WD6TB/WD6TB/Andy/Datasets/lightdiffgsdata/03001627/convert_data \
    --image_path /project/owlicdc/andy/dataset/03001627/media/andywu/WD6TB/WD6TB/Andy/Datasets/lightdiffgsdata/03001627/training_data \
    --batch_size 4 \
    --num_workers 8 \
    --epochs 1000 \
    --lr 1e-4

echo "Job finished."