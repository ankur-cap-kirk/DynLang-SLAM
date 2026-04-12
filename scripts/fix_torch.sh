#!/bin/bash
#SBATCH -p public
#SBATCH -q class
#SBATCH --gres=gpu:1
#SBATCH -t 0-0:30:00
#SBATCH -n 4
#SBATCH --mem=20G
#SBATCH -o /scratch/agurupr6/DynLang-SLAM/results/fix_torch_%j.out
#SBATCH -e /scratch/agurupr6/DynLang-SLAM/results/fix_torch_%j.err

module load cuda-12.4.1-gcc-12.1.0
eval "$(conda shell.bash hook)"
conda activate dynlang

echo "=== Reinstalling torch 2.6.0+cu124 ==="
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall

echo "=== Reinstalling gsplat (compatible with torch 2.6) ==="
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
pip install gsplat==1.5.0 --no-deps --no-cache-dir

echo "=== Verifying ==="
python -c "
import torch
print(f'torch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')
import gsplat
print('gsplat OK')
import torchvision
print(f'torchvision: {torchvision.__version__}')
"
