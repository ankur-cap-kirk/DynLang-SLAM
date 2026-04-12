#!/bin/bash
#SBATCH -p public
#SBATCH -q class
#SBATCH --gres=gpu:1
#SBATCH -t 0-0:30:00
#SBATCH -n 4
#SBATCH --mem=20G
#SBATCH -o /scratch/agurupr6/DynLang-SLAM/results/install_gsplat_%j.out
#SBATCH -e /scratch/agurupr6/DynLang-SLAM/results/install_gsplat_%j.err

module load cuda-12.4.1-gcc-12.1.0
source activate dynlang

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
echo "CUDA_HOME=$CUDA_HOME"
nvcc --version

echo "=== Installing gsplat ==="
pip install gsplat --force-reinstall --no-cache-dir

echo "=== Testing gsplat ==="
python -c "import gsplat; print('gsplat OK')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
