#!/bin/bash
#SBATCH -p public
#SBATCH -q class
#SBATCH --gres=gpu:1
#SBATCH -t 0-1:00:00
#SBATCH -n 4
#SBATCH --mem=40G
#SBATCH -o /scratch/agurupr6/DynLang-SLAM/results/fix_gsplat_%j.out
#SBATCH -e /scratch/agurupr6/DynLang-SLAM/results/fix_gsplat_%j.err

module load cuda-12.4.1-gcc-12.1.0
eval "$(conda shell.bash hook)"
conda activate dynlang

echo "=== Step 1: Downgrade to torch 2.4 + cu124 ==="
pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124 --index-url https://download.pytorch.org/whl/cu124

echo "=== Step 2: Install prebuilt gsplat for pt24cu124 ==="
pip install gsplat==1.4.0+pt24cu124 --no-deps --index-url https://docs.gsplat.studio/whl

echo "=== Step 3: Verify everything ==="
python -c "
import torch
print(f'torch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')
import torchvision
print(f'torchvision: {torchvision.__version__}')
import gsplat
print(f'gsplat: {gsplat.__version__}')
from gsplat import rasterization
means = torch.randn(100, 3, device='cuda')
quats = torch.randn(100, 4, device='cuda')
scales = torch.randn(100, 3, device='cuda')
opacities = torch.sigmoid(torch.randn(100, device='cuda'))
colors = torch.randn(100, 3, device='cuda')
viewmats = torch.eye(4, device='cuda').unsqueeze(0)
Ks = torch.tensor([[600,0,300],[0,600,340],[0,0,1]], dtype=torch.float32, device='cuda').unsqueeze(0)
out = rasterization(means=means, quats=quats, scales=scales, opacities=opacities, colors=colors, viewmats=viewmats, Ks=Ks, width=640, height=480, packed=True)
print(f'Render OK: {out[0].shape}')
print('ALL GOOD')
"

echo "=== Step 4: Run language SLAM test ==="
cd /scratch/agurupr6/DynLang-SLAM
python scripts/test_language_slam.py
