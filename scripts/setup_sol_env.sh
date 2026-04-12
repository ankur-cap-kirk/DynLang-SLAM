#!/bin/bash
#SBATCH -p public
#SBATCH -q class
#SBATCH --gres=gpu:1
#SBATCH -t 0-2:00:00
#SBATCH -n 8
#SBATCH --mem=40G
#SBATCH -o /scratch/agurupr6/DynLang-SLAM/results/setup_env_%j.out
#SBATCH -e /scratch/agurupr6/DynLang-SLAM/results/setup_env_%j.err

eval "$(conda shell.bash hook)"

echo "=== Step 0: System info ==="
ldd --version | head -1
cat /etc/redhat-release 2>/dev/null || cat /etc/os-release 2>/dev/null | head -3

echo "=== Step 1: Env with conda GCC (no sysroot pin) ==="
conda remove -n dynlang5 --all -y 2>/dev/null
conda create -n dynlang5 python=3.11 cmake ninja -c conda-forge -y
conda activate dynlang5
conda install -n dynlang5 gcc_linux-64 gxx_linux-64 -c conda-forge -y

echo "=== Step 2: Verify conda GCC ==="
ls $CONDA_PREFIX/bin/*gcc* 2>/dev/null
export CC=$(ls $CONDA_PREFIX/bin/x86_64-conda*-gcc 2>/dev/null | head -1)
export CXX=$(ls $CONDA_PREFIX/bin/x86_64-conda*-g++ 2>/dev/null | head -1)
echo "CC=$CC"
echo "CXX=$CXX"
$CC --version 2>/dev/null | head -1 || echo "CC not found"

echo "=== Step 3: torch 2.6 + cu124 ==="
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124

echo "=== Step 4: CUDA setup ==="
module load cuda-12.4.1-gcc-12.1.0
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=8
echo "CUDA_HOME=$CUDA_HOME"
nvcc --version | tail -1

echo "=== Step 5: Test compilation ==="
python -c "
import torch
from torch.utils.cpp_extension import load
import tempfile, os
# Write minimal CUDA test
src = '''
#include <torch/extension.h>
torch::Tensor test_fn(torch::Tensor x) { return x + 1; }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def(\"test_fn\", &test_fn); }
'''
tmpdir = tempfile.mkdtemp()
with open(os.path.join(tmpdir, 'test.cpp'), 'w') as f: f.write(src)
try:
    mod = load('test_ext', [os.path.join(tmpdir, 'test.cpp')], verbose=True)
    print(f'COMPILATION TEST: OK - {mod.test_fn(torch.tensor([1.0]))}')
except Exception as e:
    print(f'COMPILATION TEST: FAILED - {e}')
"

echo "=== Step 6: Build gsplat from source ==="
cd /scratch/agurupr6
rm -rf gsplat_build
git clone --depth 1 --branch v1.4.0 https://github.com/nerfstudio-project/gsplat.git gsplat_build
cd gsplat_build
git submodule update --init --recursive

# Make sure glm headers are found
export CPATH=$(pwd)/gsplat/cuda/csrc/third_party/glm:$CONDA_PREFIX/include:$CPATH
export CPLUS_INCLUDE_PATH=$(pwd)/gsplat/cuda/csrc/third_party/glm:$CPLUS_INCLUDE_PATH

pip install . --no-build-isolation --no-deps -v 2>&1 | tail -80
echo "gsplat install exit: $?"

echo "=== Step 7: Install remaining deps ==="
cd /scratch/agurupr6/DynLang-SLAM
pip install jaxtyping numpy scipy matplotlib pillow opencv-python-headless omegaconf PyYAML tqdm
pip install sam2 --no-deps
pip install hydra-core iopath
pip install open_clip_torch

echo "=== Step 8: Full test ==="
python -c "
import torch
print(f'torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')
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
print(f'Render: {out[0].shape}')
print('=== ALL GOOD ===')
"

echo "=== Step 9: Run SLAM test ==="
python scripts/test_language_slam.py
