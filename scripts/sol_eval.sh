#!/bin/bash
#SBATCH --job-name=dynlang-eval
#SBATCH --output=results/sol_%j.out
#SBATCH --error=results/sol_%j.err
#SBATCH --partition=public
#SBATCH --qos=grp_sjayasur
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=agurupr6@asu.edu

# ============================================================
# DynLang-SLAM Evaluation on Sol
# Usage: sbatch scripts/sol_eval.sh
#        sbatch scripts/sol_eval.sh room0 room1 room2
# ============================================================

echo "=== Job Start: $(date) ==="
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"

# Load GCC 12 first, then CUDA (order matters for JIT compilation)
module load gcc-12.1.0-gcc-11.2.0
module load cuda-12.4.1-gcc-12.1.0

# Set CUDA arch for Sol GPUs (A30=8.0, A100=8.0, H100=9.0)
export TORCH_CUDA_ARCH_LIST="8.0"

# Ensure the GCC from the module is used
echo "GCC version: $(gcc --version | head -1)"
export CC=$(which gcc)
export CXX=$(which g++)

# Verify nvcc is available
echo "CUDA: $(nvcc --version 2>&1 | grep release)"

# Activate environment
source activate dynlang

# Move to project directory
cd /scratch/agurupr6/DynLang-SLAM

# Parse scenes from arguments, default to room0 room1 room2
SCENES="${@:-room0}"
FRAMES=2000

# Force unbuffered output so we can see progress
export PYTHONUNBUFFERED=1

echo "Scenes: $SCENES | Frames: $FRAMES"

# Print GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Force gsplat to recompile CUDA kernels
python -c "from gsplat import rasterization; print('gsplat CUDA OK')"

# Create results dir
mkdir -p results

# Run evaluation
python evaluate.py --scenes $SCENES --frames $FRAMES

echo ""
echo "=== Job End: $(date) ==="
