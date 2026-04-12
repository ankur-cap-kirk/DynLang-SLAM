#!/bin/bash
# ============================================================
# One-time setup on Sol supercomputer
# Run from Sol terminal (Jupyter Lab > File > New > Terminal)
#   cd /scratch/agurupr6/DynLang-SLAM
#   bash scripts/sol_setup.sh
# ============================================================

set -e

echo "=== DynLang-SLAM Sol Setup ==="

# Create conda environment
conda create -n dynlang python=3.11 -y
source activate dynlang

# Install PyTorch with CUDA (Sol has A100s/H100s)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install gsplat
pip install gsplat

# Install remaining dependencies
pip install -r requirements.txt

echo ""
echo "=== Setup Complete ==="
echo "Submit eval job with: sbatch scripts/sol_eval.sh room0 room1 room2"
