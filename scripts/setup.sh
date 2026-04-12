#!/bin/bash
# DynLang-SLAM Setup Script
# Run this to set up the entire development environment.

set -e

echo "============================================"
echo "DynLang-SLAM Setup"
echo "============================================"

# 1. Install PyTorch with CUDA 12.8 (Blackwell/RTX 5070 support)
echo ""
echo "[1/5] Installing PyTorch with CUDA 12.8..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. Install diff-gaussian-rasterization
echo ""
echo "[2/5] Installing diff-gaussian-rasterization..."
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git

# 3. Install remaining dependencies
echo ""
echo "[3/5] Installing Python dependencies..."
pip install -r requirements.txt

# 4. Install SAM
echo ""
echo "[4/5] Installing Segment Anything Model..."
pip install git+https://github.com/facebookresearch/segment-anything.git

# 5. Download SAM checkpoint
echo ""
echo "[5/5] Downloading SAM ViT-L checkpoint..."
mkdir -p checkpoints
if [ ! -f checkpoints/sam_vit_l_0b3195.pth ]; then
    curl -L -o checkpoints/sam_vit_l_0b3195.pth \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    echo "SAM checkpoint downloaded."
else
    echo "SAM checkpoint already exists."
fi

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Download Replica dataset:"
echo "     python scripts/download_replica.py"
echo "  2. Run verification:"
echo "     python scripts/verify_setup.py"
echo "  3. Start training:"
echo "     python run.py"
echo "============================================"
