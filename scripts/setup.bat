@echo off
REM DynLang-SLAM Setup Script for Windows
REM Run this to set up the entire development environment.

echo ============================================
echo DynLang-SLAM Setup
echo ============================================

REM 1. Install PyTorch with CUDA 12.8 FIRST (must be before rasterizer)
echo.
echo [1/5] Installing PyTorch with CUDA 12.8...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo FAILED: PyTorch CUDA install failed. Check your Python version.
    exit /b 1
)

REM 2. Verify CUDA torch is installed
echo.
echo [1.5/5] Verifying CUDA PyTorch...
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'PyTorch {torch.__version__} with CUDA on {torch.cuda.get_device_name(0)}')"
if errorlevel 1 (
    echo WARNING: CUDA not detected. Continuing but GPU acceleration won't work.
)

REM 3. Install diff-gaussian-rasterization (needs CUDA torch)
echo.
echo [2/5] Installing diff-gaussian-rasterization...
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
if errorlevel 1 (
    echo WARNING: diff-gaussian-rasterization failed. You may need Visual Studio Build Tools.
    echo Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo Install "Desktop development with C++" workload, then re-run this script.
)

REM 4. Install remaining dependencies
echo.
echo [3/5] Installing Python dependencies...
pip install -r requirements.txt

REM 5. Install SAM
echo.
echo [4/5] Installing Segment Anything Model...
pip install git+https://github.com/facebookresearch/segment-anything.git

REM 6. Download SAM checkpoint
echo.
echo [5/5] Downloading SAM ViT-L checkpoint...
if not exist checkpoints mkdir checkpoints
if not exist checkpoints\sam_vit_l_0b3195.pth (
    curl -L -o checkpoints\sam_vit_l_0b3195.pth "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    echo SAM checkpoint downloaded.
) else (
    echo SAM checkpoint already exists.
)

echo.
echo ============================================
echo Setup complete!
echo.
echo Next steps:
echo   1. Download Replica dataset:
echo      python scripts\download_replica.py
echo   2. Run verification:
echo      python scripts\verify_setup.py
echo   3. Start training:
echo      python run.py
echo ============================================
