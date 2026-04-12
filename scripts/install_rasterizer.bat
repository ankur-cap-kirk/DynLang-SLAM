@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "PATH=%APPDATA%\Python\Python314\Scripts;%PATH%"
set "NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor"
set "TORCH_CUDA_ARCH_LIST=12.0"
set "DISTUTILS_USE_SDK=1"
echo Environment ready
where cl
where nvcc
where ninja
echo Installing diff-gaussian-rasterization...
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git --no-build-isolation
if %ERRORLEVEL% EQU 0 (
    echo Install succeeded!
    python -c "import diff_gaussian_rasterization; print('SUCCESS: imported!')"
) else (
    echo Install failed with error %ERRORLEVEL%
)
