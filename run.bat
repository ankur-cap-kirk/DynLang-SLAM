@echo off
:: DynLang-SLAM launcher - sets up MSVC + CUDA environment then runs Python
:: Usage: run.bat [args]              -> runs python run.py [args]
::        run.bat visualize.py [args] -> runs python visualize.py [args]
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "TORCH_CUDA_ARCH_LIST=12.0"

:: Check if first arg is a .py file
echo %1 | findstr /i "\.py" >nul
if %errorlevel%==0 (
    python %*
) else (
    python run.py %*
)
