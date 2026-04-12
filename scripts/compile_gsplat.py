"""Compile gsplat CUDA kernels using CUDA 12.8 (matching PyTorch)."""
import subprocess
import os

CMD_EXE = r"C:\WINDOWS\system32\cmd.exe"
APPDATA = os.environ.get("APPDATA", "")
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

bat_content = f"""@echo off
call "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat" >nul 2>&1
set "PATH={APPDATA}\\Python\\Python314\\Scripts;%PATH%"
set "CUDA_HOME=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8"
set "TORCH_CUDA_ARCH_LIST=12.0"
echo === Tools ===
echo CUDA_HOME: %CUDA_HOME%
"%CUDA_HOME%\\bin\\nvcc.exe" --version 2>&1 | findstr /C:"release"
where cl 2>nul | findstr /C:"cl.exe"
ninja --version
echo === Compiling gsplat (this takes 3-5 min first time) ===
python -c "import os; os.environ['CUDA_HOME']=r'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8'; os.environ['TORCH_CUDA_ARCH_LIST']='12.0'; from gsplat.cuda._backend import _C; print('SUCCESS: gsplat CUDA compiled!')"
"""

bat_path = os.path.join(PROJECT_DIR, "scripts", "_tmp_compile.bat")
with open(bat_path, "w") as f:
    f.write(bat_content)

print("Compiling gsplat with CUDA 12.8 + MSVC...")
print("First-time compilation takes 3-5 minutes.\n")

result = subprocess.run([CMD_EXE, "/c", bat_path], cwd=PROJECT_DIR, timeout=600)
print(f"\nReturn code: {result.returncode}")

try:
    os.remove(bat_path)
except:
    pass
