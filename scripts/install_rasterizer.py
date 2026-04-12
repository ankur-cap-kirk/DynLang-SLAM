"""Install diff-gaussian-rasterization with proper MSVC + CUDA environment."""
import subprocess
import sys
import os

CMD_EXE = r"C:\WINDOWS\system32\cmd.exe"
APPDATA = os.environ.get("APPDATA", "")

# Write a temporary batch file to avoid quoting issues
bat_content = f"""@echo off
call "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat" >nul 2>&1
set "PATH={APPDATA}\\Python\\Python314\\Scripts;%PATH%"
set "NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor"
set "TORCH_CUDA_ARCH_LIST=12.0"
set "DISTUTILS_USE_SDK=1"
echo === MSVC: cl.exe ===
where cl
echo === CUDA: nvcc ===
nvcc --version 2>&1 | findstr /C:"release"
echo === Ninja ===
ninja --version
echo === Installing ===
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git --no-build-isolation
echo === Exit code: %ERRORLEVEL% ===
"""

bat_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_install.bat")
with open(bat_path, "w") as f:
    f.write(bat_content)

print("Running installation with MSVC environment...")
print("This may take 5-10 minutes for CUDA kernel compilation.\n")

result = subprocess.run(
    [CMD_EXE, "/c", bat_path],
    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    timeout=600,
)

print(f"\nReturn code: {result.returncode}")

# Cleanup
try:
    os.remove(bat_path)
except:
    pass
