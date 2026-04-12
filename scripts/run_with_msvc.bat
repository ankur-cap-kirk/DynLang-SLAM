@echo off
REM Sets up MSVC environment and runs a Python script with CUDA compilation support.
REM Usage: scripts\run_with_msvc.bat <python_script> [args...]

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

REM Add ninja to PATH
set PATH=%APPDATA%\Python\Python314\Scripts;%PATH%

REM Run the Python script with all arguments
python %*
