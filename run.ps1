# DynLang-SLAM launcher - sets up MSVC + CUDA environment then runs Python
# Usage: .\run.ps1                          -> runs python run.py
#        .\run.ps1 dataset.max_frames=100   -> runs python run.py dataset.max_frames=100
#        .\run.ps1 visualize.py --frames 60 -> runs python visualize.py --frames 60

# Set up MSVC environment
cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && set' | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}

$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:TORCH_CUDA_ARCH_LIST = "12.0"

# Check if first arg is a .py file
if ($args.Count -gt 0 -and $args[0] -like "*.py") {
    python @args
} else {
    python run.py @args
}
