## Set permanent environment variables for DynLang-SLAM

# CUDA_HOME
[System.Environment]::SetEnvironmentVariable('CUDA_HOME', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8', 'User')
Write-Host "Set CUDA_HOME = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

# TORCH_CUDA_ARCH_LIST
[System.Environment]::SetEnvironmentVariable('TORCH_CUDA_ARCH_LIST', '12.0', 'User')
Write-Host "Set TORCH_CUDA_ARCH_LIST = 12.0"

# Add Python Scripts (ninja) to PATH
$scriptsDir = "C:\Users\ankur\AppData\Roaming\Python\Python314\Scripts"
$currentPath = [System.Environment]::GetEnvironmentVariable('Path', 'User')
if ($currentPath -notlike '*Python314\Scripts*') {
    [System.Environment]::SetEnvironmentVariable('Path', "$currentPath;$scriptsDir", 'User')
    Write-Host "Added $scriptsDir to PATH"
} else {
    Write-Host "Python314\Scripts already in PATH"
}

Write-Host "`nDone! Close and reopen your terminal for changes to take effect."
