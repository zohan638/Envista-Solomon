param(
    [string]$Python = "python",
    [string]$OutputName = "EnvistaSolomon",
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu126",
    [string]$TorchVersion = "2.7.1",
    [string]$TorchVisionVersion = "0.22.1",
    [string]$TorchAudioVersion = "2.7.1",
    [string]$DetectronFindLinks = "https://miropsota.github.io/torch_packages_builder/detectron2/",
    [string]$DetectronWheel = "detectron2==0.6+18f6958pt2.7.1cu126",
    [switch]$OneFile
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root

try {
    Write-Host "Building $OutputName with $Python"

    & $Python -m pip install --upgrade pip
    & $Python -m pip install pyinstaller pyqt5 numpy opencv-python pyserial
    $torchArgs = @("torch==$TorchVersion", "torchvision==$TorchVisionVersion", "torchaudio==$TorchAudioVersion")
    if ($TorchIndex) { $torchArgs += @("--index-url", $TorchIndex) }
    & $Python -m pip install @torchArgs

    $detectronArgs = @()
    if ($DetectronFindLinks) { $detectronArgs += @("--find-links", $DetectronFindLinks) }
    if ($DetectronWheel) { $detectronArgs += $DetectronWheel }
    if ($detectronArgs.Count -gt 0) {
        & $Python -m pip install @detectronArgs
    } else {
        Write-Warning "Detectron2 wheel not provided; assumes detectron2 already installed."
    }

    $pyiArgs = @(
        "--noconfirm",
        "--clean",
        "--name", $OutputName,
        "--windowed",
        "--collect-all", "detectron2",
        "--collect-all", "torch",
        "--collect-all", "cv2",
        "main.py"
    )
    if ($OneFile) { $pyiArgs += "--onefile" }

    & $Python -m PyInstaller @pyiArgs

    $distDir = Join-Path -Path $root -ChildPath ("dist/" + $OutputName)
    $zipPath = Join-Path -Path $root -ChildPath ("dist/" + $OutputName + ".zip")

    # Remove hydra test assets that can be locked on Windows and are not needed at runtime.
    $hydraTests = Join-Path -Path $distDir -ChildPath "_internal/hydra/test_utils"
    if (Test-Path $hydraTests) {
        try {
            Remove-Item -Path $hydraTests -Recurse -Force -ErrorAction SilentlyContinue
        } catch {
            Write-Warning "Could not remove hydra test_utils: $_"
        }
    }

    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    if (Test-Path $distDir) {
        Compress-Archive -Path (Join-Path $distDir "*") -DestinationPath $zipPath -Force
        Write-Host "Packaged artifact: $zipPath"
    } else {
        Write-Warning "PyInstaller output not found at $distDir"
    }
}
finally {
    Pop-Location
}
