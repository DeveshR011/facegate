# Builds FaceGate.exe with Nuitka (MSVC) - the whole app, the C++ engine, its runtime and models.
#
#   tools\build_exe.ps1               # folder build  -> dist\FaceGate\FaceGate.exe  (fast start; installable)
#   tools\build_exe.ps1 -OneFile      # single file   -> dist\FaceGate-portable.exe  (convenient; no elevated features)
#
# Prerequisites (one-off):
#   pip install -r requirements.txt nuitka ordered-set zstandard onnx onnxconverter-common
#   engine\build.ps1                  C++ engine + onnxruntime/DirectML runtime
#   python tools\prepare_models.py    static-shape FP16 models  (-> engine\models)
#   credential_provider\build.ps1     lock-screen component
param(
    [switch]$OneFile,
    [int]$Jobs = [Math]::Max(1, [Environment]::ProcessorCount - 2)
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$engine = "engine\build\Release"
$required = @("$engine\facegate_engine.dll", "$engine\onnxruntime.dll", "$engine\DirectML.dll",
              "$engine\onnxruntime_providers_shared.dll", "engine\models\manifest.json",
              "credential_provider\build\Release\FaceGateCP.dll", "assets\facegate.ico")
foreach ($f in $required) {
    if (-not (Test-Path $f)) { throw "Missing $f - see the prerequisites at the top of this script." }
}

# A running copy (even an elevated one, whose path cannot be read) locks its exe; replacing the
# output would then fail halfway, after the compile. Check before spending minutes on it.
foreach ($f in "dist\FaceGate\FaceGate.exe", "dist\FaceGate-portable.exe") {
    if (Test-Path $f) {
        try { [IO.File]::Open((Join-Path $root $f), "Open", "ReadWrite", "None").Dispose() }
        catch { throw "$f is in use - quit FaceGate from the tray first." }
    }
}

$version = (& $py -c "import facegate; print(facegate.__version__)").Trim()
$mode = if ($OneFile) { "--onefile" } else { "--standalone" }
$nuitka = @(
    "-m", "nuitka", $mode, "--msvc=latest", "--jobs=$Jobs",
    "--enable-plugin=pyside6", "--include-package=facegate",
    # Not needed at run time: the C++ engine replaces ONNX Runtime for Python, and the HTTP API
    # and the model-conversion tooling are for source installs and the container.
    "--nofollow-import-to=onnxruntime,onnx,onnxconverter_common,fastapi,uvicorn,starlette,pydantic,pydantic_core,httpx,anyio,h11,pytest,tkinter,setuptools,pip",
    "--windows-console-mode=attach",
    "--windows-icon-from-ico=assets\facegate.ico",
    "--company-name=FaceGate", "--product-name=FaceGate",
    "--file-description=FaceGate face unlock",
    "--file-version=$version.0", "--product-version=$version.0",
    "--include-data-files=$engine\facegate_engine.dll=native\facegate_engine.dll",
    "--include-data-files=$engine\onnxruntime.dll=native\onnxruntime.dll",
    "--include-data-files=$engine\onnxruntime_providers_shared.dll=native\onnxruntime_providers_shared.dll",
    "--include-data-files=$engine\DirectML.dll=native\DirectML.dll",
    "--include-data-dir=engine\models=native\models",
    "--include-data-files=credential_provider\build\Release\FaceGateCP.dll=native\FaceGateCP.dll",
    "--output-dir=dist", "--output-filename=FaceGate.exe"
)
if ($OneFile) {
    # Unpack once into the per-user cache instead of on every launch.
    $nuitka += "--onefile-tempdir-spec={CACHE_DIR}/FaceGate/{VERSION}"
}
$nuitka += "launch_facegate.py"

Write-Host "Compiling with Nuitka ($mode, $Jobs jobs) - the first build takes a while..."
$started = Get-Date
& $py @nuitka
if ($LASTEXITCODE -ne 0) { throw "Nuitka failed (exit $LASTEXITCODE)" }
Write-Host ("Compiled in {0:N1} min" -f ((Get-Date) - $started).TotalMinutes)

if ($OneFile) {
    $exe = Join-Path $root "dist\FaceGate-portable.exe"
    Move-Item (Join-Path $root "dist\FaceGate.exe") $exe -Force
} else {
    $dist = Join-Path $root "dist\FaceGate"
    if (Test-Path $dist) { Remove-Item $dist -Recurse -Force -Confirm:$false }
    Move-Item (Join-Path $root "dist\launch_facegate.dist") $dist
    Copy-Item "packaging\Install-FaceGate.ps1" $dist
    Copy-Item "LICENSE", "README.md" $dist
    $exe = Join-Path $dist "FaceGate.exe"
}
$size = (Get-ChildItem (Split-Path $exe) -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("`nBuilt {0}  ({1:N0} MB in {2})" -f $exe, $size, (Split-Path $exe))
