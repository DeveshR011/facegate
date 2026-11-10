# Builds FaceGate.exe with Nuitka (MSVC) - the whole app, the C++ engine, its runtime and models.
#
#   tools\build_exe.ps1               # folder build  -> dist\FaceGate\FaceGate.exe  (fast start; installable)
#   tools\build_exe.ps1 -OneFile      # single file   -> dist\FaceGate-portable.exe  (convenient; no elevated features)
#
# Rebuilds are incremental: with MSVC, Nuitka caches every compiled C file with its bundled clcache
# (ccache's MSVC counterpart) under %LOCALAPPDATA%\Nuitka\Nuitka\Cache\clcache; the hit rate is
# printed at the end. packaging\nuitka\facegate_plugin.py bundles the engine and models and trims
# what Windows never loads.
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
    "-m", "nuitka", $mode, "--msvc=latest", "--jobs=$Jobs", "--assume-yes-for-downloads",
    # Link-time optimisation redoes code generation for all 200 C files at link time, on every
    # build, cached or not (3 of 4.3 minutes here). Inference runs in the C++ engine, so the
    # compiled Python gains nothing worth that.
    "--lto=no",
    "--enable-plugin=pyside6", "--include-package=facegate",
    # Engine, runtime, models and the lock-screen DLL; Windows-only trims; bytecode for constant tables.
    "--user-plugin=packaging\nuitka\facegate_plugin.py",
    # Not needed at run time: the C++ engine replaces ONNX Runtime for Python, and the HTTP API
    # and the model-conversion tooling are for source installs and the container.
    "--nofollow-import-to=onnxruntime,onnx,onnxconverter_common,fastapi,uvicorn,starlette,pydantic,pydantic_core,httpx,anyio,h11,pytest,tkinter,setuptools,pip",
    # OpenCV's ffmpeg plugin (29 MB) decodes video files; webcams use DirectShow / Media
    # Foundation, which are built into cv2.pyd (`FaceGate.exe selftest` lists them).
    "--noinclude-dlls=cv2/opencv_videoio_ffmpeg*.dll",
    # The app paints its own icons, uses the built-in Fusion style and no TLS: only the
    # platform plugin is needed.
    "--noinclude-qt-plugins=tls,imageformats,iconengines,styles",
    "--windows-console-mode=attach",
    "--windows-icon-from-ico=assets\facegate.ico",
    "--company-name=FaceGate", "--product-name=FaceGate",
    "--file-description=FaceGate face unlock",
    "--file-version=$version.0", "--product-version=$version.0",
    "--output-dir=dist", "--output-filename=FaceGate.exe"
)
if ($OneFile) {
    # Unpack once into the per-user cache instead of on every launch.
    $nuitka += "--onefile-tempdir-spec={CACHE_DIR}/FaceGate/{VERSION}"
}
$nuitka += "launch_facegate.py"

Write-Host "Compiling with Nuitka ($mode, $Jobs jobs) - the first build takes a while..."
$started = Get-Date
$log = Join-Path $root "dist\nuitka-build.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
# Nuitka reports progress on stderr; keep it as plain text (and in a log) rather than as errors.
$ErrorActionPreference = "Continue"
& $py @nuitka 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
$exit = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($exit -ne 0) { throw "Nuitka failed (exit $exit) - see $log" }
Write-Host ("Compiled in {0:N1} min" -f ((Get-Date) - $started).TotalMinutes)
$cache = Select-String -Path $log -Pattern "C files using clcache with .*" | Select-Object -Last 1
if ($cache) { Write-Host ("C compiler cache: " + $cache.Matches[0].Value) }

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
