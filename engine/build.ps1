# Builds facegate_engine.dll (x64, Release) into engine\build\Release together with the
# runtime DLLs it needs. Requires Visual Studio 2022 Build Tools (C++) and CMake.
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$py = Join-Path (Split-Path $here -Parent) ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

& (Join-Path $here "fetch_deps.ps1")
& $py (Join-Path (Split-Path $here -Parent) "tools\gen_mask_header.py")
if ($LASTEXITCODE -ne 0) { throw "mask header generation failed" }

$build = Join-Path $here "build"
cmake -S $here -B $build -A x64
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }
cmake --build $build --config Release
if ($LASTEXITCODE -ne 0) { throw "Build failed" }
Write-Host "`nBuilt: $(Join-Path $build 'Release\facegate_engine.dll')"
