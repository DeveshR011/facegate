# Builds FaceGateCP.dll (x64, Release) into credential_provider\build\Release.
# Requires Visual Studio 2022 Build Tools with the C++ workload, and CMake.
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$build = Join-Path $here "build"

cmake -S $here -B $build -A x64
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }
cmake --build $build --config Release
if ($LASTEXITCODE -ne 0) { throw "Build failed" }

Write-Host ""
Write-Host "Built: $(Join-Path $build 'Release\FaceGateCP.dll')"
Write-Host "Install it from FaceGate > Windows > 'Unlock Windows with your face'."
