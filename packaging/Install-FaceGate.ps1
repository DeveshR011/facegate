# Installs the FaceGate folder build into Program Files and adds a Start Menu shortcut.
#
#   .\Install-FaceGate.ps1               install (asks for administrator rights)
#   .\Install-FaceGate.ps1 -Uninstall    remove it (your face data in C:\ProgramData\FaceGate is kept)
#   .\Install-FaceGate.ps1 -Uninstall -RemoveData
#
# Why install at all: FaceGate can run elevated (to lock apps that run as administrator, and for
# the lock-screen service). Code that runs elevated must not live in a folder ordinary programs can
# write to, or they could swap it for their own. Program Files is writable by administrators only.
param(
    [switch]$Uninstall,
    [switch]$RemoveData
)
$ErrorActionPreference = "Stop"
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$isAdmin = ([Security.Principal.WindowsPrincipal]$identity).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    $forward = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"")
    if ($Uninstall) { $forward += "-Uninstall" }
    if ($RemoveData) { $forward += "-RemoveData" }
    Start-Process powershell -Verb RunAs -ArgumentList $forward -Wait
    exit
}

$target = Join-Path $env:ProgramFiles "FaceGate"
$shortcut = Join-Path $env:ProgramData "Microsoft\Windows\Start Menu\Programs\FaceGate.lnk"
$data = Join-Path $env:ProgramData "FaceGate"

function Stop-FaceGate {
    Get-Process -Name "FaceGate" -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Milliseconds 700
}

if ($Uninstall) {
    $exe = Join-Path $target "FaceGate.exe"
    if (Test-Path $exe) {                                   # let the app remove its own tasks and registry keys
        & $exe admin unlock-uninstall | Out-Null
        & $exe admin autostart-off | Out-Null
    }
    Stop-FaceGate
    if (Test-Path $shortcut) { Remove-Item $shortcut -Force -Confirm:$false }
    if (Test-Path $target) { Remove-Item $target -Recurse -Force -Confirm:$false }
    if ($RemoveData -and (Test-Path $data)) { Remove-Item $data -Recurse -Force -Confirm:$false }
    Write-Host "FaceGate removed." -ForegroundColor Green
    exit
}

if (-not (Test-Path (Join-Path $PSScriptRoot "FaceGate.exe"))) {
    throw "Run this script from the FaceGate folder (the one that contains FaceGate.exe)."
}
Stop-FaceGate
New-Item -ItemType Directory -Force $target | Out-Null
robocopy $PSScriptRoot $target /MIR /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "Copy failed (robocopy exit $LASTEXITCODE)" }

$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut($shortcut)
$link.TargetPath = Join-Path $target "FaceGate.exe"
$link.WorkingDirectory = $target
$link.Description = "FaceGate face unlock"
$link.Save()

Write-Host "Installed to $target" -ForegroundColor Green
Write-Host "Start it from the Start Menu (FaceGate). Elevated features are now available."
