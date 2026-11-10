# Installs the FaceGate folder build into Program Files, adds a Start Menu shortcut and starts
# FaceGate with Windows, in the tray. There it sleeps (no camera, models unloaded, ~0% CPU) until
# a protected app or locked file is opened.
#
#   .\Install-FaceGate.ps1                install (or update), start with Windows, and start it now
#   .\Install-FaceGate.ps1 -NoAutostart   install only; start it from the Start Menu when you want it
#   .\Install-FaceGate.ps1 -Uninstall     remove it (your face data in C:\ProgramData\FaceGate is kept)
#   .\Install-FaceGate.ps1 -Uninstall -RemoveData
#
# Why install at all: FaceGate can run elevated (to lock apps that run as administrator, and for
# the lock-screen service). Code that runs elevated must not live in a folder ordinary programs can
# write to, or they could swap it for their own. Program Files is writable by administrators only.
param(
    [switch]$Uninstall,
    [switch]$RemoveData,
    [switch]$NoAutostart,
    # Set by the relaunch below. Approving the UAC prompt may switch to another (administrator)
    # account; FaceGate should still start at sign-in for the person who ran the installer.
    [string]$ForUser
)
$ErrorActionPreference = "Stop"
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$isAdmin = ([Security.Principal.WindowsPrincipal]$identity).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    $forward = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"",
                 "-ForUser", "`"$($identity.Name)`"")
    if ($Uninstall) { $forward += "-Uninstall" }
    if ($RemoveData) { $forward += "-RemoveData" }
    if ($NoAutostart) { $forward += "-NoAutostart" }
    $elevated = Start-Process powershell -Verb RunAs -ArgumentList $forward -Wait -PassThru
    exit $elevated.ExitCode
}
$relaunched = [bool]$ForUser          # running in a window of its own, which closes when we end
if (-not $ForUser) { $ForUser = $identity.Name }

$target = Join-Path $env:ProgramFiles "FaceGate"
$exe = Join-Path $target "FaceGate.exe"
$provider = Join-Path $target "FaceGateCP.dll"     # where 'admin unlock-install' puts the lock-screen DLL
$shortcut = Join-Path $env:ProgramData "Microsoft\Windows\Start Menu\Programs\FaceGate.lnk"
$data = Join-Path $env:ProgramData "FaceGate"

function Stop-FaceGate {
    Get-Process -Name "FaceGate" -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Milliseconds 700
}

function Invoke-FaceGate([string[]]$arguments) {
    # FaceGate.exe is a windowed program: PowerShell would neither wait for it nor see its exit code.
    (Start-Process $exe -ArgumentList $arguments -Wait -PassThru).ExitCode
}

function Get-FaceGateTask([string]$name) {
    Get-ScheduledTask -TaskPath "\FaceGate\" -TaskName $name -ErrorAction SilentlyContinue
}

function Uninstall-FaceGate {
    if (Test-Path $exe) {                                   # let the app remove its own tasks and registry keys
        Invoke-FaceGate @("admin", "unlock-uninstall") | Out-Null
        Invoke-FaceGate @("admin", "autostart-off") | Out-Null
    }
    Stop-FaceGate
    if (Test-Path $shortcut) { Remove-Item $shortcut -Force -Confirm:$false }
    if (Test-Path $target) { Remove-Item $target -Recurse -Force -Confirm:$false }
    if ($RemoveData -and (Test-Path $data)) { Remove-Item $data -Recurse -Force -Confirm:$false }
    Write-Host "FaceGate removed." -ForegroundColor Green
}

function Install-FaceGate {
    if (-not (Test-Path (Join-Path $PSScriptRoot "FaceGate.exe"))) {
        throw "Run this script from the FaceGate folder (the one that contains FaceGate.exe)."
    }
    Stop-FaceGate                                           # the app, and the lock-screen service if on
    New-Item -ItemType Directory -Force $target | Out-Null
    # /MIR also deletes what the new build no longer has, except the installed lock-screen DLL.
    # /R:2 /W:1: a file that stays in use fails the install in seconds, not after robocopy's
    # default of a million retries.
    robocopy $PSScriptRoot $target /MIR /XF $provider /R:2 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Copy failed (robocopy exit $LASTEXITCODE)" }
    $bundled = Join-Path $target "native\FaceGateCP.dll"
    if ((Test-Path $provider) -and (Test-Path $bundled)) {  # lock-screen unlock is on: update its DLL
        try { Copy-Item $bundled $provider -Force }
        catch { Write-Warning "The lock screen has FaceGateCP.dll open; run the installer again after a restart to update it." }
    }

    $shell = New-Object -ComObject WScript.Shell
    $link = $shell.CreateShortcut($shortcut)
    $link.TargetPath = $exe
    $link.WorkingDirectory = $target
    $link.Description = "FaceGate face unlock"
    $link.Save()
    Write-Host "Installed to $target" -ForegroundColor Green

    if (Get-FaceGateTask "UnlockService") {                 # stopped above; bring it straight back
        Start-ScheduledTask -TaskPath "\FaceGate\" -TaskName "UnlockService"
    }
    if ($NoAutostart) {
        Invoke-FaceGate @("admin", "autostart-off") | Out-Null
        Write-Host "Start it from the Start Menu (FaceGate). Elevated features are now available."
        return
    }
    # A sign-in task rather than a Run key: it starts FaceGate elevated without a UAC prompt, so
    # it can also guard apps that run as administrator.
    $code = Invoke-FaceGate @("admin", "autostart-on", "--user", "`"$ForUser`"")
    if ($code -ne 0) { throw "Could not set FaceGate to start with Windows (FaceGate.exe admin autostart-on: exit $code)." }
    Start-ScheduledTask -TaskPath "\FaceGate\" -TaskName "App"   # start it now, as at sign-in
    Write-Host "FaceGate now starts when $ForUser signs in, and is running in the tray now."
    Write-Host "Open it from the Start Menu or the tray icon to enroll your face and choose what to protect."
    Write-Host "(Turn this off under Windows > Start with Windows, or reinstall with -NoAutostart.)"
}

$code = 0
try {
    if ($Uninstall) { Uninstall-FaceGate } else { Install-FaceGate }
} catch {
    Write-Host "FaceGate setup failed: $($_.Exception.Message)" -ForegroundColor Red
    $code = 1
}
if ($relaunched) { Read-Host "Press Enter to close" | Out-Null }
exit $code
