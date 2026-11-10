# Downloads the ONNX Runtime (DirectML build) SDK and the DirectML runtime from NuGet
# into engine\third_party, and refuses to continue unless every DLL carries a valid
# Microsoft signature. Idempotent: skips what is already present and verified.
$ErrorActionPreference = "Stop"
$OrtVersion = "1.24.4"      # keep equal to onnxruntime-directml in requirements.txt
$DmlVersion = "1.15.4"      # DirectML release ONNX Runtime 1.24 is built against
$root = Join-Path $PSScriptRoot "third_party"
New-Item -ItemType Directory -Force $root | Out-Null

function Get-NuGetPackage($id, $version, $dest) {
    if (Test-Path $dest) { return }
    $url = "https://api.nuget.org/v3-flatcontainer/$id/$version/$id.$version.nupkg".ToLower()
    $zip = Join-Path $root "$id.zip"
    Write-Host "downloading $id $version"
    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
    Expand-Archive $zip -DestinationPath $dest -Force
    Remove-Item $zip -Confirm:$false
}

function Assert-MicrosoftSigned($path) {
    $sig = Get-AuthenticodeSignature $path
    if ($sig.Status -ne "Valid" -or $sig.SignerCertificate.Subject -notmatch "Microsoft") {
        throw "$path is not validly signed by Microsoft (status: $($sig.Status))"
    }
    Write-Host ("  signed by Microsoft: {0}" -f (Split-Path $path -Leaf))
}

Get-NuGetPackage "Microsoft.ML.OnnxRuntime.DirectML" $OrtVersion (Join-Path $root "ort")
Get-NuGetPackage "Microsoft.AI.DirectML" $DmlVersion (Join-Path $root "directml")

$ortDll = Join-Path $root "ort\runtimes\win-x64\native\onnxruntime.dll"
$dmlDll = Join-Path $root "directml\bin\x64-win\DirectML.dll"
foreach ($f in $ortDll, $dmlDll, (Join-Path $root "ort\build\native\include\onnxruntime_cxx_api.h")) {
    if (-not (Test-Path $f)) { throw "expected file missing after download: $f" }
}
Assert-MicrosoftSigned $ortDll
Assert-MicrosoftSigned $dmlDll
Write-Host "dependencies ready in $root"
