# ==============================================================================
# PowerShell Runner: Generate Diverse OpenNTN Dataset (.npy only)
# ==============================================================================
# Parameters:
#   - Scenarios: dur (Dense Urban), urb (Urban), sur (Suburban)
#   - Beam Footprint: 50 km radius (100 km diameter)
#   - Velocity: 20 to 40 m/s (72 to 144 km/h)
#   - Numerology: Diverse SCS in {15 kHz, 30 kHz, 60 kHz}
#   - Grid Shape: Fixed 132 subcarriers x 14 OFDM symbols
#   - Elevation: 20° to 80°
#   - Coupled: Delay spread, LoS/NLoS, and Elevation physically consistent
#   - Output: H_train.npy, H_val.npy, H_test.npy + sample_metadata.mat
#
# Usage:
#   .\run_gen_diverse_dataset.ps1 [-Samples 20000] [-BatchSize 64] [-DryRun]
# ==============================================================================

param (
    [int]$Samples = 20000,
    [int]$BatchSize = 64,
    [string]$Mode = "diverse",
    [float]$BeamRadiusKm = 50.0,
    [string]$OutName = $null,
    [Nullable[float]]$ElevMin = $null,
    [Nullable[float]]$ElevMax = $null,
    [Nullable[float]]$VMin = $null,
    [Nullable[float]]$VMax = $null,
    [Nullable[float]]$DsMin = $null,
    [Nullable[float]]$DsMax = $null,
    [string]$Los = $null,
    [Nullable[float]]$OuterRingRatio = $null,
    [Nullable[float]]$CarrierFrequency = $null,
    [Nullable[float]]$SCS = $null,
    [switch]$NoMat = $false,
    [switch]$DryRun = $false
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
if (-not $ScriptDir) { $ScriptDir = Get-Location }

$GeneratorPy = "$ScriptDir\gen_diverse_openntn_dataset.py"
$PythonExe = "C:\Users\AT30890\AppData\Local\anaconda3\envs\TF_GPU-py3_11\python.exe"
$OutputDir = "$ScriptDir\results\$OutName"

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " Diverse OpenNTN Dataset Generator Runner" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " Python Environment : $PythonExe"
Write-Host " Generator Script   : $GeneratorPy"
Write-Host " Total Drops        : $Samples (Train: $([int]($Samples*0.8)), Val: $([int]($Samples*0.1)), Test: $([int]($Samples*0.1)))"
Write-Host " Batch Size         : $BatchSize"
Write-Host " Beam Footprint     : $BeamRadiusKm km radius"
Write-Host " Target Output Dir  : $OutputDir"
Write-Host " Dry-Run Mode       : $DryRun"
Write-Host "==================================================================" -ForegroundColor Cyan

$cmdArgs = @(
    "$GeneratorPy",
    "--samples", "$Samples",
    "--batch-size", "$BatchSize",
    "--mode", "$Mode",
    "--beam-radius-km", "$BeamRadiusKm"
)
if ($OutName) {
    $cmdArgs += @("--out-name", "$OutName")
}
if ($null -ne $ElevMin) { $cmdArgs += @("--elev-min", "$ElevMin") }
if ($null -ne $ElevMax) { $cmdArgs += @("--elev-max", "$ElevMax") }
if ($null -ne $VMin) { $cmdArgs += @("--v-min", "$VMin") }
if ($null -ne $VMax) { $cmdArgs += @("--v-max", "$VMax") }
if ($null -ne $DsMin) { $cmdArgs += @("--ds-min", "$DsMin") }
if ($null -ne $DsMax) { $cmdArgs += @("--ds-max", "$DsMax") }
if ($Los) { $cmdArgs += @("--los", "$Los") }
if ($null -ne $OuterRingRatio) { $cmdArgs += @("--outer-ring-ratio", "$OuterRingRatio") }
if ($null -ne $CarrierFrequency) { $cmdArgs += @("--carrier-frequency", "$CarrierFrequency") }
if ($null -ne $SCS) { $cmdArgs += @("--scs", "$SCS") }
if ($NoMat) { $cmdArgs += @("--no-mat") }

if ($DryRun) {
    Write-Host "`n[DRY-RUN] Command to execute:" -ForegroundColor Magenta
    Write-Host "$PythonExe $($cmdArgs -join ' ')" -ForegroundColor Gray
} else {
    Write-Host "`n[START] Generating $Samples diverse OpenNTN drops..." -ForegroundColor Green
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $PythonExe $cmdArgs
    $sw.Stop()
    Write-Host "`n[SUCCESS] Completed in $([math]::Round($sw.Elapsed.TotalMinutes, 2)) minutes ($([math]::Round($sw.Elapsed.TotalSeconds, 1)) s)!" -ForegroundColor Green
}
