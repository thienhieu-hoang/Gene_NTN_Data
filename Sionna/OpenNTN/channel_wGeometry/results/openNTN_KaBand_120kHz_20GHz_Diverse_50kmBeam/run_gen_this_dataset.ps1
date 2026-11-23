# ==============================================================================
# PowerShell Execution Script: Regenerate this Ka-Band 120 kHz Dataset
# Dataset: results\openNTN_KaBand_120kHz_20GHz_Diverse_50kmBeam
# ==============================================================================

param (
    [int]$Samples = 20000,
    [int]$BatchSize = 64,
    [float]$CarrierFrequency = 20.0e9,
    [float]$SCS = 120.0e3,
    [float]$BeamRadiusKm = 50.0,
    [float]$ElevMin = 20.0,
    [float]$ElevMax = 80.0,
    [int]$DryRun = 0
)

$isDry = ($DryRun -ne 0)

$ErrorActionPreference = "Stop"

# Resolve directories relative to this script
$CurrentDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
if (-not $CurrentDir) { $CurrentDir = Get-Location }

# The dataset folder name is the current directory's name
$DatasetName = Split-Path -Leaf $CurrentDir
$GeometryDir = Split-Path -Parent (Split-Path -Parent $CurrentDir)
$GeneratorPy = "$GeometryDir\gen_diverse_openntn_dataset.py"
$PythonExe   = "C:\Users\AT30890\AppData\Local\anaconda3\envs\TF_GPU-py3_11\python.exe"

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " Generator for: $DatasetName" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " Target Dataset Dir : $CurrentDir"
Write-Host " Python Script      : $GeneratorPy"
Write-Host " Total Drops        : $Samples (Train: $([int]($Samples*0.8)), Val: $([int]($Samples*0.1)), Test: $([int]($Samples*0.1)))"
Write-Host " Carrier Frequency  : $([math]::Round($CarrierFrequency / 1e9, 2)) GHz"
Write-Host " Subcarrier Spacing : $([math]::Round($SCS / 1e3, 0)) kHz"
Write-Host " Beam Footprint     : $BeamRadiusKm km radius"
Write-Host " Elevation Range    : $ElevMin deg to $ElevMax deg"
Write-Host " Dry-Run Mode       : $DryRun"
Write-Host "==================================================================" -ForegroundColor Cyan

$cmdArgs = @(
    "$GeneratorPy",
    "--samples", "$Samples",
    "--batch-size", "$BatchSize",
    "--carrier-frequency", "$CarrierFrequency",
    "--scs", "$SCS",
    "--beam-radius-km", "$BeamRadiusKm",
    "--elev-min", "$ElevMin",
    "--elev-max", "$ElevMax",
    "--out-name", "$DatasetName",
    "--no-mat"
)

if ($isDry) {
    Write-Host "`n[DRY-RUN] Command to execute:" -ForegroundColor Magenta
    Write-Host "$PythonExe $($cmdArgs -join ' ')" -ForegroundColor Gray
} else {
    Write-Host "`n[START] Generating $Samples Ka-band 120 kHz drops..." -ForegroundColor Green
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $PythonExe $cmdArgs
    $sw.Stop()
    Write-Host "`n[SUCCESS] Completed in $([math]::Round($sw.Elapsed.TotalMinutes, 2)) minutes ($([math]::Round($sw.Elapsed.TotalSeconds, 1)) s)!" -ForegroundColor Green
}
