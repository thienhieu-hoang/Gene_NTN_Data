# PowerShell Execution Wrapper for Diverse Terrestrial 3GPP TR 38.901 Dataset Generation
# Generates 20,000 diverse terrestrial channel samples (.npy) across:
#   - BS Height: 25.0 m (UMa Macrocell)
#   - UE Height: 1.5 m
#   - Cell Radius: 500 m (uniform area density)
#   - UE Velocity: 5 to 35 m/s (18 to 126 km/h vehicular mobility)
#   - Profiles: 3GPP TDL-A, TDL-B, TDL-C, TDL-D
#   - Delay spread: 30 to 250 ns (LOS), 150 to 800 ns (NLOS)
#   - Doppler: Jakes isotropic spectrum (no satellite carrier offset)
#   - Path loss: 3GPP TR 38.901 UMa with log-normal shadow fading

param(
    [int]$NumUE = 20000,
    [string]$FolderName = "Diverse_Terrestrial_2p18e9_UMa_30kHz",
    [double]$CarrierFrequency = 2.18e9,
    [int]$SubcarrierSpacing = 30,
    [double]$BSHeight = 25.0,
    [double]$UEHeight = 1.5,
    [double]$CellRadius = 500.0
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "  Launching Diverse Terrestrial Dataset Generator (NumUE = $NumUE)" -ForegroundColor Cyan
Write-Host "  Model: 3GPP TR 38.901 UMa (BS = $BSHeight m, Cell Radius = $CellRadius m)" -ForegroundColor Cyan
Write-Host "  Output folder: generatedChannel_Results\$FolderName" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan

$matlabCmd = "& 'C:\Program Files\MATLAB\R2025a\bin\matlab.exe' -batch ""cd('$scriptDir'); Gen_Chan_Diverse_Terrestrial_Dataset('NumUE', $NumUE, 'FolderName', '$FolderName', 'CarrierFrequency', $CarrierFrequency, 'SubcarrierSpacing', $SubcarrierSpacing, 'BSHeight', $BSHeight, 'UEHeight', $UEHeight, 'CellRadius', $CellRadius);"""

$sw = [System.Diagnostics.Stopwatch]::StartNew()
Invoke-Expression $matlabCmd
$sw.Stop()

Write-Host "`nTotal elapsed time: $($sw.Elapsed.TotalMinutes.ToString('0.00')) minutes." -ForegroundColor Green
