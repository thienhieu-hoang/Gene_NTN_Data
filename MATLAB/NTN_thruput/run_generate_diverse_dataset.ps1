# PowerShell Execution Wrapper for Diverse NTN Dataset Generation
# Generates 20,000 diverse NTN channel samples (.npy) across:
#   - UE velocity: 10 to 30 m/s
#   - Elevation angle: 30 to 80 deg
#   - Delay spread: 30 to 500 ns (physically coupled with elevation and LOS)
#   - Profiles: NTN-TDL-A, B, C, D (with Rician K-factor scaling for C and D)

param(
    [int]$NumUE = 20000,
    [string]$FolderName = "Diverse_2p18e9_600km_30to80deg_30kHz"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "  Launching Diverse NTN Dataset Generator (NumUE = $NumUE)" -ForegroundColor Cyan
Write-Host "  Output folder: generatedChannel_Results\$FolderName" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan

$matlabCmd = "& 'C:\Program Files\MATLAB\R2025a\bin\matlab.exe' -batch ""cd('$scriptDir'); Gen_Chan_Diverse_Dataset('NumUE', $NumUE, 'FolderName', '$FolderName');"""

$sw = [System.Diagnostics.Stopwatch]::StartNew()
Invoke-Expression $matlabCmd
$sw.Stop()

Write-Host "`nTotal elapsed time: $($sw.Elapsed.TotalMinutes.ToString('0.00')) minutes." -ForegroundColor Green
