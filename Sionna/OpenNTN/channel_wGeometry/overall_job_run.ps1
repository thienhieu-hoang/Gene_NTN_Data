# ==============================================================================
# Master PowerShell Orchestrator: Generate OpenNTN Datasets
# Location: channel_wGeometry\overall_job_run.ps1
# ==============================================================================
# This master script discovers and invokes dataset-specific runner scripts
# located in results\<dataset_folder>\run_gen_this_dataset.ps1
#
# Usage:
#   # Run all datasets:
#   .\overall_job_run.ps1 -Target "All" -Samples 20000
#
#   # Run only the Ka-band 120 kHz dataset:
#   .\overall_job_run.ps1 -Target "KaBand_120kHz" -Samples 20000
#
#   # Dry-run check:
#   .\overall_job_run.ps1 -Target "KaBand_120kHz" -DryRun
# ==============================================================================

param (
    [string]$Target = "KaBand_120kHz",  # "KaBand_120kHz", "All", or folder name
    [int]$Samples = 20000,
    [int]$BatchSize = 64,
    [int]$DryRun = 0
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
if (-not $ScriptDir) { $ScriptDir = Get-Location }
$ResultsDir = "$ScriptDir\results"

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " Master OpenNTN Dataset Generation Orchestrator" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " Workspace  : $ScriptDir"
Write-Host " Results Dir: $ResultsDir"
Write-Host " Target Job : $Target"
Write-Host " Samples    : $Samples"
Write-Host " Dry-Run    : $DryRun"
Write-Host "==================================================================" -ForegroundColor Cyan

# Dataset Registry: Map target aliases to runner paths
$Jobs = @()

if ($Target -eq "KaBand_120kHz" -or $Target -eq "All") {
    $KaBandRunner = "$ResultsDir\openNTN_KaBand_120kHz_20GHz_Diverse_50kmBeam\run_gen_this_dataset.ps1"
    if (Test-Path $KaBandRunner) {
        $Jobs += @{
            Name   = "Ka-Band 120 kHz (20 GHz Diverse 50km Beam)"
            Runner = $KaBandRunner
        }
    }
}

if ($Jobs.Count -eq 0) {
    # Check if user specified a direct folder name inside results/
    $CustomRunner = "$ResultsDir\$Target\run_gen_this_dataset.ps1"
    if (Test-Path $CustomRunner) {
        $Jobs += @{
            Name   = $Target
            Runner = $CustomRunner
        }
    } else {
        Write-Host "Error: No runner found for target '$Target' at: $CustomRunner" -ForegroundColor Red
        exit 1
    }
}

$jobIndex = 1
$totalSw = [System.Diagnostics.Stopwatch]::StartNew()

foreach ($job in $Jobs) {
    Write-Host "`n>>> [$jobIndex/$($Jobs.Count)] Starting Job: $($job.Name)" -ForegroundColor Yellow
    Write-Host "    Runner: $($job.Runner)" -ForegroundColor DarkGray
    
    $jobSw = [System.Diagnostics.Stopwatch]::StartNew()
    & powershell -ExecutionPolicy Bypass -File $job.Runner -Samples $Samples -BatchSize $BatchSize -DryRun $DryRun
    $jobSw.Stop()
    
    Write-Host "    Finished in $([math]::Round($jobSw.Elapsed.TotalMinutes, 2)) min." -ForegroundColor Green
    $jobIndex++
}

$totalSw.Stop()
Write-Host "`n==================================================================" -ForegroundColor Green
Write-Host " All jobs completed successfully in $([math]::Round($totalSw.Elapsed.TotalMinutes, 2)) minutes!" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green
