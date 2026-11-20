# ==============================================================================
# Batch Dataset Generator: OpenNTN Geometry Scenarios
# ==============================================================================
# Generates ONLY the raw unnormalized .npy datasets (Train: 16,000, Val: 2,000, Test: 2,000)
# for the 5 scenarios specified in plan.md using gen_channel_v4_wGeometry_RandomizedUE.py.
#
# Usage:
#   .\generate_plan_npy_datasets.ps1 [-Samples 20000] [-BatchSize 64] [-DryRun]
# ==============================================================================

param (
    [int]$Samples = 20000,
    [int]$BatchSize = 64,
    [switch]$DryRun = $false,
    [switch]$SkipExisting = $true
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
if (-not $ScriptDir) { $ScriptDir = Get-Location }
$GeneratorPy = "$ScriptDir\gen_channel_v4_wGeometry_RandomizedUE.py"
$PythonExe = "C:\Users\AT30890\AppData\Local\anaconda3\envs\TF_GPU-py3_11\python.exe"
$ResultsDir = "$ScriptDir\results"

# 5 Target Scenarios from plan.md:
$PlanScenarios = @(
    @{
        Name        = "openNTN_DUR300nsFix_LoS_2p18G_600km_30deg"
        Scenario    = "dur"
        DelaySpread = 300
        Elevation   = 30
        LoS         = "los"
        FolderTag   = "DUR300nsFix_LoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps"
    },
    @{
        Name        = "openNTN_DUR500nsFix_NLoS_2p18G_600km_30deg"
        Scenario    = "dur"
        DelaySpread = 500
        Elevation   = 30
        LoS         = "nlos"
        FolderTag   = "DUR500nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps"
    },
    @{
        Name        = "openNTN_DUR300nsFix_NLoS_2p18G_600km_70deg"
        Scenario    = "dur"
        DelaySpread = 300
        Elevation   = 70
        LoS         = "nlos"
        FolderTag   = "DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_70deg_r15km_20to30mps"
    },
    @{
        Name        = "openNTN_DUR100_LoS_70deg"
        Scenario    = "dur"
        DelaySpread = 100
        Elevation   = 70
        LoS         = "los"
        FolderTag   = "DUR100nsFix_LoS_port1_Apos2_2p18G_600km_70deg_r15km_20to30mps"
    },
    @{
        Name        = "openNTN_DUR300_NLoS_30deg"
        Scenario    = "dur"
        DelaySpread = 300
        Elevation   = 30
        LoS         = "nlos"
        FolderTag   = "DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps"
    }
)

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " OpenNTN .npy Dataset Batch Generator (plan.md)" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " Target Samples : $Samples (16,000 train / 2,000 val / 2,000 test)"
Write-Host " Batch Size     : $BatchSize"
Write-Host " Generator      : $GeneratorPy"
Write-Host " Output Dir     : $ResultsDir"
Write-Host " Skip Existing  : $SkipExisting"
Write-Host " Dry-Run Mode   : $DryRun"
Write-Host "==================================================================" -ForegroundColor Cyan

$scenarioIndex = 1
foreach ($sc in $PlanScenarios) {
    $targetFolder = "$ResultsDir\$($sc.FolderTag)"
    $trainNpy = "$targetFolder\H_train.npy"
    $valNpy   = "$targetFolder\H_val.npy"
    $testNpy  = "$targetFolder\H_test.npy"

    Write-Host "`n[$scenarioIndex/$($PlanScenarios.Count)] Scenario: $($sc.Name)" -ForegroundColor Yellow
    Write-Host "  Folder Tag : $($sc.FolderTag)"

    if ($SkipExisting -and (Test-Path $trainNpy) -and (Test-Path $valNpy) -and (Test-Path $testNpy)) {
        # Check existing size to ensure it's full (e.g. > 10MB)
        $trainItem = Get-Item $trainNpy
        if ($trainItem.Length -gt 10MB) {
            Write-Host "  [SKIP] Dataset already generated with $([math]::Round($trainItem.Length / 1MB, 1)) MB. Skipping." -ForegroundColor DarkGray
            $scenarioIndex++
            continue
        } else {
            Write-Host "  [REGEN] Existing dataset is small/incomplete ($($trainItem.Length) bytes). Regenerating full dataset." -ForegroundColor Magenta
        }
    }

    $cmdArgs = @(
        "$GeneratorPy",
        "--samples", "$Samples",
        "--batch-size", "$BatchSize",
        "--save-only-npy",
        "--scenario", "$($sc.Scenario)",
        "--delay-spread", "$($sc.DelaySpread)",
        "--elevation", "$($sc.Elevation)",
        "--los", "$($sc.LoS)"
    )

    if ($DryRun) {
        Write-Host "  [DRY-RUN] $PythonExe $($cmdArgs -join ' ')" -ForegroundColor Magenta
    } else {
        Write-Host "  [RUN] Generating $Samples drops in batches of $BatchSize..." -ForegroundColor Green
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        & $PythonExe $cmdArgs
        $sw.Stop()
        Write-Host "  [DONE] Completed in $([math]::Round($sw.Elapsed.TotalSeconds, 1)) s" -ForegroundColor Green
    }
    $scenarioIndex++
}

Write-Host "`n==================================================================" -ForegroundColor Cyan
Write-Host " Batch Generation Completed!" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
