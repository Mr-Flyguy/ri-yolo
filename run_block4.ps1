# ==============================================================================
# RI-YOLO: Block 4 Training Runner (run_block4.ps1)
# ==============================================================================
# Executes the revised Block 4 matrix (Opus decision 16.09.2026):
#   1. e7p__nwd-calib-norsl__s0 (CRITICAL: NWD with live gate, use_rsl=False)
#   2. e7p__nwd-sizegate__s0 in runs_fixed/ (CRITICAL: sizegate with broadcasting fix)
#   3. e6p__lam-1e-4__s1 (HIGH: partially live gate seed 1)
#   4. e6p__lam-1e-4__s2 (HIGH: partially live gate seed 2)
#   5. e6p__lam-1e-5__s0 (HIGH: boundary collapse check lambda=1e-5)
#   --- Optional runs (if time permits) ---
#   6. e7p__nwd-calib__s1
#   7. e7p__nwd-calib__s2
# ==============================================================================

[CmdletBinding()]
param (
    [string]$Data = "exdark.yaml",
    [string]$Device = "0",
    [string]$Weights = "weights/yolov8s.pt",
    [switch]$IncludeOptional = $false
)

Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "  RI-YOLO: BLOCK 4 REVISED TRAINING QUEUE" -ForegroundColor Cyan
Write-Host "  Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan

# 1. Directory verification
if (-not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }
if (-not (Test-Path "runs_fixed")) { New-Item -ItemType Directory -Path "runs_fixed" | Out-Null }

# 2. Base queue (5 mandatory runs in revised priority order)
$RunsQueue = @(
    @{ Name = "e6p__lam-1e-5__s0"; Project = "runs"; Desc = "1. Boundary collapse check (lambda=1e-5, seed 0)" },
    @{ Name = "e6p__lam-1e-4__s1"; Project = "runs"; Desc = "2. Partially live gate multi-seed (lambda=1e-4, seed 1)" },
    @{ Name = "e6p__lam-1e-4__s2"; Project = "runs"; Desc = "3. Partially live gate multi-seed (lambda=1e-4, seed 2)" },
    @{ Name = "e7p__nwd-calib-norsl__s0"; Project = "runs"; Desc = "4. CRITICAL: NWD calib without RSL (live gate)" },
    @{ Name = "e7p__nwd-sizegate__s0"; Project = "runs_fixed"; Desc = "5. CRITICAL: Fixed size-gate NWD (in runs_fixed)" }
)

if ($IncludeOptional) {
    $RunsQueue += @(
        @{ Name = "e7p__nwd-calib__s1"; Project = "runs"; Desc = "OPTIONAL: NWD calib seed 1" },
        @{ Name = "e7p__nwd-calib__s2"; Project = "runs"; Desc = "OPTIONAL: NWD calib seed 2" }
    )
}

$OverallStartTime = Get-Date

foreach ($run in $RunsQueue) {
    $runName = $run.Name
    $runProj = $run.Project
    $runDesc = $run.Desc
    $logFile = "logs/$runName.log"
    $runStartTime = Get-Date

    Write-Host "`n--------------------------------------------------------------------" -ForegroundColor Green
    Write-Host ">>> STARTING RUN: $runName (project: $runProj)" -ForegroundColor Green
    Write-Host "    Description : $runDesc" -ForegroundColor Green
    Write-Host "    Started at  : $($runStartTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Green
    Write-Host "--------------------------------------------------------------------" -ForegroundColor Green

    try {
        python scripts/train_phase2.py `
            --run $runName `
            --data $Data `
            --device $Device `
            --weights $Weights `
            --project $runProj `
            2>&1 | Tee-Object -FilePath $logFile

        $exitCode = $LASTEXITCODE
    } catch {
        Write-Host "[ERROR] Exception during execution: $_" -ForegroundColor Red
        $exitCode = 1
    }

    $runDuration = (Get-Date) - $runStartTime
    if ($exitCode -eq 0) {
        Write-Host "[SUCCESS] $runName finished in $($runDuration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
        python scripts/postrun_check.py --run-dir "$runProj/$runName"
    } else {
        Write-Host "[FAILURE] $runName failed with exit code $exitCode" -ForegroundColor Red
        exit $exitCode
    }
}

$totalDuration = (Get-Date) - $OverallStartTime
Write-Host "`n====================================================================" -ForegroundColor Cyan
Write-Host "  BLOCK 4 ALL RUNS COMPLETED in $($totalDuration.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
