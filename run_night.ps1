# ==============================================================================
# RI-YOLO: Sequential Overnight Training Runner (run_night.ps1)
# ==============================================================================
# Executes 4 training runs sequentially for Article C1:
#   1. e2p__baseline__s0 (YOLOv8s baseline, seed 0, 100 epochs) [Block 3]
#   2. e2p__baseline__s3 (YOLOv8s baseline, seed 3, 100 epochs) [Block 3]
#   3. e3p__capctrl__s1  (Capacity control module, seed 1, 100 epochs) [Item 6]
#   4. e3p__capctrl__s2  (Capacity control module, seed 2, 100 epochs) [Item 6]
# ==============================================================================

[CmdletBinding()]
param (
    [string]$Data = "exdark.yaml",
    [string]$Device = "0",
    [int]$Epochs = 100,
    [string]$Weights = "weights/yolov8s.pt"
)

Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "  RI-YOLO: OVERNIGHT TRAINING QUEUE (4 RUNS)" -ForegroundColor Cyan
Write-Host "  Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan

# 1. Directory verification
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
    Write-Host "[INIT] Created directory logs/" -ForegroundColor Yellow
}
if (-not (Test-Path "tables")) {
    New-Item -ItemType Directory -Path "tables" | Out-Null
    Write-Host "[INIT] Created directory tables/" -ForegroundColor Yellow
}

# 2. File verification
if (-not (Test-Path $Weights)) {
    Write-Host "[ERROR] Pretrained weights '$Weights' not found! Aborting." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $Data)) {
    Write-Host "[ERROR] Dataset config '$Data' not found! Aborting." -ForegroundColor Red
    exit 1
}

# 3. Queue definition
$RunsQueue = @(
    @{ Name = "e2p__baseline__s0"; Desc = "Baseline YOLOv8s (seed 0) [Block 3]" },
    @{ Name = "e2p__baseline__s3"; Desc = "Baseline YOLOv8s (seed 3) [Block 3]" },
    @{ Name = "e3p__capctrl__s1";  Desc = "Capacity Control module (seed 1) [Item 6]" },
    @{ Name = "e3p__capctrl__s2";  Desc = "Capacity Control module (seed 2) [Item 6]" }
)

$OverallStartTime = Get-Date
$SummaryReport = @()

# 4. Sequential execution loop
foreach ($run in $RunsQueue) {
    $runName = $run.Name
    $runDesc = $run.Desc
    $logFile = "logs/$runName.log"
    $runStartTime = Get-Date

    Write-Host "`n--------------------------------------------------------------------" -ForegroundColor Green
    Write-Host ">>> STARTING RUN: $runName" -ForegroundColor Green
    Write-Host "    Description : $runDesc" -ForegroundColor Green
    Write-Host "    Log file    : $logFile" -ForegroundColor Green
    Write-Host "    Started at  : $($runStartTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Green
    Write-Host "--------------------------------------------------------------------" -ForegroundColor Green

    $runStatus = "UNKNOWN"
    $exitCode = 0

    try {
        $cmd = "python scripts/train_phase1.py --run $runName --data $Data --device $Device --epochs $Epochs --weights $Weights"
        Write-Host "[CMD] $cmd" -ForegroundColor Gray

        Invoke-Expression "$cmd 2>&1" | Tee-Object -FilePath $logFile
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            Write-Host "`n[POST-CHECK] Training $runName completed with code 0. Running verification..." -ForegroundColor Cyan

            # Detect actual output directory (Phase 1 saves into runs/detect/runs/)
            $targetRunDir = "runs/detect/runs/$runName"
            if (-not (Test-Path $targetRunDir)) {
                if (Test-Path "runs/$runName") {
                    $targetRunDir = "runs/$runName"
                }
            }
            Write-Host "[POST-CHECK] Target directory: $targetRunDir" -ForegroundColor Gray

            $checkCmd = "python scripts/postrun_check.py --run $targetRunDir"
            Invoke-Expression "$checkCmd 2>&1" | Tee-Object -FilePath $logFile -Append

            if ($LASTEXITCODE -eq 0) {
                $runStatus = "SUCCESS"
                Write-Host "[STATUS] $runName : COMPLETED AND VERIFIED" -ForegroundColor Green
            } else {
                $runStatus = "WARNING (post-check exit code $LASTEXITCODE)"
                Write-Host "[STATUS] $runName : Training finished, postrun_check reported warnings" -ForegroundColor Yellow
            }
        } else {
            $runStatus = "FAILED (exit code $exitCode)"
            Write-Host "[ERROR] Training error in $runName (Exit code: $exitCode)" -ForegroundColor Red
        }
    }
    catch {
        $runStatus = "EXCEPTION"
        $exitCode = -1
        $errText = $_.Exception.Message
        Write-Host "[EXCEPTION] Exception in $runName : $errText" -ForegroundColor Red
        Add-Content -Path $logFile -Value "`n[EXCEPTION] $errText"
    }

    $runEndTime = Get-Date
    $runElapsed = $runEndTime - $runStartTime
    $elapsedStr = "{0:D2}h {1:D2}m {2:D2}s" -f $runElapsed.Hours, $runElapsed.Minutes, $runElapsed.Seconds

    Write-Host "[TIME] Run $runName duration: $elapsedStr" -ForegroundColor Cyan

    $SummaryReport += [PSCustomObject]@{
        RunName  = $runName
        Status   = $runStatus
        ExitCode = $exitCode
        Duration = $elapsedStr
    }
}

# 5. Final aggregation and statistics
Write-Host "`n====================================================================" -ForegroundColor Cyan
Write-Host "  FINAL AGGREGATION AND STATISTICS (PAPER C1)" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan

try {
    Write-Host "[COLLECT] Regenerating tables/C1_table1.csv..." -ForegroundColor Gray
    python scripts/collect_c1.py --out tables/C1_table1.csv
    python scripts/collect_c1.py

    if (Test-Path "artifacts/phase1/summary.csv") {
        Write-Host "[STATS] Computing statistical tests (Welch, Mann-Whitney, Cohen d)..." -ForegroundColor Gray
        python scripts/stats_c1.py --summary artifacts/phase1/summary.csv --out tables/C1_stats.csv
    }
} catch {
    Write-Host "[WARN] Error during final table aggregation: $($_.Exception.Message)" -ForegroundColor Yellow
}

# 6. Executive summary
$OverallEndTime = Get-Date
$OverallElapsed = $OverallEndTime - $OverallStartTime
$OverallElapsedStr = "{0:D2}h {1:D2}m {2:D2}s" -f $OverallElapsed.Hours, $OverallElapsed.Minutes, $OverallElapsed.Seconds

Write-Host "`n====================================================================" -ForegroundColor Cyan
Write-Host "  OVERNIGHT TRAINING SUMMARY" -ForegroundColor Cyan
Write-Host "  Total elapsed time: $OverallElapsedStr" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan

$SummaryReport | Format-Table -AutoSize

Write-Host "Overnight execution finished at $($OverallEndTime.ToString('yyyy-MM-dd HH:mm:ss'))." -ForegroundColor Cyan
