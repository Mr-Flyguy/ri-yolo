# ==============================================================================
# RI-YOLO: Phase 3 Series E5' Training Runner (run_e5.ps1)
# ==============================================================================
# Executes 4 training runs sequentially for Series E5' (Duration Ablation):
#   1. e5p__baseline-ep40__s0 (YOLOv8s baseline, 40 epochs, close_mosaic=10, seed 0)
#   2. e5p__baseline-ep60__s0 (YOLOv8s baseline, 60 epochs, close_mosaic=10, seed 0)
#   3. e5p__postsppf-ep40__s0 (RFD post-SPPF, 40 epochs, close_mosaic=10, seed 0)
#   4. e5p__postsppf-ep60__s0 (RFD post-SPPF, 60 epochs, close_mosaic=10, seed 0)
# ==============================================================================

[CmdletBinding()]
param (
    [string]$Data = "exdark.yaml",
    [string]$Device = "0",
    [string]$Weights = "weights/yolov8s.pt"
)

Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "  RI-YOLO: PHASE 3 SERIES E5' TRAINING QUEUE (4 RUNS)" -ForegroundColor Cyan
Write-Host "  Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan

# 1. Directory verification
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
    Write-Host "[INIT] Created directory logs/" -ForegroundColor Yellow
}
if (-not (Test-Path "artifacts/phase3")) {
    New-Item -ItemType Directory -Path "artifacts/phase3" -Force | Out-Null
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
    @{ Name = "e5p__baseline-ep40__s0"; Desc = "YOLOv8s baseline (40 epochs, seed 0)" },
    @{ Name = "e5p__baseline-ep60__s0"; Desc = "YOLOv8s baseline (60 epochs, seed 0)" },
    @{ Name = "e5p__postsppf-ep40__s0"; Desc = "RFD post-SPPF (40 epochs, seed 0)" },
    @{ Name = "e5p__postsppf-ep60__s0"; Desc = "RFD post-SPPF (60 epochs, seed 0)" }
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
        $cmd = "python scripts/train_phase3.py --run $runName --data $Data --device $Device --weights $Weights"
        Write-Host "[CMD] $cmd" -ForegroundColor Gray

        Invoke-Expression "$cmd 2>&1" | Tee-Object -FilePath $logFile
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            Write-Host "`n[POST-CHECK] Training $runName completed with code 0. Running verification..." -ForegroundColor Cyan

            # Detect actual output directory (Phase 3 saves into runs/detect/runs/ or runs/)
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

# 5. Final check of artifacts/phase3/summary.csv
Write-Host "`n====================================================================" -ForegroundColor Cyan
Write-Host "  PHASE 3 SUMMARY TABLE (artifacts/phase3/summary.csv)" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan

if (Test-Path "artifacts/phase3/summary.csv") {
    Get-Content "artifacts/phase3/summary.csv"
}

# 6. Executive summary
$OverallEndTime = Get-Date
$OverallElapsed = $OverallEndTime - $OverallStartTime
$OverallElapsedStr = "{0:D2}h {1:D2}m {2:D2}s" -f $OverallElapsed.Hours, $OverallElapsed.Minutes, $OverallElapsed.Seconds

Write-Host "`n====================================================================" -ForegroundColor Cyan
Write-Host "  SERIES E5' EXECUTION SUMMARY" -ForegroundColor Cyan
Write-Host "  Total elapsed time: $OverallElapsedStr" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan

$SummaryReport | Format-Table -AutoSize

Write-Host "Series E5' finished at $($OverallEndTime.ToString('yyyy-MM-dd HH:mm:ss'))." -ForegroundColor Cyan
