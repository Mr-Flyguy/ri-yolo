# ==============================================================================
# RI-YOLO: Finish Stage E size evaluation and FG print (finish_stage_e.ps1)
# ==============================================================================
[CmdletBinding()]
param (
    [string]$Data = "exdark.yaml",
    [string]$Device = "0"
)

Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "  RI-YOLO: COMPLETING SIZE METRICS & CAPTURING FG ANCHORS" -ForegroundColor Cyan
Write-Host "  Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan

# 1. Look for [FG] anchors in runs_debug or logs
Write-Host "`n>>> [1/3] CHECKING FOREGROUND ANCHORS N IN LOGS" -ForegroundColor Yellow
$fgMatches = Get-ChildItem -Recurse -Path "runs_debug", "logs" -Filter "*.log" -ErrorAction SilentlyContinue | Select-String "\[FG\] positive anchors N = \d+"
if ($fgMatches) {
    Write-Host "[SUCCESS] Found foreground anchors in logs:" -ForegroundColor Green
    foreach ($m in $fgMatches) {
        Write-Host "  $($m.Line)" -ForegroundColor Green
    }
} else {
    Write-Host "[INFO] Running 1 batch forward check to print [FG] positive anchors N..." -ForegroundColor Cyan
    python -c "
import torch
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml')
# Load pretrained weights
model.train(
    data='$Data',
    epochs=1,
    batch=16,
    imgsz=640,
    device=$Device,
    project='runs_debug',
    name='fg_probe',
    exist_ok=True,
    plots=False,
    workers=2,
)
" 2>&1 | Select-String "\[FG\] positive anchors N = \d+"
}

# 2. Run eval_by_size across all checkpoints to populate AP_small/medium/large and n_small/medium/large
Write-Host "`n>>> [2/3] POPULATING COCO SIZE METRICS (eval_by_size.py)" -ForegroundColor Yellow

$ckpts = @(
    "runs/detect/runs/e3p__pos-postsppf__s0/weights/best.pt",
    "runs/detect/runs/e2p__postsppf__s1/weights/best.pt",
    "runs/detect/runs/e2p__postsppf__s2/weights/best.pt",
    "runs/detect/runs/e2p__postsppf__s3/weights/best.pt",
    "runs/detect/runs/e6p__lam-1e-4__s0/weights/best.pt",
    "runs/detect/runs/e6p__lam-1e-4__s1/weights/best.pt",
    "runs/detect/runs/e6p__lam-1e-4__s2/weights/best.pt",
    "runs/detect/runs/e6p__lam-1e-2__s0/weights/best.pt",
    "runs/detect/runs/e7p__nwd-calib__s0/weights/best.pt",
    "runs/detect/runs/e7p__nwd-scaleinv__s0/weights/best.pt",
    "runs/detect/runs/e7p__nwd-calib-norsl__s0/weights/best.pt",
    "runs/detect/runs_fixed/e7p__nwd-sizegate__s0/weights/best.pt"
)

$existingCkpts = @()
foreach ($c in $ckpts) {
    if (Test-Path $c) { $existingCkpts += $c }
}

python scripts/eval_by_size.py `
    --ckpts $existingCkpts `
    --data $Data `
    --split val `
    --device $Device `
    --out "tables/C2_table2_bysize.csv"

if (Test-Path "tables/C2_table2_bysize.csv") {
    Copy-Item -Path "tables/C2_table2_bysize.csv" -Destination "artifacts/phase2/C2_table2_bysize.csv" -Force
}

# 3. Regenerate unified tables
Write-Host "`n>>> [3/3] REGENERATING STAGE E TABLES (build_c2_stage_e.py)" -ForegroundColor Yellow
python scripts/build_c2_stage_e.py

Write-Host "`n====================================================================" -ForegroundColor Cyan
Write-Host "  TABLES SUMMARY:" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
if (Test-Path "tables/C2_table2_bysize.csv") {
    Get-Content "tables/C2_table2_bysize.csv"
}

Write-Host "`n====================================================================" -ForegroundColor Green
Write-Host "  COMPLETED! To commit results, run:" -ForegroundColor Green
Write-Host "  git add tables/ artifacts/phase2/" -ForegroundColor White
Write-Host "  git commit -m `"feat(c2): finalize Stage E size metrics and tables`"" -ForegroundColor White
Write-Host "  git push origin recovery-v2" -ForegroundColor White
Write-Host "====================================================================" -ForegroundColor Green
