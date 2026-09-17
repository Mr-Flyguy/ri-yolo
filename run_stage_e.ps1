# ==============================================================================
# RI-YOLO: Stage E Complete Execution Runner (run_stage_e.ps1)
# ==============================================================================
# Executes all remaining requirements for Article C2:
#   E.2: Compare box_loss epochs 91-100 (broken vs fixed) and capture [FG] anchors N
#   E.3 & E.4: Complete size breakdown (eval_by_size.py) with n_small, n_medium, n_large
#              including e6p__lam-1e-4__s0/s1/s2 and e2p__postsppf__s1/s2/s3
#   E.6: Map diagnostics (corr_LY.py) for postsppf seeds 1, 2, 3
#   E.7: Check rfd_log.csv dynamics
#   E.8: Generate branch scale and unified tables
# ==============================================================================

[CmdletBinding()]
param (
    [string]$Data = "exdark.yaml",
    [string]$Device = "0"
)

Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "  RI-YOLO: STAGE E AUTOMATED RUNNER (KT-10)" -ForegroundColor Cyan
Write-Host "  Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan

# ------------------------------------------------------------------------------
# 1. E.2: Compare train/box_loss (epochs 91-100) for broken vs fixed sizegate
# ------------------------------------------------------------------------------
Write-Host "`n>>> [E.2] COMPARING BOX LOSS (EPOCHS 91-100): BROKEN VS FIXED" -ForegroundColor Yellow

$brokenCsv = "runs/detect/runs/e7p__nwd-sizegate__s0/results.csv"
$fixedCsv = "runs/detect/runs_fixed/e7p__nwd-sizegate__s0/results.csv"
if (-not (Test-Path $fixedCsv)) { $fixedCsv = "runs_fixed/e7p__nwd-sizegate__s0/results.csv" }

python -c "
import pandas as pd
p_broken = '$brokenCsv'
p_fixed = '$fixedCsv'

try:
    df_b = pd.read_csv(p_broken)
    df_b.columns = [c.strip() for c in df_b.columns]
    b_loss = df_b[df_b['epoch'] >= 90]['train/box_loss'].mean()
except Exception as e:
    b_loss = 'NA (' + str(e) + ')'

try:
    df_f = pd.read_csv(p_fixed)
    df_f.columns = [c.strip() for c in df_f.columns]
    f_loss = df_f[df_f['epoch'] >= 90]['train/box_loss'].mean()
except Exception as e:
    f_loss = 'NA (' + str(e) + ')'

if isinstance(b_loss, float) and isinstance(f_loss, float) and f_loss > 0:
    ratio = b_loss / f_loss
    print(f'[BOX LOSS 91-100] broken = {b_loss:.4f}, fixed = {f_loss:.4f}, ratio (inflation factor) = {ratio:.2f}x')
else:
    print(f'[BOX LOSS 91-100] broken = {b_loss}, fixed = {f_loss}')
"

# ------------------------------------------------------------------------------
# 2. E.2: Run 1-epoch debug run to capture positive foreground anchors N
# ------------------------------------------------------------------------------
Write-Host "`n>>> [E.2] CAPTURING FOREGROUND ANCHORS COUNT (N) ON 1 EPOCH" -ForegroundColor Yellow
if (-not (Test-Path "runs_debug")) { New-Item -ItemType Directory -Path "runs_debug" | Out-Null }

python scripts/train_phase2.py `
    --run e7p__nwd-sizegate__s0 `
    --epochs 1 `
    --project runs_debug `
    --data $Data `
    --device $Device `
    --weights weights/yolov8s.pt 2>&1 | Tee-Object -Variable debugOut

$fgLine = ($debugOut | Select-String "\[FG\] positive anchors N = \d+")
if ($fgLine) {
    Write-Host "[RESULT] $fgLine" -ForegroundColor Green
} else {
    Write-Host "[WARN] [FG] line not captured in output." -ForegroundColor Red
}

# ------------------------------------------------------------------------------
# 3. E.6: Illumination map diagnostics for postsppf seeds 1, 2, 3
# ------------------------------------------------------------------------------
Write-Host "`n>>> [E.6] CORRELATION DIAGNOSTICS FOR POSTSPPF SEEDS (corr_LY.py)" -ForegroundColor Yellow

$postsppfSeeds = @("s1", "s2", "s3")
foreach ($s in $postsppfSeeds) {
    $ck = "runs/detect/runs/e2p__postsppf__${s}/weights/best.pt"
    if (Test-Path $ck) {
        Write-Host "`n--- Running corr_LY for: e2p__postsppf__${s} ---" -ForegroundColor Cyan
        python scripts/corr_LY.py `
            --ckpt $ck `
            --images $Data `
            --n 300 `
            --out-csv "artifacts/phase2/diag/rho_e2p__postsppf__${s}.csv" `
            --out-fig "artifacts/phase2/diag/Lmaps_e2p__postsppf__${s}.png"
    } else {
        Write-Host "[WARN] Checkpoint $ck not found, skipping." -ForegroundColor Red
    }
}

# ------------------------------------------------------------------------------
# 4. E.3 & E.4: Complete evaluation by object size (eval_by_size.py)
# ------------------------------------------------------------------------------
Write-Host "`n>>> [E.3 & E.4] COMPLETE EVALUATION BY OBJECT SIZE (eval_by_size.py)" -ForegroundColor Yellow

$sizeCkpts = @(
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

$validCkpts = @()
foreach ($ck in $sizeCkpts) {
    if (Test-Path $ck) {
        $validCkpts += $ck
    }
}

Write-Host "Found $($validCkpts.Count) valid checkpoints for size evaluation."
python scripts/eval_by_size.py `
    --ckpts $validCkpts `
    --data $Data `
    --split val `
    --device $Device `
    --out "tables/C2_table2_bysize.csv"

if (Test-Path "tables/C2_table2_bysize.csv") {
    Copy-Item -Path "tables/C2_table2_bysize.csv" -Destination "artifacts/phase2/C2_table2_bysize.csv" -Force
}

# ------------------------------------------------------------------------------
# 5. E.1 & E.2: Regenerate RSL curves with broken and fixed sizegate
# ------------------------------------------------------------------------------
Write-Host "`n>>> [E.1 & E.2] EXPORTING RSL CURVES (export_c2_rsl_curves.py)" -ForegroundColor Yellow
python scripts/export_c2_rsl_curves.py

# ------------------------------------------------------------------------------
# 6. E.5, E.7, E.8: Build consolidated Stage E tables (build_c2_stage_e.py)
# ------------------------------------------------------------------------------
Write-Host "`n>>> [E.5, E.7, E.8] BUILDING STAGE E TABLES (build_c2_stage_e.py)" -ForegroundColor Yellow
python scripts/build_c2_stage_e.py

Write-Host "`n====================================================================" -ForegroundColor Cyan
Write-Host "  STAGE E EXECUTION COMPLETED: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
