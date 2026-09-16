# ==============================================================================
# RI-YOLO: Block 2 (Diagnostics & Evaluations without training) + D.0.6
# ==============================================================================

[CmdletBinding()]
param (
    [string]$Data = "exdark.yaml",
    [string]$Device = "0"
)

Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "  RI-YOLO: BLOCK 2 DIAGNOSTICS & D.0.6 PROTOTYPE IDENTIFICATION" -ForegroundColor Cyan
Write-Host "  Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan

# ------------------------------------------------------------------------------
# D.0.6: Prototype Identification (runs/detect/runs/ablation_study)
# ------------------------------------------------------------------------------
Write-Host "`n====================================================================" -ForegroundColor Magenta
Write-Host ">>> D.0.6: IDENTIFICATION OF PROTOTYPES (ablation_study)" -ForegroundColor Magenta
Write-Host "====================================================================" -ForegroundColor Magenta

$ablationDir = "runs/detect/runs/ablation_study"
if (Test-Path $ablationDir) {
    foreach ($d in Get-ChildItem $ablationDir -Directory) {
        Write-Host "=== $($d.Name)" -ForegroundColor Yellow
        $argsPath = Join-Path $d.FullName "args.yaml"
        if (Test-Path $argsPath) {
            Get-Content $argsPath | Select-String -Pattern "^(model|data|epochs|batch|imgsz|seed|deterministic|close_mosaic|optimizer|lr0|lrf|use_rsl|use_nwd|lambda_tv|nwd_alpha|nwd_c|nwd_mode):"
        }
    }

    Write-Host "`n--- Metrics for 1_baseline_yolov8s ---" -ForegroundColor Yellow
    $baseCsv = "$ablationDir/1_baseline_yolov8s/results.csv"
    if (Test-Path $baseCsv) {
        python -c "import pandas as pd; d=pd.read_csv('$baseCsv'); d.columns=[c.strip() for c in d.columns]; f=0.1*d['metrics/mAP50(B)']+0.9*d['metrics/mAP50-95(B)']; i=f.idxmax(); print('best-by-fitness:', round(d['metrics/mAP50(B)'][i],4), round(d['metrics/mAP50-95(B)'][i],4), 'epoch', int(d['epoch'][i])); print('max mAP50:', round(d['metrics/mAP50(B)'].max(),4)); print('last:', round(d['metrics/mAP50(B)'].iloc[-1],4))"
    }
} else {
    Write-Host "[WARN] Ablation directory '$ablationDir' not found." -ForegroundColor Red
}

# ------------------------------------------------------------------------------
# CALC-2.1: Correlation rho_LY and L-maps (7 Phase 2 ckpts + lam0 anchor)
# ------------------------------------------------------------------------------
Write-Host "`n====================================================================" -ForegroundColor Magenta
Write-Host ">>> CALC-2.1: ILLUMINATION MAP CORRELATION (corr_LY.py)" -ForegroundColor Magenta
Write-Host "====================================================================" -ForegroundColor Magenta

New-Item -ItemType Directory -Force -Path "artifacts/phase2/diag" | Out-Null

$diagRuns = @(
    "e6p__lam-1e-4__s0",
    "e6p__lam-1e-3__s0",
    "e6p__lam-1e-2__s0",
    "e6p__lam-1e-1__s0",
    "e7p__nwd-calib__s0",
    "e7p__nwd-scaleinv__s0",
    "e7p__nwd-sizegate__s0"
)

# Helper function to find best.pt for a run
function Get-BestPt($runName) {
    $candidates = @(
        "runs/detect/runs/$runName/weights/best.pt",
        "runs/$runName/weights/best.pt"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

foreach ($r in $diagRuns) {
    Write-Host "`n--- CALC-2.1: $r ---" -ForegroundColor Green
    $ckpt = Get-BestPt $r
    if ($ckpt) {
        $outCsv = "artifacts/phase2/diag/rho_$r.csv"
        $outFig = "artifacts/phase2/diag/Lmaps_$r.png"
        if ((Test-Path $outCsv) -and (Test-Path $outFig)) {
            Write-Host "[SKIP] Already computed $outCsv and $outFig" -ForegroundColor Gray
            continue
        }
        Write-Host "[INFO] Running corr_LY.py for $r using $ckpt..." -ForegroundColor Gray
        Write-Host "[INFO] Loading PyTorch CUDA libraries takes ~5-10s on Windows, please wait..." -ForegroundColor Gray
        python scripts/corr_LY.py --ckpt $ckpt --images $Data --n 300 --out-csv $outCsv --out-fig $outFig
    } else {
        Write-Host "[ERROR] Checkpoint for $r not found!" -ForegroundColor Red
    }
}

# lam0 anchor (Phase 1 e3p__pos-postsppf__s0)
Write-Host "`n--- CALC-2.1: lam0 anchor (e3p__pos-postsppf__s0) ---" -ForegroundColor Green
$lam0Ckpt = Get-BestPt "e3p__pos-postsppf__s0"
if ($lam0Ckpt) {
    $outCsv = "artifacts/phase2/diag/rho_lam0.csv"
    $outFig = "artifacts/phase2/diag/Lmaps_lam0.png"
    if ((Test-Path $outCsv) -and (Test-Path $outFig)) {
        Write-Host "[SKIP] Already computed $outCsv and $outFig" -ForegroundColor Gray
    } else {
        Write-Host "[INFO] Running corr_LY.py for lam0 using $lam0Ckpt..." -ForegroundColor Gray
        python scripts/corr_LY.py --ckpt $lam0Ckpt --images $Data --n 300 --out-csv $outCsv --out-fig $outFig
    }
} else {
    Write-Host "[ERROR] Checkpoint for e3p__pos-postsppf__s0 not found!" -ForegroundColor Red
}

# ------------------------------------------------------------------------------
# CALC-2.2: Diagnostic summary table (collect_c2_diag.py)
# ------------------------------------------------------------------------------
Write-Host "`n====================================================================" -ForegroundColor Magenta
Write-Host ">>> CALC-2.2: DIAGNOSTICS SUMMARY TABLE (tables/C2_diag.csv)" -ForegroundColor Magenta
Write-Host "====================================================================" -ForegroundColor Magenta

python scripts/collect_c2_diag.py --diag-dir "artifacts/phase2/diag" --out "tables/C2_diag.csv"

# ------------------------------------------------------------------------------
# CALC-2.3: Gradient figure (fig_gradients.py)
# ------------------------------------------------------------------------------
Write-Host "`n====================================================================" -ForegroundColor Magenta
Write-Host ">>> CALC-2.3: LOCALIZATION GRADIENT CURVES (figs/C2_fig1_gradients.png)" -ForegroundColor Magenta
Write-Host "====================================================================" -ForegroundColor Magenta

python scripts/fig_gradients.py --C 12.8 --C-calib 10.24 --out "figs/C2_fig1_gradients.png"

# ------------------------------------------------------------------------------
# CALC-2.4: Object size breakdown (eval_by_size.py)
# ------------------------------------------------------------------------------
Write-Host "`n====================================================================" -ForegroundColor Magenta
Write-Host ">>> CALC-2.4: EVALUATION BY OBJECT SIZE (tables/C2_table2_bysize.csv)" -ForegroundColor Magenta
Write-Host "====================================================================" -ForegroundColor Magenta

$sizeCkpts = @(
    "e3p__pos-postsppf__s0",
    "e6p__lam-1e-2__s0",
    "e7p__nwd-calib__s0",
    "e7p__nwd-scaleinv__s0",
    "e7p__nwd-sizegate__s0"
)
$resolvedSizeCkpts = @()
foreach ($sc in $sizeCkpts) {
    $pt = Get-BestPt $sc
    if ($pt) {
        $resolvedSizeCkpts += $pt
    } else {
        Write-Host "[WARN] Checkpoint for $sc not found!" -ForegroundColor Yellow
    }
}

if ($resolvedSizeCkpts.Count -gt 0) {
    python scripts/eval_by_size.py --ckpts $resolvedSizeCkpts --data $Data --split val --device $Device --out "tables/C2_table2_bysize.csv"
} else {
    Write-Host "[ERROR] No size checkpoints found to evaluate!" -ForegroundColor Red
}

# ------------------------------------------------------------------------------
# CALC-2.5: Test split evaluation (eval_test_split.py) if test split is present
# ------------------------------------------------------------------------------
Write-Host "`n====================================================================" -ForegroundColor Magenta
Write-Host ">>> CALC-2.5: HELD-OUT TEST SPLIT EVALUATION (tables/C3_test_generalization.csv)" -ForegroundColor Magenta
Write-Host "====================================================================" -ForegroundColor Magenta

$testImagesDir = "datasets/ExDark/images/test"
if (Test-Path $testImagesDir) {
    Write-Host "[INFO] Test set found in $testImagesDir. Running test generalisation..." -ForegroundColor Green
    python scripts/eval_test_split.py --data $Data --split test --ckpts runs/detect/runs/*/weights/best.pt --device $Device --out "tables/C3_test_generalization.csv"
} else {
    Write-Host "[NOTE] Test set not yet deployed to $testImagesDir. Skipping CALC-2.5 for now." -ForegroundColor Yellow
}

Write-Host "`n====================================================================" -ForegroundColor Cyan
Write-Host "  BLOCK 2 EXECUTION COMPLETE" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
