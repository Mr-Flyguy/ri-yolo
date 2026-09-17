# ==============================================================================
# RI-YOLO: Script to resolve C2 reviewer findings (run_c2_fixes.ps1)
# ==============================================================================
# 1. Prints verbatim args.yaml values for all E7' runs (both runs/ and runs_fixed/)
# 2. Computes corr_LY diagnostics for:
#      - e7p__nwd-calib-norsl__s0 (genuine measurement, fixing copy-paste error)
#      - e6p__lam-1e-4__s1 (multi-seed live gate diagnostic)
#      - e6p__lam-1e-4__s2 (multi-seed live gate diagnostic)
#      - e7p__nwd-sizegate__s0 in runs_fixed (fixed sizegate checkpoint)
# 3. Computes AP breakdown by object size (eval_by_size.py) for:
#      - e7p__nwd-calib-norsl__s0
#      - e7p__nwd-sizegate__s0 (runs_fixed)
# 4. Re-collects C2 tables (C2_diag.csv, C2_table1.csv, C2_table2.csv, C2_rsl_curves.csv)
# ==============================================================================

[CmdletBinding()]
param (
    [string]$Data = "exdark.yaml",
    [string]$Device = "0"
)

Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "  RI-YOLO: C2 FIXES & DIAGNOSTICS COMPLETION" -ForegroundColor Cyan
Write-Host "  Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan

# ------------------------------------------------------------------------------
# STEP 1: Verbatim args.yaml inspection for E7' series
# ------------------------------------------------------------------------------
Write-Host "`n>>> STEP 1: VERBATIM args.yaml INSPECTION" -ForegroundColor Yellow

$e7Runs = @("nwd-calib", "nwd-scaleinv", "nwd-sizegate", "nwd-calib-norsl")
foreach ($r in $e7Runs) {
    $p = "runs/detect/runs/e7p__${r}__s0/args.yaml"
    if (-not (Test-Path $p)) {
        $p = "runs/e7p__${r}__s0/args.yaml"
    }
    Write-Host "`n=== runs/ : e7p__${r}__s0 ===" -ForegroundColor Green
    if (Test-Path $p) {
        Select-String -Path $p -Pattern "lambda_tv|use_rsl|use_nwd|nwd_mode|nwd_c|size_tau"
    } else {
        Write-Host "[WARN] $p not found" -ForegroundColor Red
    }
}

Write-Host "`n=== runs_fixed/ : e7p__nwd-sizegate__s0 ===" -ForegroundColor Green
$fixedArgs = Get-ChildItem -Recurse -Filter "args.yaml" -Path "runs_fixed" -ErrorAction SilentlyContinue
if ($fixedArgs) {
    foreach ($fa in $fixedArgs) {
        Write-Host "File: $($fa.FullName)"
        Select-String -Path $fa.FullName -Pattern "lambda_tv|use_rsl|use_nwd|nwd_mode|nwd_c|size_tau"
    }
} else {
    Write-Host "[WARN] No args.yaml found in runs_fixed" -ForegroundColor Red
}

# ------------------------------------------------------------------------------
# STEP 2: Run corr_LY diagnostics on missing & fixed checkpoints
# ------------------------------------------------------------------------------
Write-Host "`n>>> STEP 2: CORRELATION & ILLUMINATION DIAGNOSTICS (corr_LY.py)" -ForegroundColor Yellow

$diagTasks = @(
    @{
        Ckpt = "runs/detect/runs/e7p__nwd-calib-norsl__s0/weights/best.pt"
        OutCsv = "artifacts/phase2/diag/rho_e7p__nwd-calib-norsl__s0.csv"
        OutFig = "artifacts/phase2/diag/Lmaps_e7p__nwd-calib-norsl__s0.png"
        Desc = "NWD calib without RSL (genuine live gate diagnostic)"
    },
    @{
        Ckpt = "runs/detect/runs/e6p__lam-1e-4__s1/weights/best.pt"
        OutCsv = "artifacts/phase2/diag/rho_e6p__lam-1e-4__s1.csv"
        OutFig = "artifacts/phase2/diag/Lmaps_e6p__lam-1e-4__s1.png"
        Desc = "Partially live gate multi-seed seed 1"
    },
    @{
        Ckpt = "runs/detect/runs/e6p__lam-1e-4__s2/weights/best.pt"
        OutCsv = "artifacts/phase2/diag/rho_e6p__lam-1e-4__s2.csv"
        OutFig = "artifacts/phase2/diag/Lmaps_e6p__lam-1e-4__s2.png"
        Desc = "Partially live gate multi-seed seed 2"
    },
    @{
        Ckpt = "runs_fixed/e7p__nwd-sizegate__s0/weights/best.pt"
        OutCsv = "artifacts/phase2/diag/rho_e7p__nwd-sizegate__s0.csv"
        OutFig = "artifacts/phase2/diag/Lmaps_e7p__nwd-sizegate__s0.png"
        Desc = "Fixed sizegate checkpoint in runs_fixed"
    }
)

foreach ($dt in $diagTasks) {
    Write-Host "`n--------------------------------------------------------------------"
    Write-Host ">>> Running corr_LY for: $($dt.Desc)" -ForegroundColor Cyan
    Write-Host "    Ckpt: $($dt.Ckpt)"
    Write-Host "--------------------------------------------------------------------"
    python scripts/corr_LY.py `
        --ckpt $dt.Ckpt `
        --images $Data `
        --n 300 `
        --out-csv $dt.OutCsv `
        --out-fig $dt.OutFig
}

# ------------------------------------------------------------------------------
# STEP 3: Object scale breakdown (eval_by_size.py)
# ------------------------------------------------------------------------------
Write-Host "`n>>> STEP 3: EVALUATION BY OBJECT SIZE (eval_by_size.py)" -ForegroundColor Yellow

python scripts/eval_by_size.py `
    --ckpts "runs/detect/runs/e7p__nwd-calib-norsl__s0/weights/best.pt" "runs_fixed/e7p__nwd-sizegate__s0/weights/best.pt" `
    --data $Data `
    --split val `
    --device $Device `
    --out "tables/C2_table2_bysize.csv"

if (Test-Path "tables/C2_table2_bysize.csv") {
    Copy-Item -Path "tables/C2_table2_bysize.csv" -Destination "artifacts/phase2/C2_table2_bysize.csv" -Force
}

# ------------------------------------------------------------------------------
# STEP 4: Regenerate aggregated tables
# ------------------------------------------------------------------------------
Write-Host "`n>>> STEP 4: REGENERATING C2 TABLES" -ForegroundColor Yellow

python scripts/collect_c2_diag.py
python scripts/collect_c2.py
python scripts/export_c2_rsl_curves.py

Write-Host "`n====================================================================" -ForegroundColor Cyan
Write-Host "  C2 FIXES COMPLETED: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
