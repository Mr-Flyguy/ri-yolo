# ==============================================================================
# RI-YOLO: Stage F Master Automation Script (run_stage_f.ps1)
# ==============================================================================
[CmdletBinding()]
param (
    [string]$Data = "exdark.yaml",
    [string]$Device = "0"
)

Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "  RI-YOLO: RUNNING STAGE F (DATA INTEGRITY & UNIFIED RE-EVALUATION)" -ForegroundColor Cyan
Write-Host "  Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan

# ------------------------------------------------------------------------------
# F.4: Verbatim args.yaml PowerShell inspection
# ------------------------------------------------------------------------------
Write-Host "`n>>> [1/6] F.4: VERBATIM ARGS.YAML POWERSHELL EXTRACTION" -ForegroundColor Yellow
foreach ($d in @("e6p__lam-1e-5__s0","e6p__lam-1e-4__s0","e6p__lam-1e-4__s1",
                 "e6p__lam-1e-4__s2","e6p__lam-1e-3__s0","e6p__lam-1e-2__s0",
                 "e6p__lam-1e-1__s0","e7p__nwd-calib__s0","e7p__nwd-scaleinv__s0",
                 "e7p__nwd-sizegate__s0","e7p__nwd-calib-norsl__s0")) {
  Write-Host "=== $d"
  Get-Content "runs/detect/runs/$d/args.yaml" -ErrorAction SilentlyContinue | Select-String -Pattern `
    "^(use_rsl|lambda_tv|use_nwd|nwd_alpha|nwd_mode|nwd_c|size_tau|seed|epochs):"
}
Write-Host "=== runs_fixed/e7p__nwd-sizegate__s0"
Get-Content "runs_fixed/e7p__nwd-sizegate__s0/args.yaml" -ErrorAction SilentlyContinue | Select-String -Pattern `
  "^(use_rsl|lambda_tv|use_nwd|nwd_alpha|nwd_mode|nwd_c|size_tau|seed|epochs):"

# ------------------------------------------------------------------------------
# F.4: Assemble tables/C2_hyperparams.csv directly from args.yaml
# ------------------------------------------------------------------------------
Write-Host "`n>>> [2/6] F.4: ASSEMBLING tables/C2_hyperparams.csv DIRECTLY FROM args.yaml" -ForegroundColor Yellow
python scripts/collect_hyperparams.py

# ------------------------------------------------------------------------------
# F.5: Ground truth object counts by COCO category & percentiles
# ------------------------------------------------------------------------------
Write-Host "`n>>> [3/6] F.5: COUNTING GROUND TRUTH OBJECTS BY COCO SIZE & PERCENTILES" -ForegroundColor Yellow
python scripts/count_objects_by_size.py --data $Data

# ------------------------------------------------------------------------------
# F.2: Unified final evaluation protocol across all 13 checkpoints
# ------------------------------------------------------------------------------
Write-Host "`n>>> [4/6] F.2: UNIFIED FINAL METRICS EVALUATION (final_eval.py)" -ForegroundColor Yellow
python scripts/final_eval.py --data $Data --device $Device

# ------------------------------------------------------------------------------
# F.3 & F.7: Unified illumination map diagnostics on 300 images
# ------------------------------------------------------------------------------
Write-Host "`n>>> [5/6] F.3 & F.7: UNIFIED ILLUMINATION MAP DIAGNOSTICS (final_diag.py)" -ForegroundColor Yellow
python scripts/final_diag.py --data $Data --device $Device --n 300

# ------------------------------------------------------------------------------
# F.6: Artifact reproducibility check (*_before.csv diff)
# ------------------------------------------------------------------------------
Write-Host "`n>>> [6/6] F.6: VERIFYING ARTIFACT REPRODUCIBILITY" -ForegroundColor Yellow
python scripts/check_reproducibility.py

Write-Host "`n====================================================================" -ForegroundColor Green
Write-Host "  STAGE F COMPLETE! To commit and push, run:" -ForegroundColor Green
Write-Host "  git add tables/ artifacts/phase2/ scripts/ run_stage_f.ps1" -ForegroundColor White
Write-Host "  git commit -m `"feat(c2): complete Stage F data reconciliation and unified evaluation`"" -ForegroundColor White
Write-Host "  git push origin recovery-v2" -ForegroundColor White
Write-Host "====================================================================" -ForegroundColor Green
