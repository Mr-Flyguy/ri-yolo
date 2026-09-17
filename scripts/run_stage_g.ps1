[CmdletBinding()]
param ()

Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "  RI-YOLO: RUNNING STAGE G AUDITS & REBUILDS" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan

# 1. Run G.3 diagnostics
Write-Host "`n>>> [1/4] AUDITING RSL LOSS (G.3)" -ForegroundColor Yellow
python scripts/check_g3.py
Write-Host "[SUCCESS] G.3 audit written to artifacts/phase2/g3_rsl_audit.txt" -ForegroundColor Green

# 2. Re-run Diagnostics (G.4)
Write-Host "`n>>> [2/4] RECALCULATING FINAL DIAGNOSTICS (G.4)" -ForegroundColor Yellow
python scripts/final_diag.py

# 3. Rebuild Stage E tables (G.1 & G.2)
Write-Host "`n>>> [3/4] REBUILDING C2 TABLES (G.1 & G.2)" -ForegroundColor Yellow
python scripts/build_c2_stage_e.py

# 4. Consistency Checks (G.6)
Write-Host "`n>>> [4/4] RUNNING CONSISTENCY CHECKER (G.6)" -ForegroundColor Yellow
python scripts/consistency_check.py
Write-Host "[SUCCESS] G.6 consistency report written to artifacts/phase2/g6_consistency_report.txt" -ForegroundColor Green

# 5. Git Commit and Push
Write-Host "`n>>> COMMITTING RESULTS..." -ForegroundColor Yellow
git add tables/ artifacts/phase2/
git commit -m "chore(stage-g): run audits, rebuild tables, and generate consistency reports"
git push origin recovery-v2
Write-Host "[SUCCESS] Pushed updates to recovery-v2!" -ForegroundColor Green
