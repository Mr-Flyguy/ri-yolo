import pandas as pd
from pathlib import Path
import sys

def check_consistency():
    import sys
    Path("artifacts/phase2").mkdir(parents=True, exist_ok=True)
    sys.stdout = open("artifacts/phase2/g6_consistency_report.txt", "w", encoding="utf-8")
    print("="*80)
    print("  G.6: CONSISTENCY CHECKER (C2 TABLES)")
    print("="*80)

    tables = {
        "diag": Path("tables/C2_final_diag.csv"),
        "t1": Path("tables/C2_table1.csv"),
        "bs": Path("tables/C2_branch_scale.csv"),
        "metrics": Path("tables/C2_metrics_unified.csv"),
    }

    dfs = {}
    for k, p in tables.items():
        if p.exists():
            dfs[k] = pd.read_csv(p)
            print(f"[INFO] Loaded {p.name}")
        else:
            print(f"[ERROR] Missing {p.name}!")
            sys.exit(1)

    errors = 0

    # 1. mAP50 / mAP50_95 : C2_table1 vs C2_metrics_unified
    print("\n--- CHECK 1: mAP metrics (table1 vs metrics_unified) ---")
    t1_runs = dfs["t1"]["run"].tolist()
    for run in t1_runs:
        t1_row = dfs["t1"][dfs["t1"]["run"] == run].iloc[0]
        met_row = dfs["metrics"][dfs["metrics"]["run"] == run]
        
        if len(met_row) == 0:
            print(f"[WARN] {run} missing in C2_metrics_unified.csv")
            continue
            
        met_row = met_row.iloc[-1]  # or only row
        
        for met in ["mAP50", "mAP50_95"]:
            v1 = float(t1_row[met])
            v2 = float(met_row[met])
            if abs(v1 - v2) > 1e-4:
                print(f"[ERROR] {run} {met} mismatch: C2_table1={v1} != C2_metrics_unified={v2}")
                errors += 1

    # 2. L_mean, L_std, gamma_final : C2_table1 vs C2_final_diag
    print("\n--- CHECK 2: L_mean, L_std, gamma_final (table1 vs final_diag) ---")
    for run in t1_runs:
        t1_row = dfs["t1"][dfs["t1"]["run"] == run].iloc[0]
        diag_row = dfs["diag"][dfs["diag"]["run"] == run]
        
        if len(diag_row) == 0:
            print(f"[WARN] {run} missing in C2_final_diag.csv")
            continue
        diag_row = diag_row.iloc[0]
        
        for col in ["L_mean", "L_std", "gamma_final"]:
            if str(t1_row[col]) == "NA" or str(diag_row[col]) == "NA":
                continue
            v1 = float(t1_row[col])
            v2 = float(diag_row[col])
            if abs(v1 - v2) > 1e-4:
                print(f"[ERROR] {run} {col} mismatch: C2_table1={v1} != C2_final_diag={v2}")
                errors += 1

    # 3. L_mean, L_std, gamma_final : C2_branch_scale vs C2_final_diag
    print("\n--- CHECK 3: L_mean, L_std, gamma_final (branch_scale vs final_diag) ---")
    bs_runs = dfs["bs"]["run"].tolist()
    for run in bs_runs:
        bs_row = dfs["bs"][dfs["bs"]["run"] == run].iloc[0]
        diag_row = dfs["diag"][dfs["diag"]["run"] == run]
        
        if len(diag_row) == 0:
            print(f"[WARN] {run} missing in C2_final_diag.csv")
            continue
        diag_row = diag_row.iloc[0]
        
        for col in ["L_mean", "L_std", "gamma_final"]:
            v1 = float(bs_row[col])
            v2 = float(diag_row[col])
            if abs(v1 - v2) > 1e-4:
                print(f"[ERROR] {run} {col} mismatch: C2_branch_scale={v1} != C2_final_diag={v2}")
                errors += 1

    if errors == 0:
        print("\n[SUCCESS] All data consistency checks passed!")
    else:
        print(f"\n[FAILED] Found {errors} data inconsistencies. Please fix the generation scripts.")

if __name__ == "__main__":
    check_consistency()
