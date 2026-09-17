import pandas as pd
from pathlib import Path

def check_consistency():
    print("=" * 80)
    print("  STAGE H: C2 CONSISTENCY CHECKER")
    print("=" * 80)
    
    # Load all tables
    tables = {
        "t1": pd.read_csv("tables/C2_table1.csv"),
        "fm": pd.read_csv("tables/C2_final_metrics.csv"),
        "fd": pd.read_csv("tables/C2_final_diag.csv"),
        "bs": pd.read_csv("tables/C2_branch_scale.csv"),
        "hyp": pd.read_csv("tables/C2_hyperparams.csv"),
        "rsl": pd.read_csv("tables/C2_rsl_curves.csv"),
        "gamma": pd.read_csv("tables/C2_gamma_curves.csv")
    }
    
    hyp_runs = set(tables["hyp"]["run"].unique())
    # Add mapped names for rsl_curves which uses broken/fixed aliases
    hyp_runs.add("e7p__nwd-sizegate__s0__broken")
    hyp_runs.add("e7p__nwd-sizegate__s0__fixed")
    
    discrepancies = 0
    
    def report(msg, is_err=True):
        nonlocal discrepancies
        print(f"[{'FAIL' if is_err else 'OK'}] {msg}")
        if is_err:
            discrepancies += 1

    def almost_equal(a, b, tol=1e-4):
        if pd.isna(a) and pd.isna(b): return True
        if pd.isna(a) or pd.isna(b): return False
        try:
            return abs(float(a) - float(b)) <= tol
        except:
            return str(a).strip() == str(b).strip()

    # 1. mAP50, mAP50_95: C2_table1.csv vs C2_final_metrics.csv
    print("\n--- 1. Checking mAP50 & mAP50_95 (table1 vs final_metrics) ---")
    for _, r in tables["t1"].iterrows():
        run = r["run"]
        fm_row = tables["fm"][tables["fm"]["run"] == run]
        if fm_row.empty:
            report(f"{run} missing in final_metrics")
            continue
        fm_row = fm_row.iloc[0]
        for metric in ["mAP50", "mAP50_95"]:
            v1 = r[metric]
            v2 = fm_row[metric]
            if not almost_equal(v1, v2):
                report(f"{metric} mismatch for {run}: table1={v1}, final_metrics={v2}")
    
    # 2. L_mean, L_std, gamma_final, mean_one_minus_L: C2_table1 & C2_branch_scale vs C2_final_diag
    print("\n--- 2. Checking L_metrics (table1 & branch_scale vs final_diag) ---")
    for _, r in tables["fd"].iterrows():
        run = r["run"]
        # table1
        t1_row = tables["t1"][tables["t1"]["run"] == run]
        if not t1_row.empty:
            t1_row = t1_row.iloc[0]
            for metric in ["L_mean", "L_std", "gamma_final"]:
                v_diag = r[metric]
                v_t1 = t1_row[metric]
                if not almost_equal(v_t1, v_diag):
                    report(f"{metric} mismatch for {run}: table1={v_t1}, final_diag={v_diag}")
        
        # branch_scale
        bs_row = tables["bs"][tables["bs"]["run"] == run]
        if not bs_row.empty:
            bs_row = bs_row.iloc[0]
            for metric in ["L_mean", "L_std", "gamma_final", "mean_one_minus_L"]:
                v_diag = r[metric]
                v_bs = bs_row[metric]
                if not almost_equal(v_bs, v_diag):
                    report(f"{metric} mismatch for {run}: branch_scale={v_bs}, final_diag={v_diag}")
                    
    # 3. lambda_tv: C2_table1, C2_branch_scale, C2_rsl_curves vs C2_hyperparams
    print("\n--- 3. Checking lambda_tv vs hyperparams ---")
    for table_name, df in [("table1", tables["t1"]), ("branch_scale", tables["bs"]), ("rsl_curves", tables["rsl"])]:
        if "lambda_tv" in df.columns:
            for _, r in df.iterrows():
                run = r["run"]
                
                # Map aliases to hyperparams run names
                hyp_run_name = run
                if run == "e7p__nwd-sizegate__s0__broken":
                    hyp_run_name = "runs/detect/runs/e7p__nwd-sizegate__s0"
                elif run == "e7p__nwd-sizegate__s0__fixed":
                    hyp_run_name = "runs_fixed/e7p__nwd-sizegate__s0"
                    
                v_hyp = tables["hyp"][tables["hyp"]["run"] == hyp_run_name]["lambda_tv"]
                if v_hyp.empty:
                    continue
                v_hyp = v_hyp.iloc[0]
                v_df = r["lambda_tv"]
                if not almost_equal(v_df, v_hyp):
                    report(f"lambda_tv mismatch for {run} in {table_name}: {table_name}={v_df}, hyperparams={v_hyp}")
                    
    # 4. No tables have runs missing in C2_hyperparams.csv
    print("\n--- 4. Checking for missing runs in hyperparams ---")
    for table_name, df in tables.items():
        if "run" in df.columns:
            missing = set(df["run"].unique()) - hyp_runs
            missing = {m for m in missing if "baseline" not in m and m != "runs"}
            if missing:
                for m in missing:
                    report(f"Run '{m}' found in {table_name} but missing in hyperparams")

    # 5. C2_gamma_curves.csv has no baseline runs
    print("\n--- 5. Checking C2_gamma_curves.csv for baseline runs ---")
    base_in_gamma = tables["gamma"][tables["gamma"]["run"].str.contains("baseline", case=False, na=False)]
    if not base_in_gamma.empty:
        for run in base_in_gamma["run"].unique():
            report(f"Baseline run {run} found in C2_gamma_curves.csv")
            
    print("=" * 80)
    if discrepancies == 0:
        print(f"[SUCCESS] Consistency check passed! 0 discrepancies found.")
    else:
        print(f"[ERROR] Found {discrepancies} discrepancies across tables.")
    print("=" * 80)

if __name__ == "__main__":
    check_consistency()
