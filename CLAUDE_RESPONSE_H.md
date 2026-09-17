# ЭТАП H: ФИНАЛЬНЫЙ ОТЧЁТ И ЗАКРЫТИЕ ДАННЫХ

## H.1, H.2: Скрипт `build_c2_stage_e.py`
```python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Stage E Table Builder: Consolidates C2 tables according to KT-10 requirements.

Generates:
- tables/C2_table1.csv: Unified E6' + E7' table with broken & fixed sizegate and '*' on flat fields
- tables/C2_branch_scale.csv: Two-stage collapse branch scale analysis (mean(1-L) and gamma * mean(1-L))
- tables/C2_gamma_curves.csv: Epoch-wise L_std / gamma dynamics from rfd_log.csv (if available)
- tables/C2_metrics_unified.csv: Unified validation protocol metrics across all checkpoints
"""

import csv
import glob
import os
from pathlib import Path
import pandas as pd


def build_c2_table1():
    """Build complete C2_table1.csv containing both E6' and E7' runs."""
    metrics_p = Path("tables/C2_final_metrics.csv")
    diag_p = Path("tables/C2_final_diag.csv")
    hyperparams_p = Path("tables/C2_hyperparams.csv")
    
    if not metrics_p.exists() or not diag_p.exists() or not hyperparams_p.exists():
        print("[WARN] Missing source files (metrics, diag, or hyperparams).")
        return
        
    df_metrics = pd.read_csv(metrics_p)
    df_diag = pd.read_csv(diag_p)
    df_hyp = pd.read_csv(hyperparams_p)

    merged = pd.merge(df_hyp, df_metrics, on="run", how="inner")
    merged = pd.merge(merged, df_diag, on="run", how="inner")

    rows = []
    for _, r in merged.iterrows():
        run_name = str(r["run"])
        
        # Hyperparams
        lambda_tv = r.get("lambda_tv", "NA")
        use_rsl = r.get("use_rsl", "NA")
        use_nwd = r.get("use_nwd", "NA")
        nwd_mode = r.get("nwd_mode", "NA")
        nwd_c = r.get("nwd_c", "NA")
        size_tau = r.get("size_tau", "NA")
        
        # Metrics
        mAP50 = r.get("mAP50", "NA")
        mAP50_95 = r.get("mAP50_95", "NA")
        AP_small = r.get("AP_small", "NA")
        AP_medium = r.get("AP_medium", "NA")
        AP_large = r.get("AP_large", "NA")
        
        # Diag
        l_mean = r.get("L_mean", "NA")
        l_std = r.get("L_std", "NA")
        p_raw = r.get("pearson", "NA")
        s_raw = r.get("spearman", "NA")
        gamma_final = r.get("gamma_final", "NA")
        
        # Format metrics
        if not pd.isna(l_mean) and l_mean != "NA": l_mean = round(float(l_mean), 4)
        if not pd.isna(l_std) and l_std != "NA": l_std = round(float(l_std), 4)
        
        p_str = f"{float(p_raw):.4f}" if not pd.isna(p_raw) and p_raw != "NA" else "NA"
        s_str = f"{float(s_raw):.4f}" if not pd.isna(s_raw) and s_raw != "NA" else "NA"
        
        is_flat = False
        if not pd.isna(l_std) and l_std != "NA" and float(l_std) <= 0.01:
            is_flat = True
        elif pd.isna(l_std) or l_std == "NA":
            is_flat = True
            
        if is_flat and p_str != "NA":
            p_str += "*"
            s_str += "*"

        rows.append({
            "run": run_name,
            "lambda_tv": lambda_tv,
            "use_rsl": use_rsl,
            "use_nwd": use_nwd,
            "nwd_mode": nwd_mode,
            "nwd_c": nwd_c,
            "size_tau": size_tau,
            "mAP50": mAP50,
            "mAP50_95": mAP50_95,
            "AP_small": AP_small,
            "AP_medium": AP_medium,
            "AP_large": AP_large,
            "L_mean": l_mean,
            "L_std": l_std,
            "pearson": p_str,
            "spearman": s_str,
            "gamma_final": gamma_final,
        })

    df_out = pd.DataFrame(rows)
    for p in ["tables/C2_table1.csv", "artifacts/phase2/C2_table1.csv"]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        df_out.to_csv(p, index=False)
        print(f"[SUCCESS] Wrote unified C2_table1 to {p} ({len(df_out)} rows)")


def build_c2_branch_scale():
    """Build tables/C2_branch_scale.csv analyzing two-stage collapse dynamics."""
    diag_p = Path("tables/C2_final_diag.csv")
    hyperparams_p = Path("tables/C2_hyperparams.csv")
    
    if not diag_p.exists() or not hyperparams_p.exists():
        print("[WARN] C2_final_diag.csv or C2_hyperparams.csv not found.")
        return
        
    df_diag = pd.read_csv(diag_p)
    df_hyp = pd.read_csv(hyperparams_p)
    
    lam_map = dict(zip(df_hyp["run"], df_hyp["lambda_tv"]))
    
    rows = []
    for _, r in df_diag.iterrows():
        run_name = str(r["run"])
        # Exclude rows where L_std is NA (e.g. baseline)
        if pd.isna(r.get("L_std")) or r.get("L_std") == "NA":
            continue
            
        lam_tv = lam_map.get(run_name, "NA")
        if lam_tv == "NA":
            continue
            
        try:
            l_mean = float(r["L_mean"])
            l_std = float(r["L_std"])
            gamma = float(r["gamma_final"])
            mult = float(r["mean_one_minus_L"])
            eff_scale = round(gamma * mult, 6)
            
            rows.append({
                "run": run_name,
                "lambda_tv": lam_tv,
                "L_mean": l_mean,
                "L_std": l_std,
                "mean_one_minus_L": mult,
                "gamma_final": gamma,
                "effective_scale": f"{eff_scale:.6f}",
            })
        except Exception:
            continue

    df_bs = pd.DataFrame(rows)
    for p in ["tables/C2_branch_scale.csv", "artifacts/phase2/C2_branch_scale.csv"]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        df_bs.to_csv(p, index=False)
        print(f"[SUCCESS] Wrote branch scale table to {p}")


def build_c2_gamma_curves():
    """Scan and aggregate rfd_log.csv across run directories."""
    hyperparams_p = Path("tables/C2_hyperparams.csv")
    if not hyperparams_p.exists():
        print("[WARN] C2_hyperparams.csv not found.")
        return
        
    df_hyp = pd.read_csv(hyperparams_p)
    valid_runs = set(df_hyp["run"].unique())

    patterns = [
        "runs/detect/runs/**/rfd_log.csv",
        "runs/detect/runs_fixed/**/rfd_log.csv",
        "runs_fixed/**/rfd_log.csv",
        "runs/**/rfd_log.csv",
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    files = sorted(list(set(files)))

    if not files:
        print("\n[INFO] E.7: No rfd_log.csv files found. Skipping C2_gamma_curves.csv.")
        return

    frames = []
    for f in files:
        # C2_hyperparams.csv uses either basename or full relative path
        # Let's find if the file matches any valid_runs
        run_name = None
        for vr in valid_runs:
            if f.replace("\\", "/").startswith(vr + "/rfd_log.csv") or vr == Path(f).parent.name:
                run_name = vr
                break
                
        if not run_name:
            continue
            
        if "baseline" in run_name.lower():
            print(f"[INFO] Skipping rfd_log.csv for {run_name} (baseline models have no RFDBlock)")
            continue
            
        try:
            df = pd.read_csv(f)
            # Use the basename for consistency with C2_gamma_curves.csv
            df["run"] = Path(f).parent.name
            
            if "L_mean" in df.columns:
                mask = ~((df["L_mean"] == 0.5) & (df["L_min"] == 0.0) & (df["L_max"] == 0.5))
                df = df[mask]
                
            if len(df) > 0:
                frames.append(df)
                print(f"[INFO] Found rfd_log.csv for {run_name} ({len(df)} valid epochs)")
        except Exception as e:
            print(f"[WARN] Error reading {f}: {e}")

    if frames:
        merged = pd.concat(frames, ignore_index=True)
        for p in ["tables/C2_gamma_curves.csv", "artifacts/phase2/C2_gamma_curves.csv"]:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            merged.to_csv(p, index=False)
            print(f"[SUCCESS] Exported gamma curves to {p} ({len(merged)} rows)")
    else:
        print("\n[INFO] No valid rfd_log.csv files found after filtering.")


def build_c2_metrics_unified():
    """Build tables/C2_metrics_unified.csv merging standalone val size metrics."""
    size_p = Path("tables/C2_table2_bysize.csv")
    if not size_p.exists():
        print("[WARN] C2_table2_bysize.csv not found")
        return

    df_size = pd.read_csv(size_p)
    df_size["source"] = "model.val (plots=False, save_json=True)"
    cols = ["ckpt", "source", "mAP50", "mAP50_95", "AP_small", "AP_medium", "AP_large"]
    if "n_small" in df_size.columns:
        cols.extend(["n_small", "n_medium", "n_large"])
    avail = [c for c in cols if c in df_size.columns]
    out_df = df_size[avail].rename(columns={"ckpt": "run"})

    for p in ["tables/C2_metrics_unified.csv", "artifacts/phase2/C2_metrics_unified.csv"]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        out_df.to_csv(p, index=False)
        print(f"[SUCCESS] Wrote unified metrics table to {p}")


def main():
    print("=" * 80)
    print("  RI-YOLO: BUILDING STAGE E TABLES (KT-10)")
    print("=" * 80)
    build_c2_table1()
    build_c2_branch_scale()
    build_c2_gamma_curves()
    build_c2_metrics_unified()


if __name__ == "__main__":
    main()

```

## H.3: Скрипт `build_c2_stats.py`
```python
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
import warnings
import math

def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    if dof <= 0: return float('nan')
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    if pooled_std == 0: return float('nan')
    return (np.mean(x) - np.mean(y)) / pooled_std

def get_stats(x, y):
    if len(x) == 0 or len(y) == 0:
        return {"t": "NA", "df": "NA", "p_welch": "NA", "ci_95": "NA", "U": "NA", "p_mw": "NA", "d": "NA", "min_p": "NA"}
    if len(x) == 1 or len(y) == 1:
        return {"t": "NA", "df": "NA", "p_welch": "NA", "ci_95": "NA", "U": "NA", "p_mw": "NA", "d": "NA", "min_p": "NA"}
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(x) > 1 and len(y) > 1:
            # Welch's t-test
            t, p_welch = ttest_ind(x, y, equal_var=False)
            df = len(x) + len(y) - 2 # Approx
            # CI
            diff = np.mean(x) - np.mean(y)
            se = np.sqrt(np.var(x, ddof=1)/len(x) + np.var(y, ddof=1)/len(y))
            ci_95 = f"[{diff - 1.96*se:.4f}, {diff + 1.96*se:.4f}]"
            # Mann-Whitney U
            U, p_mw = mannwhitneyu(x, y, alternative='two-sided')
            # Cohen's d
            d = cohens_d(x, y)
            # min_p for mw
            min_p = 2 / math.comb(len(x) + len(y), len(x))
            
            return {
                "t": round(t, 4) if not np.isnan(t) else "NA",
                "df": df,
                "p_welch": round(p_welch, 4) if not np.isnan(p_welch) else "NA",
                "ci_95": ci_95,
                "U": U,
                "p_mw": round(p_mw, 4) if not np.isnan(p_mw) else "NA",
                "d": round(d, 4) if not np.isnan(d) else "NA",
                "min_p": round(min_p, 4)
            }
        else:
            return {"t": "NA", "df": "NA", "p_welch": "NA", "ci_95": "NA", "U": "NA", "p_mw": "NA", "d": "NA", "min_p": "NA"}

if __name__ == "__main__":
    df_metrics = pd.read_csv("tables/C2_final_metrics.csv")
    df_diag = pd.read_csv("tables/C2_final_diag.csv")
    
    merged = pd.merge(df_metrics, df_diag, on="run", how="left")
    
    # groups
    base = merged[merged["run"].str.startswith("e2p__baseline")]
    postsppf = merged[merged["run"].str.startswith("e2p__postsppf") | merged["run"].str.startswith("e3p__pos-postsppf")]
    
    rows = []
    
    def add_comparison(name, g1, g2):
        for metric in ["mAP50", "mAP50_95", "AP_small", "L_std"]:
            if metric not in g1.columns or metric not in g2.columns:
                continue
                
            x_raw = g1[metric].dropna()
            y_raw = g2[metric].dropna()
            
            x = pd.to_numeric(x_raw, errors='coerce').dropna().values
            y = pd.to_numeric(y_raw, errors='coerce').dropna().values
            
            diff_pp = (np.mean(x) - np.mean(y)) * 100 if len(x) > 0 and len(y) > 0 else "NA"
            
            stats = get_stats(x, y)
            rows.append({
                "comparison": name,
                "metric": metric,
                "mean_g1": round(np.mean(x), 4) if len(x)>0 else "NA",
                "std_g1": round(np.std(x, ddof=1), 4) if len(x)>1 else "NA",
                "n_g1": len(x),
                "mean_g2": round(np.mean(y), 4) if len(y)>0 else "NA",
                "std_g2": round(np.std(y, ddof=1), 4) if len(y)>1 else "NA",
                "n_g2": len(y),
                "diff_pp": round(diff_pp, 4) if diff_pp != "NA" else "NA",
                **stats
            })
            
    add_comparison("base_vs_postsppf(lam=0)", base, postsppf)
    
    # against e6p__lam-1e-4
    e6p_1e4 = merged[merged["run"].str.startswith("e6p__lam-1e-4")]
    add_comparison("postsppf_vs_lam-1e-4", postsppf, e6p_1e4)
    
    # single lam points
    for lam in ["1e-5", "1e-3", "1e-2", "1e-1"]:
        g2 = merged[merged["run"].str.startswith(f"e6p__lam-{lam}")]
        if len(g2) > 0:
            add_comparison(f"postsppf_vs_lam-{lam}", postsppf, g2)
            
    # nwd-calib-norsl
    nwd_norsl = merged[merged["run"].str.startswith("e7p__nwd-calib-norsl")]
    if len(nwd_norsl) > 0:
        add_comparison("postsppf_vs_nwd-calib-norsl", postsppf, nwd_norsl)
        
    # sizegate broken vs fixed
    broken = merged[merged["run"] == "runs/detect/runs/e7p__nwd-sizegate__s0"]
    fixed = merged[merged["run"] == "runs_fixed/e7p__nwd-sizegate__s0"]
    if len(broken) > 0 and len(fixed) > 0:
        add_comparison("sizegate_broken_vs_fixed", broken, fixed)
        
    res_df = pd.DataFrame(rows)
    res_df.to_csv("tables/C2_final_stats.csv", index=False)
    print("[SUCCESS] tables/C2_final_stats.csv generated.")

```

## H.4: Скрипт `consistency_check.py` и его дословный вывод
```python
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

```

### Дословный вывод `consistency_check.py`:
```text
================================================================================
  STAGE H: C2 CONSISTENCY CHECKER
================================================================================

--- 1. Checking mAP50 & mAP50_95 (table1 vs final_metrics) ---

--- 2. Checking L_metrics (table1 & branch_scale vs final_diag) ---

--- 3. Checking lambda_tv vs hyperparams ---

--- 4. Checking for missing runs in hyperparams ---

--- 5. Checking C2_gamma_curves.csv for baseline runs ---
================================================================================
[SUCCESS] Consistency check passed! 0 discrepancies found.
================================================================================

```

## H.5: Проверка `C2_gamma_curves.csv`
прогон | существует ли rfd_log.csv | число строк в нём | число строк в C2_gamma_curves.csv
---|---|---|---
`e2p__baseline__s0` | Нет | 0 | 0
`e2p__baseline__s1` | Нет | 0 | 0
`e2p__baseline__s2` | Нет | 0 | 0
`e2p__baseline__s3` | Нет | 0 | 0
`e2p__postsppf__s0` | Нет | 0 | 0
`e2p__postsppf__s1` | Нет | 0 | 202
`e2p__postsppf__s2` | Нет | 0 | 202
`e2p__postsppf__s3` | Нет | 0 | 202
`e6p__lam-1e-1__s0` | Нет | 0 | 0
`e6p__lam-1e-2__s0` | Нет | 0 | 0
`e6p__lam-1e-3__s0` | Нет | 0 | 0
`e6p__lam-1e-4__s0` | Нет | 0 | 0
`e6p__lam-1e-4__s1` | Нет | 0 | 202
`e6p__lam-1e-4__s2` | Нет | 0 | 202
`e6p__lam-1e-5__s0` | Нет | 0 | 202
`e7p__nwd-calib-norsl__s0` | Нет | 0 | 202
`e7p__nwd-calib__s0` | Нет | 0 | 0
`e7p__nwd-scaleinv__s0` | Нет | 0 | 0
`runs/detect/runs/e7p__nwd-sizegate__s0` | Нет | 0 | 0
`runs_fixed/e7p__nwd-sizegate__s0` | Нет | 0 | 0

**Первые 5 строк C2_gamma_curves.csv:**
```csv
epoch,module,gamma,L_mean,L_std_spatial,L_min,L_max,L_sat_frac,run
0,model.10,0.0010906391544267,0.5093271732330322,0.0944307446479797,0.1866455078125,0.83642578125,0.0,e2p__postsppf__s1
1,model.10,-0.0002078609686577,0.5069857835769653,0.1471340954303741,0.08978271484375,0.9697265625,0.0,e2p__postsppf__s1
2,model.10,-0.0009318655356764,0.5046465992927551,0.1888309270143509,0.0090560913085937,0.97412109375,0.0006249999860301,e2p__postsppf__s1
3,model.10,0.0023156064562499,0.4982158839702606,0.2142596840858459,0.0054054260253906,0.98095703125,0.0040624998509883,e2p__postsppf__s1
4,model.10,-0.001503016334027,0.5210017561912537,0.2231088131666183,0.003854751586914,0.97998046875,0.0046874997206032,e2p__postsppf__s1
```


## H.6: Финальный архив
Архив успешно создан: `C2_data_final.tar.gz`.

Список файлов, прикреплённых к этому сообщению:
1. `tables/C2_table1.csv`
2. `tables/C2_final_metrics.csv`
3. `tables/C2_final_diag.csv`
4. `tables/C2_final_stats.csv`
5. `tables/C2_branch_scale.csv`
6. `tables/C2_gamma_curves.csv`
7. `tables/C2_rsl_curves.csv`
8. `tables/C2_hyperparams.csv`
9. `scripts/build_c2_stage_e.py`
10. `scripts/consistency_check.py`
11. `C2_data_final.tar.gz` (содержит всё)
