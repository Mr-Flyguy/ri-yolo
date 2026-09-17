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
