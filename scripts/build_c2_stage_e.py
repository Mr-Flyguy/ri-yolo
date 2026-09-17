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
    diag_p = Path("tables/C2_diag.csv")
    diag_map = {}
    if diag_p.exists():
        df_diag = pd.read_csv(diag_p)
        for _, r in df_diag.iterrows():
            diag_map[str(r["run"])] = r.to_dict()

    # Base records for E6' and E7'
    # Source metadata for all runs
    specs = [
        # E3 anchor
        {
            "run": "e3p__pos-postsppf__s0",
            "lambda_tv": 0.0,
            "use_rsl": False,
            "use_nwd": False,
            "nwd_mode": "NA",
            "nwd_c": "NA",
            "size_tau": "NA",
            "mAP50": 0.6937,
            "mAP50_95": 0.4348,
            "diag_key": "lam0",
            "gamma_final": 0.003155,
        },
        # E6' TV-loss sweep
        {
            "run": "e6p__lam-1e-5__s0",
            "lambda_tv": 0.00001,
            "use_rsl": True,
            "use_nwd": False,
            "nwd_mode": "NA",
            "nwd_c": "NA",
            "size_tau": "NA",
            "mAP50": 0.6941,
            "mAP50_95": 0.4317,
            "diag_key": "e6p__lam-1e-5__s0",
            "gamma_final": 0.016006,
        },
        {
            "run": "e6p__lam-1e-4__s0",
            "lambda_tv": 0.0001,
            "use_rsl": True,
            "use_nwd": False,
            "nwd_mode": "NA",
            "nwd_c": "NA",
            "size_tau": "NA",
            "mAP50": 0.6937,
            "mAP50_95": 0.4349,
            "diag_key": "e6p__lam-1e-4__s0",
            "gamma_final": 0.002043,
        },
        {
            "run": "e6p__lam-1e-4__s1",
            "lambda_tv": 0.0001,
            "use_rsl": True,
            "use_nwd": False,
            "nwd_mode": "NA",
            "nwd_c": "NA",
            "size_tau": "NA",
            "mAP50": 0.6942,
            "mAP50_95": 0.4322,
            "diag_key": "e6p__lam-1e-4__s1",
            "gamma_final": 0.007996,
        },
        {
            "run": "e6p__lam-1e-4__s2",
            "lambda_tv": 0.0001,
            "use_rsl": True,
            "use_nwd": False,
            "nwd_mode": "NA",
            "nwd_c": "NA",
            "size_tau": "NA",
            "mAP50": 0.6978,
            "mAP50_95": 0.4356,
            "diag_key": "e6p__lam-1e-4__s2",
            "gamma_final": 0.002247,
        },
        {
            "run": "e6p__lam-1e-3__s0",
            "lambda_tv": 0.001,
            "use_rsl": True,
            "use_nwd": False,
            "nwd_mode": "NA",
            "nwd_c": "NA",
            "size_tau": "NA",
            "mAP50": 0.6938,
            "mAP50_95": 0.4303,
            "diag_key": "e6p__lam-1e-3__s0",
            "gamma_final": 0.000282,
        },
        {
            "run": "e6p__lam-1e-2__s0",
            "lambda_tv": 0.01,
            "use_rsl": True,
            "use_nwd": False,
            "nwd_mode": "NA",
            "nwd_c": "NA",
            "size_tau": "NA",
            "mAP50": 0.6960,
            "mAP50_95": 0.4355,
            "diag_key": "e6p__lam-1e-2__s0",
            "gamma_final": 0.002626,
        },
        {
            "run": "e6p__lam-1e-1__s0",
            "lambda_tv": 0.1,
            "use_rsl": True,
            "use_nwd": False,
            "nwd_mode": "NA",
            "nwd_c": "NA",
            "size_tau": "NA",
            "mAP50": 0.6916,
            "mAP50_95": 0.4329,
            "diag_key": "e6p__lam-1e-1__s0",
            "gamma_final": 0.014175,
        },
        # E7' NWD series
        {
            "run": "e7p__nwd-calib__s0",
            "lambda_tv": 0.01,
            "use_rsl": True,
            "use_nwd": True,
            "nwd_mode": "abs",
            "nwd_c": 10.24,
            "size_tau": "NA",
            "mAP50": 0.6976,
            "mAP50_95": 0.4376,
            "diag_key": "e7p__nwd-calib__s0",
            "gamma_final": 0.011795,
        },
        {
            "run": "e7p__nwd-scaleinv__s0",
            "lambda_tv": 0.01,
            "use_rsl": True,
            "use_nwd": True,
            "nwd_mode": "scaleinv",
            "nwd_c": 0.3,
            "size_tau": "NA",
            "mAP50": 0.6935,
            "mAP50_95": 0.4367,
            "diag_key": "e7p__nwd-scaleinv__s0",
            "gamma_final": 0.008553,
        },
        {
            "run": "e7p__nwd-calib-norsl__s0",
            "lambda_tv": 0.0,
            "use_rsl": False,
            "use_nwd": True,
            "nwd_mode": "abs",
            "nwd_c": 10.24,
            "size_tau": "NA",
            "mAP50": 0.6986,
            "mAP50_95": 0.4357,
            "diag_key": "e7p__nwd-calib-norsl__s0",
            "gamma_final": 0.017899,
        },
        {
            "run": "e7p__nwd-sizegate__s0__broken",
            "lambda_tv": 0.01,
            "use_rsl": True,
            "use_nwd": True,
            "nwd_mode": "sizegate",
            "nwd_c": 10.24,
            "size_tau": 5.12,
            "mAP50": 0.5906,
            "mAP50_95": 0.3607,
            "diag_key": "e7p__nwd-sizegate__s0__broken",
            "gamma_final": -0.006260,
            "fallback_diag": {
                "L_mean": 0.8237,
                "L_std": 0.2200,
                "pearson": 0.0742,
                "spearman": 0.0883,
            },
        },
        {
            "run": "e7p__nwd-sizegate__s0__fixed",
            "lambda_tv": 0.0001,
            "use_rsl": True,
            "use_nwd": True,
            "nwd_mode": "sizegate",
            "nwd_c": 10.24,
            "size_tau": 5.12,
            "mAP50": 0.6933,
            "mAP50_95": 0.4311,
            "diag_key": "e7p__nwd-sizegate__s0",
            "gamma_final": 0.010483,
        },
    ]

    rows = []
    for s in specs:
        dkey = s["diag_key"]
        d = diag_map.get(dkey, s.get("fallback_diag", {}))
        l_std = float(d.get("L_std", 0.0)) if d.get("L_std") not in (None, "NA") else 0.0
        l_mean = float(d.get("L_mean", 0.0)) if d.get("L_mean") not in (None, "NA") else 0.0
        p_raw = d.get("pearson")
        s_raw = d.get("spearman")

        # Flag flat field correlations (L_std <= 0.01) with asterisk (*)
        p_str = f"{float(p_raw):.4f}" if p_raw not in (None, "NA") else "NA"
        s_str = f"{float(s_raw):.4f}" if s_raw not in (None, "NA") else "NA"
        if l_std <= 0.01 and p_str != "NA":
            p_str += "*"
            s_str += "*"

        rows.append({
            "run": s["run"],
            "lambda_tv": s["lambda_tv"],
            "use_rsl": s["use_rsl"],
            "use_nwd": s["use_nwd"],
            "nwd_mode": s["nwd_mode"],
            "nwd_c": s["nwd_c"],
            "size_tau": s["size_tau"],
            "mAP50": s["mAP50"],
            "mAP50_95": s["mAP50_95"],
            "L_mean": round(l_mean, 4) if l_mean else "NA",
            "L_std": round(l_std, 4) if l_std else "NA",
            "pearson": p_str,
            "spearman": s_str,
            "gamma_final": s["gamma_final"],
        })

    df_out = pd.DataFrame(rows)
    for p in ["tables/C2_table1.csv", "artifacts/phase2/C2_table1.csv"]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        df_out.to_csv(p, index=False)
        print(f"[SUCCESS] Wrote unified C2_table1 to {p} ({len(df_out)} rows)")
    print(df_out.to_string(index=False))


def build_c2_branch_scale():
    """Build tables/C2_branch_scale.csv analyzing two-stage collapse dynamics."""
    t1_p = Path("tables/C2_table1.csv")
    if not t1_p.exists():
        build_c2_table1()
    df = pd.read_csv(t1_p)
    rows = []
    for _, r in df.iterrows():
        try:
            l_mean = float(r["L_mean"])
            l_std = float(r["L_std"])
            gamma = float(r["gamma_final"])
            mult = round(1.0 - l_mean, 4)
            eff_scale = round(gamma * mult, 6)
            rows.append({
                "run": r["run"],
                "lambda_tv": r["lambda_tv"],
                "L_mean": l_mean,
                "L_std": l_std,
                "mean_branch_multiplier": mult,
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
    print("\n=== Branch Scale Summary ===")
    print(df_bs.to_string(index=False))


def build_c2_gamma_curves():
    """Scan and aggregate rfd_log.csv across run directories."""
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
        run_name = Path(f).parent.name
        try:
            df = pd.read_csv(f)
            df["run"] = run_name
            frames.append(df)
            print(f"[INFO] Found rfd_log.csv for {run_name} ({len(df)} epochs)")
        except Exception as e:
            print(f"[WARN] Error reading {f}: {e}")

    if frames:
        merged = pd.concat(frames, ignore_index=True)
        for p in ["tables/C2_gamma_curves.csv", "artifacts/phase2/C2_gamma_curves.csv"]:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            merged.to_csv(p, index=False)
            print(f"[SUCCESS] Exported gamma curves to {p} ({len(merged)} rows)")


def build_c2_metrics_unified():
    """Build tables/C2_metrics_unified.csv merging standalone val size metrics."""
    size_p = Path("tables/C2_table2_bysize.csv")
    if not size_p.exists():
        print("[WARN] C2_table2_bysize.csv not found")
        return

    df_size = pd.read_csv(size_p)
    df_size["source"] = "model.val (plots=False, save_json=True)"
    # Reorder columns
    cols = ["ckpt", "source", "mAP50", "mAP50_95", "AP_small", "AP_medium", "AP_large"]
    if "n_small" in df_size.columns:
        cols.extend(["n_small", "n_medium", "n_large"])
    avail = [c for c in cols if c in df_size.columns]
    out_df = df_size[avail].rename(columns={"ckpt": "run"})

    for p in ["tables/C2_metrics_unified.csv", "artifacts/phase2/C2_metrics_unified.csv"]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        out_df.to_csv(p, index=False)
        print(f"[SUCCESS] Wrote unified metrics table to {p}")
    print("\n=== Unified Metrics Table ===")
    print(out_df.to_string(index=False))


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
