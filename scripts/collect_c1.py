# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Collects and aggregates Phase 1 experimental results into Table 1 for Paper C1 (RSCI).

Computes mAP50, mAP50-95, receptive field (TRF), parameter overhead,
and relative improvement over baseline across seeds.
"""

import argparse
import csv
import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd

TRF = {
    "baseline": (None, None, 0.0),
    "pos-p3": (95, 14.8, 0.164),
    "pos-p4": (239, 37.3, 0.657),
    "pos-presppf": (399, 62.3, 2.624),
    "pos-postsppf": (783, 122.3, 2.624),
    "capctrl": (783, 122.3, 2.623),
}
ORDER = ["baseline", "pos-p3", "pos-p4", "pos-presppf", "pos-postsppf", "capctrl"]


def parse_args():
    parser = argparse.ArgumentParser(description="Collect C1 Table 1 metrics.")
    parser.add_argument("--out", default="artifacts/phase1/C1_table1.csv", help="Output CSV path")
    parser.add_argument("--summary", default="artifacts/phase1/summary.csv", help="Phase 1 summary CSV path")
    return parser.parse_args()


def extract_metrics_from_csv(results_path):
    """Extract best mAP metrics from results.csv."""
    df = pd.read_csv(results_path)
    df.columns = [c.strip() for c in df.columns]
    m50_col = "metrics/mAP50(B)"
    m95_col = "metrics/mAP50-95(B)"
    if m50_col in df.columns:
        best_i = df[m50_col].idxmax()
        return float(df[m50_col].iloc[best_i]), float(df[m95_col].iloc[best_i])
    return 0.0, 0.0


def main():
    args = parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # 1. Parse from summary.csv if available
    summary_data = {}
    if os.path.exists(args.summary):
        sdf = pd.read_csv(args.summary)
        for _, row in sdf.iterrows():
            rname = row["run"]
            summary_data[rname] = {
                "mAP50": float(row["mAP50"]),
                "mAP50_95": float(row["mAP50_95"]),
                "gamma": float(row["gamma_final"]) if str(row["gamma_final"]).strip() not in ("NA", "None", "") else None,
            }

    # 2. Group by configuration
    table_rows = []
    for cfg in ORDER:
        runs_for_cfg = []

        # Find in summary data
        for rname, data in summary_data.items():
            if cfg in rname or (cfg == "pos-postsppf" and "postsppf" in rname):
                runs_for_cfg.append(data)

        # Fallback to scanning runs/ directly
        if not runs_for_cfg:
            seen_runs = set()
            for p in sorted(Path("runs").glob(f"**/*{cfg}*/results.csv")):
                run_tag = p.parent.name
                if run_tag in seen_runs:
                    continue
                seen_runs.add(run_tag)
                m50, m95 = extract_metrics_from_csv(p)
                runs_for_cfg.append({"mAP50": m50, "mAP50_95": m95, "gamma": None})

        if not runs_for_cfg:
            continue

        m50_arr = np.array([r["mAP50"] for r in runs_for_cfg])
        m95_arr = np.array([r["mAP50_95"] for r in runs_for_cfg])
        gammas = [r["gamma"] for r in runs_for_cfg if r.get("gamma") is not None]

        trf_px, cov, dpar = TRF[cfg]
        table_rows.append({
            "config": cfg,
            "n_runs": len(runs_for_cfg),
            "TRF_px": trf_px,
            "coverage_pct": cov,
            "delta_params_M": dpar,
            "mAP50_mean": round(float(m50_arr.mean()), 4),
            "mAP50_std": round(float(m50_arr.std(ddof=1)), 4) if len(runs_for_cfg) > 1 else None,
            "mAP50_95_mean": round(float(m95_arr.mean()), 4),
            "mAP50_95_std": round(float(m95_arr.std(ddof=1)), 4) if len(runs_for_cfg) > 1 else None,
            "gamma_mean": round(float(np.mean(gammas)), 6) if gammas else None,
        })

    if not table_rows:
        print("[WARNING] No completed run results found to generate C1 Table 1.")
        return

    out_df = pd.DataFrame(table_rows)

    # Compute delta to baseline if baseline exists
    base_rows = out_df[out_df["config"] == "baseline"]
    if len(base_rows) > 0:
        base_m50 = base_rows["mAP50_mean"].iloc[0]
        out_df["delta_mAP50_pp"] = out_df["mAP50_mean"].apply(lambda x: round(100.0 * (x - base_m50), 2))
    else:
        out_df["delta_mAP50_pp"] = None

    out_df.to_csv(args.out, index=False)
    print("\n" + "=" * 90)
    print("                    TABLE 1: PAPER C1 ABLATION SUMMARY")
    print("=" * 90)
    print(out_df.to_string(index=False))
    print("=" * 90)
    print(f"[INFO] Saved Table 1 to {args.out}\n")


if __name__ == "__main__":
    main()
