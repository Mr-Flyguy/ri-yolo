# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Phase 2 Table Aggregator: TV-loss Regularization and NWD Metric Formulations.

Aggregates results for Article C2 (VAK / RSCI):
- Table 1: E6' lambda_tv Regularization Sweep (joined with L_std, rho, gamma diagnostics)
- Table 2: E7' Bounding Box Metric Formulations
"""

import argparse
import csv
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Collect and generate Tables for Paper C2")
    parser.add_argument(
        "--summary",
        type=str,
        default="artifacts/phase2/summary.csv",
        help="Path to Phase 2 summary.csv",
    )
    parser.add_argument(
        "--diag",
        type=str,
        default="tables/C2_diag.csv",
        help="Path to C2_diag.csv",
    )
    parser.add_argument(
        "--p1-summary",
        type=str,
        default="artifacts/phase1/summary.csv",
        help="Path to Phase 1 summary.csv (for lambda=0 anchor)",
    )
    parser.add_argument(
        "--out-t1",
        type=str,
        default="artifacts/phase2/C2_table1.csv",
        help="Output path for Table 1 (E6' TV-loss sweep)",
    )
    parser.add_argument(
        "--out-t2",
        type=str,
        default="artifacts/phase2/C2_table2.csv",
        help="Output path for Table 2 (E7' NWD variants)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary_file = Path(args.summary)
    diag_file = Path(args.diag)
    p1_file = Path(args.p1_summary)

    if not summary_file.exists():
        print(f"[WARN] {summary_file} does not exist yet. Run Phase 2 experiments first.")
        return

    sdf = pd.read_csv(summary_file)
    diag_df = pd.read_csv(diag_file) if diag_file.exists() else pd.DataFrame()
    p1_df = pd.read_csv(p1_file) if p1_file.exists() else pd.DataFrame()

    print(f"[INFO] Loaded {len(sdf)} record(s) from {summary_file}.")

    # --- Table 1: E6' TV-loss sweep ---
    e6_df = sdf[sdf["group"] == "E6"].copy()

    # Add lambda=0 anchor from Phase 1 postsppf if available
    anchor_rows = []
    if not p1_df.empty:
        p1_anchor = p1_df[p1_df["run"] == "e3p__pos-postsppf__s0"]
        if not p1_anchor.empty:
            r = p1_anchor.iloc[0].to_dict()
            anchor_rows.append({
                "run": "e3p__pos-postsppf__s0",
                "group": "E3",
                "model": r.get("model", "yolov8s-rfd-postsppf.yaml"),
                "seed": 0,
                "epochs": 100,
                "lambda_tv": 0.0,
                "mAP50": r.get("mAP50", "NA"),
                "mAP50_95": r.get("mAP50_95", "NA"),
                "gamma_final": r.get("gamma_final", "NA"),
            })

    if anchor_rows:
        e6_full = pd.concat([pd.DataFrame(anchor_rows), e6_df], ignore_index=True)
    else:
        e6_full = e6_df

    # Merge with diagnostics (L_mean, L_std, pearson, spearman)
    if not diag_df.empty:
        # Match by run name or map lam0 to e3p__pos-postsppf__s0
        diag_map = {}
        for _, row in diag_df.iterrows():
            rname = str(row["run"])
            if rname == "lam0":
                diag_map["e3p__pos-postsppf__s0"] = row
            else:
                diag_map[rname] = row

        for col in ["L_mean", "L_std", "pearson", "spearman"]:
            e6_full[col] = e6_full["run"].apply(lambda x: diag_map[x][col] if x in diag_map else "NA")

    # Order and clean columns
    cols_order = [
        "run", "lambda_tv", "mAP50", "mAP50_95",
        "L_mean", "L_std", "pearson", "spearman", "gamma_final"
    ]
    avail_cols = [c for c in cols_order if c in e6_full.columns]
    e6_out = e6_full[avail_cols]

    out_t1 = Path(args.out_t1)
    out_t1.parent.mkdir(parents=True, exist_ok=True)
    e6_out.to_csv(out_t1, index=False)
    # Mirror to tables/
    tables_t1 = Path("tables/C2_table1.csv")
    tables_t1.parent.mkdir(parents=True, exist_ok=True)
    e6_out.to_csv(tables_t1, index=False)
    print(f"[SUCCESS] Table 1 written to {out_t1} and {tables_t1} ({len(e6_out)} rows)")
    print(e6_out.to_string(index=False))

    # --- Table 2: E7' NWD variants ---
    e7_df = sdf[sdf["group"] == "E7"].copy()
    if not e7_df.empty:
        out_t2 = Path(args.out_t2)
        out_t2.parent.mkdir(parents=True, exist_ok=True)
        e7_df.to_csv(out_t2, index=False)
        tables_t2 = Path("tables/C2_table2.csv")
        tables_t2.parent.mkdir(parents=True, exist_ok=True)
        e7_df.to_csv(tables_t2, index=False)
        print(f"\n[SUCCESS] Table 2 written to {out_t2} and {tables_t2} ({len(e7_df)} rows)")
        print(e7_df.to_string(index=False))


if __name__ == "__main__":
    main()
