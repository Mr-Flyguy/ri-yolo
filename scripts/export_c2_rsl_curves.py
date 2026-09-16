# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Export C2 RSL curves and loss terms balance summary table.

Aggregates training curves from results.csv for Retinex Regularization (E6')
and NWD metric (E7') runs, computes loss component ratios, and exports
tables/C2_rsl_curves.csv and a markdown summary.
"""

import argparse
import os
from pathlib import Path
import pandas as pd


LAMBDA_MAP = {
    "e3p__pos-postsppf__s0": 0.0,
    "e6p__lam-1e-4__s0": 0.0001,
    "e6p__lam-1e-3__s0": 0.001,
    "e6p__lam-1e-2__s0": 0.01,
    "e6p__lam-1e-1__s0": 0.1,
    "e7p__nwd-calib__s0": 0.001,
    "e7p__nwd-scaleinv__s0": 0.001,
    "e7p__nwd-sizegate__s0": 0.001,
}


def find_results_csv(run_name):
    candidates = [
        Path("runs/detect/runs") / run_name / "results.csv",
        Path("runs") / run_name / "results.csv",
        Path(f"runs/detect/{run_name}/results.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c
    # Recursive glob under runs/
    matches = list(Path("runs").glob(f"**/{run_name}/results.csv"))
    if matches:
        return matches[0]
    return None


def main():
    parser = argparse.ArgumentParser(description="Export C2 RSL Curves and Loss Ratios")
    parser.add_argument("--out-csv", default="tables/C2_rsl_curves.csv", help="Output path for curves CSV")
    parser.add_argument("--out-art", default="artifacts/phase2/C2_rsl_curves.csv", help="Secondary artifact path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_art)), exist_ok=True)

    all_frames = []
    summary_rows = []

    print("=" * 80)
    print("  RI-YOLO: C2 RSL CURVES & LOSS TERMS BALANCE EXTRACTOR")
    print("=" * 80)

    for run_name, lam in LAMBDA_MAP.items():
        csv_path = find_results_csv(run_name)
        if not csv_path:
            print(f"[WARN] results.csv for {run_name} not found, skipping.")
            continue

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        # Ensure required loss columns exist
        for col in ["train/box_loss", "train/cls_loss", "train/dfl_loss"]:
            if col not in df.columns:
                df[col] = 0.0

        if "train/rsl_loss" not in df.columns:
            df["train/rsl_loss"] = 0.0

        df["train/rsl_loss"] = df["train/rsl_loss"].fillna(0.0)
        df["run"] = run_name
        df["lambda_tv"] = lam
        df["weighted_rsl_loss"] = lam * df["train/rsl_loss"]
        df["train/det_loss"] = df["train/box_loss"] + df["train/cls_loss"] + df["train/dfl_loss"]

        df["ratio_rsl_to_box"] = (df["weighted_rsl_loss"] / df["train/box_loss"].replace(0, float("nan"))).fillna(0.0)
        df["ratio_rsl_to_det"] = (df["weighted_rsl_loss"] / df["train/det_loss"].replace(0, float("nan"))).fillna(0.0)

        all_frames.append(df)

        # Compute summary metrics across 100 epochs
        mean_box = df["train/box_loss"].mean()
        mean_cls = df["train/cls_loss"].mean()
        mean_dfl = df["train/dfl_loss"].mean()
        mean_det = df["train/det_loss"].mean()
        mean_rsl_raw = df["train/rsl_loss"].mean()
        mean_rsl_weighted = df["weighted_rsl_loss"].mean()
        mean_ratio_box = df["ratio_rsl_to_box"].mean()
        mean_ratio_det = df["ratio_rsl_to_det"].mean()

        # Early (1-10) vs Late (91-100) raw RSL
        early_rsl = df[df["epoch"] <= 10]["train/rsl_loss"].mean() if len(df) >= 10 else mean_rsl_raw
        late_rsl = df[df["epoch"] >= 90]["train/rsl_loss"].mean() if len(df) >= 90 else mean_rsl_raw

        summary_rows.append({
            "run": run_name,
            "lambda_tv": lam,
            "L_box_mean": round(mean_box, 4),
            "L_cls_mean": round(mean_cls, 4),
            "L_dfl_mean": round(mean_dfl, 4),
            "L_det_mean": round(mean_det, 4),
            "L_tv_early(1-10)": round(early_rsl, 5),
            "L_tv_late(91-100)": round(late_rsl, 5),
            "L_tv_raw_mean": round(mean_rsl_raw, 5),
            "weighted_L_tv": f"{mean_rsl_weighted:.2e}",
            "ratio_tv_to_box": f"{mean_ratio_box:.2e}",
            "ratio_tv_to_det": f"{mean_ratio_det:.2e}",
        })

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        # Select clean column subset for C2_rsl_curves.csv
        cols = [
            "run", "epoch", "lambda_tv",
            "train/box_loss", "train/cls_loss", "train/dfl_loss", "train/det_loss",
            "train/rsl_loss", "weighted_rsl_loss", "ratio_rsl_to_box", "ratio_rsl_to_det",
            "metrics/mAP50(B)", "metrics/mAP50-95(B)"
        ]
        # Keep only available cols
        avail_cols = [c for c in cols if c in combined.columns]
        combined[avail_cols].to_csv(args.out_csv, index=False)
        combined[avail_cols].to_csv(args.out_art, index=False)
        print(f"[SUCCESS] Exported curves to {args.out_csv} and {args.out_art} ({len(combined)} rows)")

        sdf = pd.DataFrame(summary_rows)
        print("\n" + "=" * 80)
        print("                 LOSS TERMS BALANCE SUMMARY TABLE")
        print("=" * 80)
        print(sdf.to_string(index=False))
        print("=" * 80)
    else:
        print("[ERROR] No runs were loaded.")


if __name__ == "__main__":
    main()
