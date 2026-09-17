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


import yaml


LAMBDA_MAP = {
    # Only runs where RSL loss was active (use_rsl=True, lambda_tv > 0)
    "e6p__lam-1e-5__s0": 0.00001,
    "e6p__lam-1e-4__s0": 0.0001,
    "e6p__lam-1e-4__s1": 0.0001,
    "e6p__lam-1e-4__s2": 0.0001,
    "e6p__lam-1e-3__s0": 0.001,
    "e6p__lam-1e-2__s0": 0.01,
    "e6p__lam-1e-1__s0": 0.1,
    "e7p__nwd-calib__s0": 0.01,
    "e7p__nwd-scaleinv__s0": 0.01,
    "e7p__nwd-sizegate__s0": 0.01,
}


def find_results_csv(run_name):
    candidates = [
        Path("runs_fixed") / run_name / "results.csv",
        Path("runs_fixed/detect/runs") / run_name / "results.csv",
        Path("runs/detect/runs_fixed") / run_name / "results.csv",
        Path("runs/detect/runs") / run_name / "results.csv",
        Path("runs") / run_name / "results.csv",
        Path(f"runs/detect/{run_name}/results.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c
    # Recursive glob under runs/ and runs_fixed/
    for base in ["runs_fixed", "runs"]:
        if Path(base).exists():
            matches = list(Path(base).glob(f"**/{run_name}/results.csv"))
            if matches:
                return matches[0]
    return None


def find_args_yaml(run_name, results_csv_path=None):
    if results_csv_path:
        cand = results_csv_path.parent / "args.yaml"
        if cand.exists():
            return cand
    candidates = [
        Path("runs_fixed") / run_name / "args.yaml",
        Path("runs_fixed/detect/runs") / run_name / "args.yaml",
        Path("runs/detect/runs_fixed") / run_name / "args.yaml",
        Path("runs/detect/runs") / run_name / "args.yaml",
        Path("runs") / run_name / "args.yaml",
        Path(f"runs/detect/{run_name}/args.yaml"),
    ]
    for c in candidates:
        if c.exists():
            return c
    for base in ["runs_fixed", "runs"]:
        if Path(base).exists():
            matches = list(Path(base).glob(f"**/{run_name}/args.yaml"))
            if matches:
                return matches[0]
    return None


def get_lambda_tv_from_args(args_yaml_path, fallback=None):
    if not args_yaml_path or not Path(args_yaml_path).exists():
        return fallback
    try:
        with open(args_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data and "lambda_tv" in data:
                val = float(data["lambda_tv"])
                print(f"[INFO] Read lambda_tv={val} from {args_yaml_path}")
                return val
    except Exception as e:
        print(f"[WARN] Error reading {args_yaml_path}: {e}")
    return fallback


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

        args_path = find_args_yaml(run_name, csv_path)
        actual_lam = get_lambda_tv_from_args(args_path, fallback=lam)

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
        df["lambda_tv"] = actual_lam
        df["weighted_rsl_loss"] = actual_lam * df["train/rsl_loss"]
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
            "lambda_tv": actual_lam,
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
