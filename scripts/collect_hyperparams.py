# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Stage F.4: Hyperparameter Extractor for Paper C2.

Extracts hyperparameters directly from args.yaml of each run:
- Zero manual dictionary entry or hardcoding
- Direct parsing of YAML configuration files
- Produces tables/C2_hyperparams.csv
"""

import argparse
import csv
import glob
import os
from pathlib import Path
import yaml
import pandas as pd


RUN_DIRS = [
    ("e2p__postsppf__s0", ["runs/detect/runs/e3p__pos-postsppf__s0", "runs/detect/runs/e2p__postsppf__s0", "runs/e3p__pos-postsppf__s0"]),
    ("e2p__postsppf__s1", ["runs/detect/runs/e2p__postsppf__s1", "runs/e2p__postsppf__s1"]),
    ("e2p__postsppf__s2", ["runs/detect/runs/e2p__postsppf__s2", "runs/e2p__postsppf__s2"]),
    ("e2p__postsppf__s3", ["runs/detect/runs/e2p__postsppf__s3", "runs/e2p__postsppf__s3"]),
    ("e6p__lam-1e-5__s0", ["runs/detect/runs/e6p__lam-1e-5__s0", "runs/e6p__lam-1e-5__s0"]),
    ("e6p__lam-1e-4__s0", ["runs/detect/runs/e6p__lam-1e-4__s0", "runs/e6p__lam-1e-4__s0"]),
    ("e6p__lam-1e-4__s1", ["runs/detect/runs/e6p__lam-1e-4__s1", "runs/e6p__lam-1e-4__s1"]),
    ("e6p__lam-1e-4__s2", ["runs/detect/runs/e6p__lam-1e-4__s2", "runs/e6p__lam-1e-4__s2"]),
    ("e6p__lam-1e-3__s0", ["runs/detect/runs/e6p__lam-1e-3__s0", "runs/e6p__lam-1e-3__s0"]),
    ("e6p__lam-1e-2__s0", ["runs/detect/runs/e6p__lam-1e-2__s0", "runs/e6p__lam-1e-2__s0"]),
    ("e6p__lam-1e-1__s0", ["runs/detect/runs/e6p__lam-1e-1__s0", "runs/e6p__lam-1e-1__s0"]),
    ("e7p__nwd-calib__s0", ["runs/detect/runs/e7p__nwd-calib__s0", "runs/e7p__nwd-calib__s0"]),
    ("e7p__nwd-scaleinv__s0", ["runs/detect/runs/e7p__nwd-scaleinv__s0", "runs/e7p__nwd-scaleinv__s0"]),
    ("e7p__nwd-calib-norsl__s0", ["runs/detect/runs/e7p__nwd-calib-norsl__s0", "runs/e7p__nwd-calib-norsl__s0"]),
    ("runs_fixed/e7p__nwd-sizegate__s0", ["runs_fixed/e7p__nwd-sizegate__s0", "runs/detect/runs_fixed/e7p__nwd-sizegate__s0", "runs_fixed/detect/runs/e7p__nwd-sizegate__s0"]),
    ("runs/detect/runs/e7p__nwd-sizegate__s0", ["runs/detect/runs/e7p__nwd-sizegate__s0", "runs/e7p__nwd-sizegate__s0"]),
]


def find_args_yaml(candidates, run_name):
    for c in candidates:
        p = Path(c) / "args.yaml"
        if p.is_file():
            return p
    clean_name = run_name.split("/")[-1]
    matches = list(Path(".").glob(f"**/{clean_name}/args.yaml"))
    if matches:
        if "fixed" in run_name:
            fixed_matches = [m for m in matches if "fixed" in str(m)]
            if fixed_matches:
                return fixed_matches[0]
        else:
            non_fixed = [m for m in matches if "fixed" not in str(m)]
            if non_fixed:
                return non_fixed[0]
        return matches[0]
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Collect hyperparameters directly from args.yaml")
    parser.add_argument("--out-csv", default="tables/C2_hyperparams.csv", help="Output CSV path")
    parser.add_argument("--out-art", default="artifacts/phase2/C2_hyperparams.csv", help="Artifact CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []

    print("=" * 80)
    print("  RI-YOLO: HYPERPARAMETERS EXTRACTOR (Stage F.4)")
    print("=" * 80)

    for run_name, candidates in RUN_DIRS:
        yaml_p = find_args_yaml(candidates, run_name)
        if yaml_p is None or not yaml_p.is_file():
            print(f"[WARN] args.yaml for '{run_name}' NOT FOUND. Skipping.")
            continue

        with open(yaml_p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        row = {
            "run": run_name,
            "args_yaml_path": str(yaml_p).replace("\\", "/"),
            "epochs": cfg.get("epochs", "NA"),
            "batch": cfg.get("batch", "NA"),
            "imgsz": cfg.get("imgsz", "NA"),
            "seed": cfg.get("seed", "NA"),
            "close_mosaic": cfg.get("close_mosaic", "NA"),
            "use_rsl": cfg.get("use_rsl", False),
            "lambda_tv": cfg.get("lambda_tv", "NA"),
            "use_nwd": cfg.get("use_nwd", False),
            "nwd_alpha": cfg.get("nwd_alpha", "NA"),
            "nwd_mode": cfg.get("nwd_mode", "NA"),
            "nwd_c": cfg.get("nwd_c", "NA"),
            "size_tau": cfg.get("size_tau", "NA"),
        }
        rows.append(row)
        print(f"[OK] {run_name}: use_rsl={row['use_rsl']}, lambda_tv={row['lambda_tv']}, "
              f"use_nwd={row['use_nwd']}, nwd_mode={row['nwd_mode']}, "
              f"nwd_c={row['nwd_c']}, size_tau={row['size_tau']}")

    for out_p in [args.out_csv, args.out_art]:
        p = Path(out_p)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        print(f"[SUCCESS] Wrote hyperparams to {p} ({len(rows)} runs)")

    print("\n" + "=" * 80)
    print("                    C2 HYPERPARAMETERS TABLE")
    print("=" * 80)
    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
