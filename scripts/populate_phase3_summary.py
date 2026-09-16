# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Populates artifacts/phase3/summary.csv with existing Phase 3 E4' runs (D.0.4)."""

import csv
import json
import os
import shutil
from pathlib import Path
import pandas as pd
import torch


def get_gamma(run_dir):
    ck = Path(run_dir) / "weights" / "best.pt"
    if ck.exists():
        try:
            sd = torch.load(ck, map_location="cpu", weights_only=False)["model"].state_dict()
            g = [float(v.reshape(-1)[0]) for k, v in sd.items() if k.endswith("gamma")]
            return f"{g[0]:+.6f}" if g else "NA"
        except Exception:
            pass
    return "NA"


def main():
    phase3_dir = Path("artifacts/phase3")
    phase3_dir.mkdir(parents=True, exist_ok=True)
    runs_target = phase3_dir / "runs"
    runs_target.mkdir(parents=True, exist_ok=True)

    candidates = [
        {
            "run": "e4p__mosaic-close0__s0",
            "group": "E4",
            "model": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
            "seed": 0,
            "epochs": 100,
            "close_mosaic": 0,
            "clean_epochs_pct": "0.0",
        },
        {
            "run": "e4p__mosaic-close50__s0",
            "group": "E4",
            "model": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
            "seed": 0,
            "epochs": 100,
            "close_mosaic": 50,
            "clean_epochs_pct": "50.0",
        },
    ]

    summary_rows = []
    for c in candidates:
        rname = c["run"]
        r_dirs = [
            Path("runs/detect/runs") / rname,
            Path("runs") / rname,
        ]
        found_dir = None
        for rd in r_dirs:
            if (rd / "results.csv").exists():
                found_dir = rd
                break

        if not found_dir:
            print(f"[WARN] Run {rname} not found in runs/")
            continue

        # 1. Copy results.csv and args.yaml to artifacts/phase3/runs/<rname>/
        dest = runs_target / rname
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found_dir / "results.csv", dest / "results.csv")
        if (found_dir / "args.yaml").exists():
            shutil.copy2(found_dir / "args.yaml", dest / "args.yaml")

        # 2. Extract metrics
        df = pd.read_csv(found_dir / "results.csv")
        df.columns = [col.strip() for col in df.columns]
        m50_col, m95_col = "metrics/mAP50(B)", "metrics/mAP50-95(B)"
        best_i = df[m95_col].idxmax() if m95_col in df.columns else df[m50_col].idxmax()
        m50 = float(df[m50_col].iloc[best_i])
        m95 = float(df[m95_col].iloc[best_i])
        gamma_val = get_gamma(found_dir)

        row = {
            "run": rname,
            "group": c["group"],
            "model": c["model"],
            "seed": c["seed"],
            "epochs": c["epochs"],
            "close_mosaic": c["close_mosaic"],
            "clean_epochs_pct": c["clean_epochs_pct"],
            "mAP50": f"{m50:.4f}",
            "mAP50_95": f"{m95:.4f}",
            "gamma_final": gamma_val,
            "L_std_final": "NA",
        }
        summary_rows.append(row)

    # 3. Write artifacts/phase3/summary.csv
    out_csv = phase3_dir / "summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "run",
            "group",
            "model",
            "seed",
            "epochs",
            "close_mosaic",
            "clean_epochs_pct",
            "mAP50",
            "mAP50_95",
            "gamma_final",
            "L_std_final",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[SUCCESS] Wrote {len(summary_rows)} rows to {out_csv}")
    for r in summary_rows:
        print(r)


if __name__ == "__main__":
    main()
