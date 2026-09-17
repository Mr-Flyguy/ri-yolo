# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Track B: Test validation on ExDark test split."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import torch
import pandas as pd
from ultralytics import YOLO

RUN_SPECS = [
    ("e2p__baseline__s0", ["runs/detect/runs/e2p__baseline__s0/weights/best.pt", "runs/e2p__baseline__s0/weights/best.pt"]),
    ("e2p__baseline__s1", ["runs/detect/runs/e2p__baseline__s1/weights/best.pt", "runs/e2p__baseline__s1/weights/best.pt"]),
    ("e2p__baseline__s2", ["runs/detect/runs/e2p__baseline__s2/weights/best.pt", "runs/e2p__baseline__s2/weights/best.pt"]),
    ("e2p__baseline__s3", ["runs/detect/runs/e2p__baseline__s3/weights/best.pt", "runs/e2p__baseline__s3/weights/best.pt"]),
    ("e2p__postsppf__s0", ["runs/detect/runs/e3p__pos-postsppf__s0/weights/best.pt", "runs/e3p__pos-postsppf__s0/weights/best.pt", "runs/detect/runs/e2p__postsppf__s0/weights/best.pt", "runs/e2p__postsppf__s0/weights/best.pt"]),
    ("e2p__postsppf__s1", ["runs/detect/runs/e2p__postsppf__s1/weights/best.pt", "runs/e2p__postsppf__s1/weights/best.pt"]),
    ("e2p__postsppf__s2", ["runs/detect/runs/e2p__postsppf__s2/weights/best.pt", "runs/e2p__postsppf__s2/weights/best.pt"]),
    ("e2p__postsppf__s3", ["runs/detect/runs/e2p__postsppf__s3/weights/best.pt", "runs/e2p__postsppf__s3/weights/best.pt"]),
    ("e3p__capctrl__s0", ["runs/detect/runs/e3p__capctrl__s0/weights/best.pt", "runs/e3p__capctrl__s0/weights/best.pt"]),
    ("e3p__capctrl__s1", ["runs/detect/runs/e3p__capctrl__s1/weights/best.pt", "runs/e3p__capctrl__s1/weights/best.pt"]),
    ("e3p__capctrl__s2", ["runs/detect/runs/e3p__capctrl__s2/weights/best.pt", "runs/e3p__capctrl__s2/weights/best.pt"]),
    ("e6p__lam-1e-2__s0", ["runs/detect/runs/e6p__lam-1e-2__s0/weights/best.pt", "runs/e6p__lam-1e-2__s0/weights/best.pt"]),
    ("e7p__nwd-calib__s0", ["runs/detect/runs/e7p__nwd-calib__s0/weights/best.pt", "runs/e7p__nwd-calib__s0/weights/best.pt"]),
    ("e5p__postsppf-ep40__s0", ["runs/detect/runs/e5p__postsppf-ep40__s0/weights/best.pt", "runs/e5p__postsppf-ep40__s0/weights/best.pt"]),
]

def resolve_checkpoint(candidates, run_name):
    for c in candidates:
        if Path(c).is_file():
            return Path(c)
    return None

def main():
    device = "0" if torch.cuda.is_available() else "cpu"
    val_params_dict = {
        "data": "exdark.yaml",
        "imgsz": 640,
        "batch": 16,
        "conf": 0.001,
        "iou": 0.6,
        "max_det": 300,
        "rect": False,
        "split": "test",
        "device": device,
        "half": False,
        "plots": False,
        "save_json": True,
    }
    
    rows = []
    for run_name, candidates in RUN_SPECS:
        ckpt_path = resolve_checkpoint(candidates, run_name)
        if not ckpt_path:
            continue
            
        print(f"\n[RUN] Testing '{run_name}'")
        model = YOLO(str(ckpt_path))
        val_args = {**model.overrides, "mode": "val", **val_params_dict}
        
        validator = model._smart_load("validator")(args=val_args, _callbacks=model.callbacks)
        val_stats = validator(model=model.model)
        res = validator.metrics
        d = res.results_dict if hasattr(res, "results_dict") else {}
        
        map50 = float(d.get("metrics/mAP50(B)", getattr(res.box, "map50", 0.0)))
        map50_95 = float(d.get("metrics/mAP50-95(B)", getattr(res.box, "map", 0.0)))
        
        rows.append({
            "run": run_name,
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4)
        })
        
    df_test = pd.DataFrame(rows)
    os.makedirs("tables", exist_ok=True)
    df_test.to_csv("tables/C3_test_metrics.csv", index=False)
    print("\n[SUCCESS] Wrote tables/C3_test_metrics.csv")
    
    # Compare with validation
    val_p = Path("tables/C2_final_metrics.csv")
    if val_p.exists():
        df_val = pd.read_csv(val_p)
        merged = pd.merge(df_val[["run", "mAP50", "mAP50_95"]], df_test, on="run", suffixes=("_val", "_test"))
        
        merged["delta_pp"] = (merged["mAP50_val"] - merged["mAP50_test"]) * 100
        merged["delta_95_pp"] = (merged["mAP50_95_val"] - merged["mAP50_95_test"]) * 100
        
        merged.to_csv("tables/C3_val_vs_test.csv", index=False)
        print("[SUCCESS] Wrote tables/C3_val_vs_test.csv")
        print(f"\nAverage delta mAP50 (val - test) : {merged['delta_pp'].mean():.2f} pp")
        print(f"Average delta mAP50_95 (val - test): {merged['delta_95_pp'].mean():.2f} pp")

if __name__ == "__main__":
    main()
