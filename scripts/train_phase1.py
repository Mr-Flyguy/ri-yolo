# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Phase 1 Experiment Runner: Structural Ablation and Multi-Seed Validation.

Executes controlled runs for Article C1 (RSCI):
- E3': Depth/Integration Position Ablation (5 runs, seed 0, 100 epochs)
- E2': Multi-Seed Validation (5 runs: 2 baseline seeds + 3 RFD seeds, 100 epochs)
"""

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.rfd_callbacks import add_rfd_logging
except ImportError:
    from rfd_callbacks import add_rfd_logging

from ultralytics import YOLO


# Complete Phase 1 experiment catalog
PHASE1_EXPERIMENTS = {
    # --- E3' Ablation of RFD insertion depth ---
    "e3p__pos-p3__s0": {
        "group": "E3",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-p3.yaml",
        "seed": 0,
        "desc": "RFD integrated after P3 (C2f-4)",
    },
    "e3p__pos-p4__s0": {
        "group": "E3",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-p4.yaml",
        "seed": 0,
        "desc": "RFD integrated after P4 (C2f-6)",
    },
    "e3p__pos-presppf__s0": {
        "group": "E3",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-presppf.yaml",
        "seed": 0,
        "desc": "RFD integrated before SPPF (layer 9)",
    },
    "e3p__pos-postsppf__s0": {
        "group": "E3",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "seed": 0,
        "desc": "RFD integrated after SPPF (canonical base, layer 10)",
    },
    "e3p__capctrl__s0": {
        "group": "E3",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-capctrl.yaml",
        "seed": 0,
        "desc": "Capacity control module (RFDBlockNoGate, constant 0.5 gate)",
    },

    # --- E2' Multi-Seed Validation ---
    "e2p__baseline__s1": {
        "group": "E2",
        "cfg": "yolov8s.yaml",
        "seed": 1,
        "desc": "Standard YOLOv8s baseline (seed 1)",
    },
    "e2p__baseline__s2": {
        "group": "E2",
        "cfg": "yolov8s.yaml",
        "seed": 2,
        "desc": "Standard YOLOv8s baseline (seed 2)",
    },
    "e2p__postsppf__s1": {
        "group": "E2",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "seed": 1,
        "desc": "YOLOv8s-RFD post-SPPF (seed 1)",
    },
    "e2p__postsppf__s2": {
        "group": "E2",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "seed": 2,
        "desc": "YOLOv8s-RFD post-SPPF (seed 2)",
    },
    "e2p__postsppf__s3": {
        "group": "E2",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "seed": 3,
        "desc": "YOLOv8s-RFD post-SPPF (seed 3)",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1 Experiment Runner (RI-YOLO)")
    parser.add_argument(
        "--run",
        type=str,
        default="e3p__pos-postsppf__s0",
        help="Run identifier, or group: 'e3', 'e2', 'all', or specific run name",
    )
    parser.add_argument("--data", type=str, default="exdark.yaml", help="Path to dataset YAML file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", default="", help="Device: '0', '0,1', 'cpu'")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--weights", type=str, default="yolov8s.pt", help="Pretrained weights file")
    parser.add_argument("--project", type=str, default="runs", help="Save directory root")
    parser.add_argument("--resume", action="store_true", help="Resume training from last.pt if exists")
    return parser.parse_args()


def train_single_run(run_name, exp_info, args):
    print("\n" + "=" * 80)
    print(f"  LAUNCHING RUN: {run_name}")
    print(f"  Group        : {exp_info['group']}")
    print(f"  Description  : {exp_info['desc']}")
    print(f"  Model config : {exp_info['cfg']}")
    print(f"  Seed         : {exp_info['seed']}")
    print(f"  Epochs       : {args.epochs} | Batch: {args.batch} | Imgsz: {args.imgsz}")
    print("=" * 80 + "\n")

    run_dir = Path(args.project) / run_name
    last_ckpt = run_dir / "weights" / "last.pt"

    # Check if already completed
    results_csv = run_dir / "results.csv"
    if results_csv.exists() and not args.resume:
        with open(results_csv) as f:
            lines = [line.strip() for line in f if line.strip()]
        if len(lines) >= args.epochs + 1:  # header + epochs
            print(f"[SKIP] Run {run_name} already completed ({len(lines)-1} epochs found).")
            return

    # Check for resume
    if args.resume and last_ckpt.exists():
        print(f"[INFO] Resuming training from {last_ckpt}...")
        model = YOLO(str(last_ckpt))
        model.train(resume=True)
        return

    # Initialize model
    model = YOLO(exp_info["cfg"])

    # Register RFD logging callback
    model.add_callback("on_fit_epoch_end", add_rfd_logging)

    # Load pretrained backbone/head weights
    if args.weights and Path(args.weights).exists():
        print(f"[INFO] Transferring weights from {args.weights}...")
        model.load(args.weights)
    elif args.weights:
        from ultralytics.utils.downloads import attempt_download_asset
        attempt_download_asset(args.weights)
        if Path(args.weights).exists():
            print(f"[INFO] Transferring weights from downloaded {args.weights}...")
            model.load(args.weights)

    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "project": args.project,
        "name": run_name,
        "seed": exp_info["seed"],
        "deterministic": True,
        "workers": args.workers,
        "exist_ok": True,
        "save": True,
        "plots": True,
    }
    if args.device:
        train_kwargs["device"] = args.device

    # Launch training
    results = model.train(**train_kwargs)

    # Post-training summary logging
    summary_path = Path("artifacts/phase1/summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not summary_path.exists()

    # Extract metrics
    map50 = 0.0
    map50_95 = 0.0
    if results is not None:
        if hasattr(results, "box") and results.box is not None:
            map50 = float(getattr(results.box, "map50", 0.0))
            map50_95 = float(getattr(results.box, "map", 0.0))
        elif hasattr(results, "results_dict"):
            map50 = float(results.results_dict.get("metrics/mAP50(B)", 0.0))
            map50_95 = float(results.results_dict.get("metrics/mAP50-95(B)", 0.0))

    # Extract final gamma if present
    gamma_val = "NA"
    for m in model.model.modules():
        if m.__class__.__name__ in ("RFDBlock", "RFDBlockNoGate"):
            if hasattr(m, "gamma"):
                gamma_val = f"{float(m.gamma.detach().cpu().reshape(-1)[0]):+.6f}"
                break

    row = {
        "run": run_name,
        "group": exp_info["group"],
        "model": exp_info["cfg"],
        "seed": exp_info["seed"],
        "epochs": args.epochs,
        "mAP50": f"{map50:.4f}",
        "mAP50_95": f"{map50_95:.4f}",
        "gamma_final": gamma_val,
    }

    # Update or append row in summary.csv (prevents duplicate entries from smoke tests)
    existing_rows = []
    if summary_path.exists():
        with open(summary_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = [r for r in reader if r.get("run") != run_name]
    existing_rows.append(row)

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerows(existing_rows)

    print(f"[SUCCESS] Completed {run_name}! mAP50: {map50:.4f}, mAP50-95: {map50_95:.4f}, gamma: {gamma_val}")


def main():
    args = parse_args()
    target = args.run.lower().strip()

    if target in ("all", "phase1"):
        selected_runs = list(PHASE1_EXPERIMENTS.keys())
    elif target in ("e3", "e3p"):
        selected_runs = [k for k, v in PHASE1_EXPERIMENTS.items() if v["group"] == "E3"]
    elif target in ("e2", "e2p"):
        selected_runs = [k for k, v in PHASE1_EXPERIMENTS.items() if v["group"] == "E2"]
    elif target in PHASE1_EXPERIMENTS:
        selected_runs = [target]
    else:
        print(f"[ERROR] Unknown run target: '{args.run}'")
        print("Available options:")
        print("  --run e3                 (Run all 5 E3' ablation runs)")
        print("  --run e2                 (Run all 5 E2' multi-seed runs)")
        print("  --run all                (Run all 10 Phase 1 runs)")
        print("  --run <run_name>         (Specific run from below:)")
        for k, v in PHASE1_EXPERIMENTS.items():
            print(f"    - {k} ({v['desc']})")
        sys.exit(1)

    print(f"[INFO] Selected {len(selected_runs)} experiment(s) to execute.")
    for run_name in selected_runs:
        train_single_run(run_name, PHASE1_EXPERIMENTS[run_name], args)


if __name__ == "__main__":
    main()
