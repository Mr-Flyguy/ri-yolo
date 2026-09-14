# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Phase 3 Experiment Runner: System Synergy, Mosaic Augmentation, and Sample Efficiency.

Executes controlled runs for Article C3 (IEEE / Scopus Q1):
- E4': Mosaic augmentation sensitivity ablation (2 runs: close_mosaic=0, close_mosaic=50; seed 0, 100 epochs)
- E5': Duration and sample efficiency ablation (4 runs: baseline 40/60 ep, postsppf 40/60 ep; seed 0, close_mosaic=10)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rfd_callbacks import add_rfd_logging  # noqa: E402
from ultralytics import YOLO  # noqa: E402

# Complete Phase 3 experiment catalog (Runs 18-23 of 23)
PHASE3_EXPERIMENTS = {
    # --- E4' Mosaic Augmentation Sensitivity (seed 0, 100 epochs) ---
    "e4p__mosaic-close0__s0": {
        "group": "E4",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "seed": 0,
        "epochs": 100,
        "close_mosaic": 0,
        "desc": "RFD post-SPPF with mosaic enabled across all 100 epochs (close_mosaic=0)",
        "hyp_overrides": {},
    },
    "e4p__mosaic-close50__s0": {
        "group": "E4",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "seed": 0,
        "epochs": 100,
        "close_mosaic": 50,
        "desc": "RFD post-SPPF with mosaic disabled for last 50 epochs (close_mosaic=50)",
        "hyp_overrides": {},
    },
    # --- E5' Sample Efficiency / Duration Ablation (seed 0, close_mosaic=10) ---
    "e5p__baseline-ep40__s0": {
        "group": "E5",
        "cfg": "yolov8s.yaml",
        "seed": 0,
        "epochs": 40,
        "close_mosaic": 10,
        "desc": "YOLOv8s baseline sample efficiency at 40 epochs",
        "hyp_overrides": {},
    },
    "e5p__baseline-ep60__s0": {
        "group": "E5",
        "cfg": "yolov8s.yaml",
        "seed": 0,
        "epochs": 60,
        "close_mosaic": 10,
        "desc": "YOLOv8s baseline sample efficiency at 60 epochs",
        "hyp_overrides": {},
    },
    "e5p__postsppf-ep40__s0": {
        "group": "E5",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "seed": 0,
        "epochs": 40,
        "close_mosaic": 10,
        "desc": "RFD post-SPPF sample efficiency at 40 epochs",
        "hyp_overrides": {},
    },
    "e5p__postsppf-ep60__s0": {
        "group": "E5",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "seed": 0,
        "epochs": 60,
        "close_mosaic": 10,
        "desc": "RFD post-SPPF sample efficiency at 60 epochs",
        "hyp_overrides": {},
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3 Experiment Runner (RI-YOLO)")
    parser.add_argument(
        "--run",
        type=str,
        default="all",
        help="Run identifier, or group: 'all', 'e4', 'e5', or specific run name",
    )
    parser.add_argument("--data", type=str, default="exdark.yaml", help="Path to dataset YAML file")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index (e.g. 0)")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epoch count (default: experiment-specific)",
    )
    parser.add_argument(
        "--close_mosaic",
        type=int,
        default=None,
        help="Override close_mosaic count (default: experiment-specific)",
    )
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers (default: 8)")
    parser.add_argument("--project", type=str, default="runs", help="Output project directory")
    parser.add_argument("--weights", type=str, default="weights/yolov8s.pt", help="Pretrained weights checkpoint")
    return parser.parse_args()


def run_experiment(run_name: str, exp_info: dict, args):
    epochs = args.epochs if args.epochs is not None else exp_info["epochs"]
    close_mosaic = args.close_mosaic if args.close_mosaic is not None else exp_info["close_mosaic"]
    clean_pct = round(close_mosaic / epochs * 100.0, 1) if epochs > 0 else 0.0

    print(f"\n{'='*70}")
    print(f"[START] Running Phase 3 Experiment: {run_name}")
    print(f"Group:            {exp_info['group']}")
    print(f"Description:      {exp_info['desc']}")
    print(f"Model Config:     {exp_info['cfg']}")
    print(f"Seed:             {exp_info['seed']}")
    print(f"Epochs:           {epochs}")
    print(f"Close Mosaic:     {close_mosaic} ({clean_pct}% clean epochs)")
    print(f"{'='*70}\n")

    # Initialize model
    model = YOLO(exp_info["cfg"])
    model.add_callback("on_fit_epoch_end", add_rfd_logging)

    # Load pretrained weights
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
        "epochs": epochs,
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
        "close_mosaic": close_mosaic,
    }
    if args.device:
        train_kwargs["device"] = args.device

    # Apply hyperparameter overrides if any
    if "hyp_overrides" in exp_info:
        train_kwargs.update(exp_info["hyp_overrides"])

    # Launch training
    results = model.train(**train_kwargs)

    # Post-training summary logging
    summary_path = Path("artifacts/phase3/summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

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

    # Extract final L_std from rfd_log.csv if available
    l_std_final = "NA"
    rfd_log_file = Path(args.project) / run_name / "rfd_log.csv"
    if rfd_log_file.exists():
        try:
            with open(rfd_log_file, "r") as rf:
                r_reader = list(csv.DictReader(rf))
                if r_reader and "L_std_spatial" in r_reader[-1]:
                    l_std_final = f"{float(r_reader[-1]['L_std_spatial']):.4f}"
        except Exception as e:
            print(f"[WARN] Could not parse rfd_log.csv: {e}")

    row = {
        "run": run_name,
        "group": exp_info["group"],
        "model": exp_info["cfg"],
        "seed": exp_info["seed"],
        "epochs": epochs,
        "close_mosaic": close_mosaic,
        "clean_epochs_pct": f"{clean_pct:.1f}",
        "mAP50": f"{map50:.4f}",
        "mAP50_95": f"{map50_95:.4f}",
        "gamma_final": gamma_val,
        "L_std_final": l_std_final,
    }

    # Update or append row in summary.csv
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

    # Also save individual summary.json inside the run directory for roadmap compliance
    run_dir = Path(args.project) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_json_file = run_dir / "summary.json"
    try:
        json_data = {
            "run": run_name,
            "group": exp_info["group"],
            "model": exp_info["cfg"],
            "seed": exp_info["seed"],
            "epochs": epochs,
            "close_mosaic": close_mosaic,
            "clean_pct": clean_pct,
            "best_mAP50": map50,
            "best_mAP50_95": map50_95,
            "gamma": float(gamma_val) if gamma_val != "NA" else None,
            "L_std": float(l_std_final) if l_std_final != "NA" else None,
        }
        with open(summary_json_file, "w") as jf:
            json.dump(json_data, jf, indent=2)
    except Exception as e:
        print(f"[WARN] Could not write {summary_json_file}: {e}")

    print(
        f"[SUCCESS] Completed {run_name}! mAP50: {map50:.4f}, mAP50-95: {map50_95:.4f}, "
        f"gamma: {gamma_val}, L_std: {l_std_final}"
    )


def main():
    args = parse_args()
    target = args.run.lower().strip()

    if target in ("all", "phase3"):
        selected_runs = list(PHASE3_EXPERIMENTS.keys())
    elif target in ("e4", "e4p"):
        selected_runs = [k for k, v in PHASE3_EXPERIMENTS.items() if v["group"] == "E4"]
    elif target in ("e5", "e5p"):
        selected_runs = [k for k, v in PHASE3_EXPERIMENTS.items() if v["group"] == "E5"]
    elif target in PHASE3_EXPERIMENTS:
        selected_runs = [target]
    else:
        print(f"[ERROR] Unknown run or group: '{args.run}'.")
        print("Available runs/groups:")
        print("  --run all      (Run all 6 Phase 3 experiments: E4' + E5')")
        print("  --run e4       (Run 2 E4' mosaic ablation experiments)")
        print("  --run e5       (Run 4 E5' duration/sample efficiency experiments)")
        for k in sorted(PHASE3_EXPERIMENTS.keys()):
            print(f"  - {k}")
        sys.exit(1)

    print(f"[INFO] Phase 3 Execution Plan: {len(selected_runs)} run(s) queued.")
    for idx, r_name in enumerate(selected_runs, 1):
        print(f"  {idx}. {r_name} ({PHASE3_EXPERIMENTS[r_name]['desc']})")

    for r_name in selected_runs:
        run_experiment(r_name, PHASE3_EXPERIMENTS[r_name], args)

    print(f"\n[ALL DONE] Finished {len(selected_runs)} run(s) for Phase 3!")


if __name__ == "__main__":
    main()
