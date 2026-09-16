# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Phase 2 Experiment Runner: Retinex Decomposition Regularization and NWD Metric Formulations.

Executes controlled runs for Article C2 (VAK / RSCI):
- E6': Regularization weight lambda_tv grid sweep (4 runs: 1e-4, 1e-3, 1e-2, 1e-1, seed 0, 100 epochs)
- E7': Bounding box metric formulation variants (3 runs: abs C=10.24, scaleinv C=0.3, sizegate C=10.24 tau=5.12, seed 0, 100 epochs)
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rfd_callbacks import add_rfd_logging  # noqa: E402
from ultralytics import YOLO  # noqa: E402

# Complete Phase 2 experiment catalog (Runs 11-17 of 23)
PHASE2_EXPERIMENTS = {
    # --- E6' TV-loss Regularization Weight Sweep (seed 0, 100 epochs) ---
    "e6p__lam-1e-4__s0": {
        "group": "E6",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "seed": 0,
        "desc": "TV-loss regularization lambda=1e-4 (0.0001)",
        "hyp_overrides": {
            "use_rsl": True,
            "lambda_tv": 0.0001,
            "use_nwd": False,
        },
    },
    "e6p__lam-1e-3__s0": {
        "group": "E6",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "seed": 0,
        "desc": "TV-loss regularization lambda=1e-3 (0.001, canonical)",
        "hyp_overrides": {
            "use_rsl": True,
            "lambda_tv": 0.001,
            "use_nwd": False,
        },
    },
    "e6p__lam-1e-2__s0": {
        "group": "E6",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "seed": 0,
        "desc": "TV-loss regularization lambda=1e-2 (0.01)",
        "hyp_overrides": {
            "use_rsl": True,
            "lambda_tv": 0.01,
            "use_nwd": False,
        },
    },
    "e6p__lam-1e-1__s0": {
        "group": "E6",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "seed": 0,
        "desc": "TV-loss regularization lambda=1e-1 (0.1)",
        "hyp_overrides": {
            "use_rsl": True,
            "lambda_tv": 0.1,
            "use_nwd": False,
        },
    },

    # --- E7' Bounding Box Metric Formulations (seed 0, 100 epochs) ---
    "e7p__nwd-calib__s0": {
        "group": "E7",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "seed": 0,
        "desc": "Calibrated NWD mode=abs C=10.24 (P4-stride anchor)",
        "hyp_overrides": {
            "use_rsl": True,
            "lambda_tv": 0.001,
            "use_nwd": True,
            "nwd_alpha": 0.2,
            "nwd_mode": "abs",
            "nwd_c": 10.24,
        },
    },
    "e7p__nwd-scaleinv__s0": {
        "group": "E7",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "seed": 0,
        "desc": "Scale-invariant NWD mode=scaleinv C=0.3",
        "hyp_overrides": {
            "use_rsl": True,
            "lambda_tv": 0.001,
            "use_nwd": True,
            "nwd_alpha": 0.2,
            "nwd_mode": "scaleinv",
            "nwd_c": 0.3,
        },
    },
    "e7p__nwd-sizegate__s0": {
        "group": "E7",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "seed": 0,
        "desc": "Size-gated NWD mode=sizegate C=10.24 tau=5.12",
        "hyp_overrides": {
            "use_rsl": True,
            "lambda_tv": 0.001,
            "use_nwd": True,
            "nwd_alpha": 0.2,
            "nwd_mode": "sizegate",
            "nwd_c": 10.24,
            "size_tau": 5.12,
        },
    },

    # --- Block 4 Updated Runs (Decisions C2 / Claude Opus 16.09.2026) ---
    "e7p__nwd-calib-norsl__s0": {
        "group": "E7",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "seed": 0,
        "desc": "Calibrated NWD without RSL (isolated metric on live gate)",
        "hyp_overrides": {
            "use_rsl": False,
            "lambda_tv": 0.0,
            "use_nwd": True,
            "nwd_alpha": 0.2,
            "nwd_mode": "abs",
            "nwd_c": 10.24,
        },
    },
    "e6p__lam-1e-4__s1": {
        "group": "E6",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "seed": 1,
        "desc": "Partially live gate multi-seed seed 1 (lambda=1e-4)",
        "hyp_overrides": {
            "use_rsl": True,
            "lambda_tv": 0.0001,
            "use_nwd": False,
        },
    },
    "e6p__lam-1e-4__s2": {
        "group": "E6",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "seed": 2,
        "desc": "Partially live gate multi-seed seed 2 (lambda=1e-4)",
        "hyp_overrides": {
            "use_rsl": True,
            "lambda_tv": 0.0001,
            "use_nwd": False,
        },
    },
    "e6p__lam-1e-5__s0": {
        "group": "E6",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "seed": 0,
        "desc": "Boundary collapse check (lambda=1e-5, seed 0)",
        "hyp_overrides": {
            "use_rsl": True,
            "lambda_tv": 0.00001,
            "use_nwd": False,
        },
    },
    "e7p__nwd-calib__s1": {
        "group": "E7",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "seed": 1,
        "desc": "Calibrated NWD seed 1",
        "hyp_overrides": {
            "use_rsl": True,
            "lambda_tv": 0.001,
            "use_nwd": True,
            "nwd_alpha": 0.2,
            "nwd_mode": "abs",
            "nwd_c": 10.24,
        },
    },
    "e7p__nwd-calib__s2": {
        "group": "E7",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "seed": 2,
        "desc": "Calibrated NWD seed 2",
        "hyp_overrides": {
            "use_rsl": True,
            "lambda_tv": 0.001,
            "use_nwd": True,
            "nwd_alpha": 0.2,
            "nwd_mode": "abs",
            "nwd_c": 10.24,
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 Experiment Runner (RI-YOLO)")
    parser.add_argument(
        "--run",
        type=str,
        default="e6",
        help="Run identifier, or group: 'e6', 'e7', 'all', or specific run name",
    )
    parser.add_argument("--data", type=str, default="exdark.yaml", help="Path to dataset YAML file")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index (e.g. 0)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epoch count (default: 100)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers (default: 8)")
    parser.add_argument("--project", type=str, default="runs", help="Output project directory")
    parser.add_argument("--weights", type=str, default="weights/yolov8s.pt", help="Pretrained weights checkpoint")
    parser.add_argument(
        "--lambda_tv",
        type=float,
        default=None,
        help="Optimal lambda_tv determined from E6 sweep (used for E7 runs)",
    )
    return parser.parse_args()


def run_experiment(run_name: str, exp_info: dict, args):
    print(f"\n{'='*70}")
    print(f"[START] Running Phase 2 Experiment: {run_name}")
    print(f"Group:       {exp_info['group']}")
    print(f"Description: {exp_info['desc']}")
    print(f"Model Config:{exp_info['cfg']}")
    print(f"Overrides:   {exp_info.get('hyp_overrides', {})}")
    print(f"Seed:        {exp_info['seed']}")
    print(f"Epochs:      {args.epochs}")
    print(f"{'='*70}\n")

    # Initialize model
    model = YOLO(exp_info["cfg"])
    model.add_callback("on_fit_epoch_end", add_rfd_logging)

    # Load pretrained weights
    if args.weights and Path(args.weights).exists():
        import hashlib

        h = hashlib.md5(open(args.weights, "rb").read()).hexdigest()
        print(f"[INIT] weights={args.weights} md5={h}")
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
        "close_mosaic": 10,
    }
    if args.device:
        train_kwargs["device"] = args.device

    # Apply hyperparameter overrides (use_rsl, lambda_tv, use_nwd, etc.)
    if "hyp_overrides" in exp_info:
        overrides = dict(exp_info["hyp_overrides"])
        # For E7 runs, dynamically resolve optimal lambda_tv from E6' (only if use_rsl is True)
        if exp_info["group"] == "E7" and overrides.get("use_rsl", True):
            chosen_lambda = None
            if args.lambda_tv is not None:
                chosen_lambda = args.lambda_tv
                print(f"[INFO] Using explicitly provided lambda_tv={chosen_lambda} for {run_name}")
            else:
                summary_p = Path("artifacts/phase2/summary.csv")
                if summary_p.exists():
                    try:
                        with open(summary_p, "r") as sf:
                            e6_rows = [r for r in csv.DictReader(sf) if r.get("group") == "E6" and r.get("lambda_tv") not in ("NA", "")]
                            if e6_rows:
                                best_e6 = max(e6_rows, key=lambda x: float(x.get("mAP50", 0.0)))
                                chosen_lambda = float(best_e6["lambda_tv"])
                                print(f"[INFO] Auto-detected best lambda_tv={chosen_lambda} from E6' ({best_e6['run']}, mAP50={best_e6['mAP50']})")
                    except Exception as e:
                        print(f"[WARN] Error reading E6' summary: {e}")
            if chosen_lambda is not None:
                overrides["lambda_tv"] = chosen_lambda
            else:
                print(f"[WARN] No E6' results found and --lambda_tv not set. Falling back to default lambda_tv=0.001 for {run_name}")
        train_kwargs.update(overrides)

    # Launch training
    results = model.train(**train_kwargs)

    # Post-training summary logging
    summary_path = Path("artifacts/phase2/summary.csv")
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
        "epochs": args.epochs,
        "lambda_tv": train_kwargs.get("lambda_tv", "NA"),
        "use_rsl": train_kwargs.get("use_rsl", False),
        "use_nwd": train_kwargs.get("use_nwd", False),
        "nwd_mode": train_kwargs.get("nwd_mode", "NA"),
        "nwd_c": train_kwargs.get("nwd_c", "NA"),
        "size_tau": train_kwargs.get("size_tau", "NA"),
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

    print(f"[SUCCESS] Completed {run_name}! mAP50: {map50:.4f}, mAP50-95: {map50_95:.4f}, gamma: {gamma_val}, L_std: {l_std_final}")


def main():
    args = parse_args()
    target = args.run.lower().strip()

    if target in ("all", "phase2"):
        print("[ERROR] Simultaneous execution of all Phase 2 experiments (--run all) is prohibited!")
        print("According to RI_YOLO_Technical_Roadmap.md (Section 2), E7' (NWD variants) strictly")
        print("depends on the optimal lambda_tv determined from E6' results.")
        print("\n[ACTION] Please execute E6' first:")
        print("  python scripts/train_phase2.py --run e6 --data exdark.yaml --device 0 --epochs 100")
        print("\nAfter analyzing E6' results, proceed to E7' with the winning lambda_tv:")
        print("  python scripts/train_phase2.py --run e7 --lambda_tv <winner> --data exdark.yaml --device 0 --epochs 100")
        sys.exit(1)
    elif target in ("e6", "e6p"):
        selected_runs = [k for k, v in PHASE2_EXPERIMENTS.items() if v["group"] == "E6"]
    elif target in ("e7", "e7p"):
        selected_runs = [k for k, v in PHASE2_EXPERIMENTS.items() if v["group"] == "E7"]
    elif target in PHASE2_EXPERIMENTS:
        selected_runs = [target]
    else:
        print(f"[ERROR] Unknown run or group: '{args.run}'.")
        print("Available runs/groups: 'all', 'e6', 'e7', or specific:")
        for k in sorted(PHASE2_EXPERIMENTS.keys()):
            print(f"  - {k}")
        sys.exit(1)

    print(f"[INFO] Phase 2 Execution Plan: {len(selected_runs)} run(s) queued.")
    for idx, r_name in enumerate(selected_runs, 1):
        print(f"  {idx}. {r_name} ({PHASE2_EXPERIMENTS[r_name]['desc']})")

    for r_name in selected_runs:
        run_experiment(r_name, PHASE2_EXPERIMENTS[r_name], args)

    print(f"\n[ALL DONE] Finished {len(selected_runs)} run(s) for Phase 2!")


if __name__ == "__main__":
    main()
