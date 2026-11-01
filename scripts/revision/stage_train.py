# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Stage Train: Repeated Training Runs for Reviewer R1.2.

Executes additional training schedules for baseline and RFD post-SPPF:
- Mandatory (8 runs): ep40 & ep60, close_mosaic=10, seeds 1, 2 (seed 0 already exists)
- Priority 2 (--with-mosaic, 4 runs): ep100, close_mosaic=0 & close_mosaic=50, seeds 1, 2
Includes strict idempotency, resume capability, and dry-run time estimation.
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional

import torch

# Ensure repo root is in sys.path when executed directly
REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from scripts.revision.config import (  # noqa: E402
    CONTROL_EP40_SGD_RUNS,
    DEFAULT_DATA_YAML,
    DEFAULT_SMOKE_DIR,
    HISTORICAL_EPOCH_TIME_SEC,
    MANDATORY_NEW_RUNS,
    MOSAIC_PRIORITY2_RUNS,
    REPO_ROOT,
    REPRO_CHECK_RUN,
    init_nms_backend,
    resolve_data_yaml,
)


def estimate_run_duration(epochs: int, epoch_sec: float = HISTORICAL_EPOCH_TIME_SEC) -> float:
    """Estimate training duration in minutes."""
    return round((epochs * epoch_sec) / 60.0, 1)


def is_run_completed(run_name: str, target_epochs: int, runs_dir: Path = REPO_ROOT / "runs/detect/runs") -> bool:
    """Check if a run is already finished with best.pt and complete results.csv."""
    run_dir = runs_dir / run_name
    if not run_dir.exists():
        run_dir = REPO_ROOT / "runs" / run_name
    if not run_dir.exists():
        return False

    best_pt = run_dir / "weights" / "best.pt"
    results_csv = run_dir / "results.csv"

    if not best_pt.is_file() or not results_csv.is_file():
        return False

    # Check results.csv lines
    try:
        with open(results_csv, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        # Header + target_epochs
        if len(lines) >= target_epochs + 1:
            return True
    except Exception:
        pass

    return best_pt.is_file()


def print_training_plan(
    runs: List[Dict],
    runs_dir: Path,
    with_mosaic: bool = False,
    with_ep40sgd: bool = True,
    repro_check: bool = False,
):
    """Print training schedule table and estimated time."""
    print("=" * 96)
    print("  STAGE TRAIN: REPEATED SCHEDULES & CONTROL EXECUTION PLAN (R1.2)")
    print("=" * 96)

    total_epochs = 0
    pending_epochs = 0
    rows = []

    for idx, r in enumerate(runs, 1):
        completed = is_run_completed(r["run"], r["epochs"], runs_dir)
        total_epochs += r["epochs"]
        if not completed:
            pending_epochs += r["epochs"]

        dur_est = estimate_run_duration(r["epochs"])
        status = "[DONE]" if completed else "[PENDING]"
        opt_str = r.get("optimizer", "auto")
        lr_str = str(r.get("lr0", "auto"))
        rows.append(
            (
                idx,
                r["run"],
                r["family"],
                r["epochs"],
                opt_str,
                lr_str,
                r["close_mosaic"],
                r["seed"],
                f"{dur_est} min",
                status,
            )
        )

    print(
        f"{'#':<3} {'Run Name':<30} {'Family':<9} {'Ep':<4} {'Opt':<7} {'lr0':<9} {'CloseMos':<9} {'Seed':<5} {'Est. Time':<10} {'Status':<8}"
    )
    print("-" * 96)
    for r in rows:
        print(f"{r[0]:<3} {r[1]:<30} {r[2]:<9} {r[3]:<4} {r[4]:<7} {r[5]:<9} {r[6]:<9} {r[7]:<5} {r[8]:<10} {r[9]:<8}")

    total_min = round((total_epochs * HISTORICAL_EPOCH_TIME_SEC) / 60.0, 1)
    pending_min = round((pending_epochs * HISTORICAL_EPOCH_TIME_SEC) / 60.0, 1)
    total_hr = round(total_min / 60.0, 2)
    pending_hr = round(pending_min / 60.0, 2)

    print("-" * 96)
    print(
        f"Total runs: {len(runs)} ({len(MANDATORY_NEW_RUNS)} mandatory, "
        f"{len(CONTROL_EP40_SGD_RUNS) if with_ep40sgd else 0} ep40-sgd control, "
        f"{len(MOSAIC_PRIORITY2_RUNS) if with_mosaic else 0} mosaic, "
        f"{1 if repro_check else 0} repro-check)"
    )
    print(f"Total workload   : {total_epochs} epochs (~{total_hr} hours @ {HISTORICAL_EPOCH_TIME_SEC}s/epoch)")
    print(f"Remaining work   : {pending_epochs} epochs (~{pending_hr} hours)")
    print("=" * 96 + "\n")


def execute_single_run(
    exp_info: Dict,
    data_yaml: Path,
    device: str,
    runs_dir: Path,
    smoke: bool = False,
    force: bool = False,
):
    """Execute training for a single run using Ultralytics YOLO API with RFD callback."""
    run_name = exp_info["run"]
    epochs = 1 if smoke else exp_info["epochs"]
    batch = 4 if smoke else 16
    close_mosaic = 0 if smoke else exp_info["close_mosaic"]

    run_dir = runs_dir / run_name
    best_pt = run_dir / "weights" / "best.pt"

    if not force and not smoke and is_run_completed(run_name, epochs, runs_dir):
        print(f"[SKIP] Run '{run_name}' already completed. Found: {best_pt}")
        return

    print("\n" + "=" * 80)
    print(f"  STARTING RUN: {run_name}")
    print(f"  Model YAML   : {exp_info['cfg']}")
    print(f"  Epochs       : {epochs} | Batch: {batch} | Seed: {exp_info['seed']} | CloseMosaic: {close_mosaic}")
    print(f"  Device       : {device}")
    print(f"  Save Dir     : {run_dir}")
    print("=" * 80 + "\n")

    # Import ultralytics & RFD callbacks lazily
    from scripts.rfd_callbacks import add_rfd_logging
    from ultralytics import YOLO

    # Check for resume
    last_pt = run_dir / "weights" / "last.pt"
    if last_pt.is_file() and not force:
        print(f"[INFO] Resuming training from existing checkpoint: {last_pt}")
        model = YOLO(str(last_pt))
        model.train(resume=True)
        return

    # Load configuration
    model = YOLO(exp_info["cfg"])

    # Register RFD logging callback for gamma and L tracking
    model.add_callback("on_fit_epoch_end", add_rfd_logging)

    # Pretrained COCO weights (transfer backbone/head)
    weights_path = REPO_ROOT / "weights" / "yolov8s.pt"
    if weights_path.is_file():
        print(f"[INFO] Initializing from pretrained weights: {weights_path}")
        model.load(str(weights_path))
    else:
        # Check root or download
        root_pt = REPO_ROOT / "yolov8s.pt"
        if root_pt.is_file():
            print(f"[INFO] Initializing from pretrained weights: {root_pt}")
            model.load(str(root_pt))
        else:
            try:
                from ultralytics.utils.downloads import attempt_download_asset

                attempt_download_asset("yolov8s.pt")
                if Path("yolov8s.pt").is_file():
                    model.load("yolov8s.pt")
            except Exception as e:
                print(f"[WARN] Could not download pretrained weights: {e}")

    train_kwargs = {
        "data": str(data_yaml),
        "epochs": epochs,
        "batch": batch,
        "imgsz": 640,
        "project": str(runs_dir),
        "name": run_name,
        "seed": exp_info["seed"],
        "deterministic": True,
        "workers": 0 if smoke else 8,
        "exist_ok": True,
        "save": True,
        "plots": True,
        "close_mosaic": close_mosaic,
        "device": device,
    }

    if exp_info.get("optimizer"):
        train_kwargs["optimizer"] = exp_info["optimizer"]
    if exp_info.get("lr0"):
        train_kwargs["lr0"] = exp_info["lr0"]

    if smoke:
        train_kwargs["fraction"] = 0.02

    # Launch training
    start_t = time.time()
    model.train(**train_kwargs)
    elapsed = round((time.time() - start_t) / 60.0, 2)

    print(f"\n[COMPLETED] Run '{run_name}' in {elapsed} minutes. Output: {run_dir}")


def run_stage_train(
    data_yaml: Path,
    device: str = "0",
    with_mosaic: bool = False,
    with_ep40sgd: bool = True,
    repro_check: bool = False,
    dry_run: bool = False,
    smoke: bool = False,
    force: bool = False,
    runs_dir: Optional[Path] = None,
):
    """Orchestrate training of all new revision schedules."""
    if not dry_run or smoke:
        init_nms_backend(smoke=smoke)

    if runs_dir is None:
        runs_dir = DEFAULT_SMOKE_DIR / "runs" if smoke else REPO_ROOT / "runs" / "detect" / "runs"

    runs_dir.mkdir(parents=True, exist_ok=True)

    scheduled_runs = list(MANDATORY_NEW_RUNS)
    if with_ep40sgd:
        scheduled_runs.extend(CONTROL_EP40_SGD_RUNS)
    if with_mosaic:
        scheduled_runs.extend(MOSAIC_PRIORITY2_RUNS)
    if repro_check:
        scheduled_runs.append(REPRO_CHECK_RUN)
    if smoke:
        scheduled_runs = scheduled_runs[:1]

    # Always print plan first
    print_training_plan(
        scheduled_runs,
        runs_dir,
        with_mosaic=with_mosaic,
        with_ep40sgd=with_ep40sgd,
        repro_check=repro_check,
    )

    if dry_run:
        print("[DRY-RUN] Execution plan displayed. Skipping model training as requested.")
        return

    # Execute runs
    for run_info in scheduled_runs:
        execute_single_run(
            exp_info=run_info,
            data_yaml=data_yaml,
            device=device,
            runs_dir=runs_dir,
            smoke=smoke,
            force=force,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage Train: Multi-seed schedule training.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_YAML), help="Path to dataset yaml")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu", help="Device index or 'cpu'")
    parser.add_argument(
        "--with-mosaic", action="store_true", help="Include priority 2 mosaic runs (close_mosaic=0, 50)"
    )
    parser.add_argument(
        "--no-ep40sgd", action="store_true", help="Disable control series ep40-sgd (enabled by default)"
    )
    parser.add_argument(
        "--repro-check", action="store_true", help="Retrain e5p__baseline-ep40__s0 as e9p to check reproducibility"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan and estimated time without training")
    parser.add_argument("--smoke", action="store_true", help="Smoke test on tiny dataset/CPU")
    parser.add_argument("--force", action="store_true", help="Force re-train already completed runs")
    args = parser.parse_args()

    run_stage_train(
        data_yaml=resolve_data_yaml(args.data),
        device=args.device,
        with_mosaic=args.with_mosaic,
        with_ep40sgd=not args.no_ep40sgd,
        repro_check=args.repro_check,
        dry_run=args.dry_run,
        smoke=args.smoke,
        force=args.force,
    )
