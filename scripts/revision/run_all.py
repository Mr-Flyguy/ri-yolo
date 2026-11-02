# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Master Orchestrator for PIERE-2026 Revision Experiments.

Executes all revision stages with a single command:
  python scripts/revision/run_all.py --stage all
  ./scripts/revision/run_all.sh --stage all

Supports:
  --stage {info,train,eval,corr,illum,stats,report,all}
  --dry-run      : Inspect plan, run list, and time estimation without executing
  --smoke        : Quick pipeline test on CPU (1 epoch, isolated smoke directory)
  --force        : Re-execute completed runs and stages
  --with-mosaic  : Include priority 2 mosaic runs (close_mosaic=0, close_mosaic=50)
  --device       : Specify GPU device index or 'cpu'
  --data         : Path to dataset YAML (default: exdark.yaml)
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import time
from typing import Optional

# Ensure ultralytics is in sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.revision.config import (  # noqa: E402
    DEFAULT_DATA_YAML,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SMOKE_DIR,
    REPO_ROOT,
    REPRO_CHECK_RUN,
    init_nms_backend,
    resolve_data_yaml,
)
from scripts.revision.stage_corr import run_stage_corr  # noqa: E402
from scripts.revision.stage_eval import run_stage_eval  # noqa: E402
from scripts.revision.stage_illum import run_stage_illum  # noqa: E402
from scripts.revision.stage_info import run_stage_info  # noqa: E402
from scripts.revision.stage_report import run_stage_report  # noqa: E402
from scripts.revision.stage_stats import run_stage_stats  # noqa: E402
from scripts.revision.stage_train import run_stage_train  # noqa: E402


class TeeLogger:
    """Duplicate stdout/stderr to a log file."""

    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.terminal = sys.stdout
        self.log_file = open(log_path, "a", encoding="utf-8")

    def write(self, message):
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            enc = getattr(self.terminal, "encoding", "utf-8") or "utf-8"
            self.terminal.write(message.encode(enc, errors="replace").decode(enc))
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def parse_args():
    parser = argparse.ArgumentParser(description="PIERE-2026 Revision Master Runner")
    parser.add_argument(
        "--stage",
        choices=["info", "train", "eval", "corr", "illum", "stats", "report", "all"],
        default="all",
        help="Stage to execute (default: all)",
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA_YAML), help="Path to dataset YAML file")
    parser.add_argument(
        "--device",
        default="0" if os.environ.get("CUDA_VISIBLE_DEVICES") or os.name != "nt" else "0",
        help="Device to use ('0', '1', 'cpu')",
    )
    parser.add_argument(
        "--with-mosaic", action="store_true", help="Include priority 2 mosaic runs (close_mosaic=0, 50)"
    )
    parser.add_argument(
        "--with-ep40sgd",
        action="store_true",
        help="Include control series ep40-sgd (6 runs with optimizer=SGD, lr0=0.01; disabled by default)",
    )
    parser.add_argument(
        "--repro-check",
        action="store_true",
        help="Explicitly enable reproducibility check run e9p (also enabled by default via repro gate)",
    )
    parser.add_argument(
        "--skip-repro-gate",
        action="store_true",
        help="Skip reproducibility check gate (|Delta| <= 0.3 pp) before bulk training",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Display execution plan and time estimates without executing"
    )
    parser.add_argument("--smoke", action="store_true", help="Execute smoke verification on CPU with tiny fixture")
    parser.add_argument("--force", action="store_true", help="Force re-run even if outputs/checkpoints already exist")
    return parser.parse_args()


def ensure_smoke_fixture(fixture_dir: Path) -> Path:
    """Create minimal 6-image YOLO dataset fixture for CPU smoke testing."""
    fixture_dir.mkdir(parents=True, exist_ok=True)
    data_yaml_p = fixture_dir / "smoke_exdark.yaml"
    if data_yaml_p.is_file():
        return data_yaml_p

    import cv2
    import numpy as np

    for split in ["train", "val"]:
        img_dir = fixture_dir / "images" / split
        lbl_dir = fixture_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        n_imgs = 4 if split == "train" else 2
        for i in range(n_imgs):
            img_p = img_dir / f"img_{i}.jpg"
            lbl_p = lbl_dir / f"img_{i}.txt"
            if not img_p.is_file():
                img = (np.random.rand(640, 640, 3) * 255).astype(np.uint8)
                cv2.imwrite(str(img_p), img)
            if not lbl_p.is_file():
                with open(lbl_p, "w", encoding="utf-8") as f:
                    f.write("0 0.5 0.5 0.2 0.2\n")

    yaml_content = f"""path: {fixture_dir.as_posix()}
train: images/train
val: images/val
test: images/val

names:
  0: bicycle
  1: boat
  2: bottle
  3: bus
  4: car
  5: cat
  6: chair
  7: cup
  8: dog
  9: motorbike
  10: people
  11: table
"""
    with open(data_yaml_p, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    return data_yaml_p


def check_reproducibility_gate(
    data_yaml: Path,
    device: str,
    runs_dir: Path,
    smoke: bool = False,
    force: bool = False,
    dry_run: bool = False,
    results_dir: Optional[Path] = None,
) -> bool:
    """Execute e9p reproducibility check before bulk training.

    Compares newly trained e9p__repro-baseline-ep40__s0 against target 0.7159.
    Returns True if PASS (|Δ| <= 0.3 pp), False otherwise.
    """
    from scripts.revision.stage_eval import evaluate_checkpoint
    from scripts.revision.stage_train import execute_single_run

    target_mAP50 = float(REPRO_CHECK_RUN.get("target_mAP50", 0.7159))
    tolerance_pp = float(REPRO_CHECK_RUN.get("tolerance", 0.003)) * 100.0  # 0.30 pp

    print("=" * 80)
    print("  REPRODUCIBILITY GATE: VERIFYING BASELINE BEFORE BULK TRAINING (R1.2)")
    print("=" * 80)
    print("  Target Run        : e5p__baseline-ep40__s0 (historical 40-epoch baseline)")
    print(f"  Repro Run         : {REPRO_CHECK_RUN['run']}")
    print(f"  Expected mAP50    : {target_mAP50:.4f} ({target_mAP50 * 100:.2f}%)")
    print(f"  Allowed tolerance : +- {tolerance_pp:.2f} pp")
    print("=" * 80 + "\n")

    if dry_run:
        print("[DRY-RUN] Repro gate check planned: train e9p, evaluate on val, require |Delta| <= 0.3 pp.")
        return True

    # 1. Train e9p
    print("[REPRO GATE] Training reproducibility model e9p__repro-baseline-ep40__s0...")
    execute_single_run(
        exp_info=REPRO_CHECK_RUN,
        data_yaml=data_yaml,
        device=device,
        runs_dir=runs_dir,
        smoke=smoke,
        force=force,
        results_dir=results_dir,
    )

    ckpt_candidates = [
        runs_dir / REPRO_CHECK_RUN["run"] / "weights" / "best.pt",
        REPO_ROOT / "runs" / "detect" / "runs" / REPRO_CHECK_RUN["run"] / "weights" / "best.pt",
        REPO_ROOT / "runs" / REPRO_CHECK_RUN["run"] / "weights" / "best.pt",
    ]
    ckpt = next((c for c in ckpt_candidates if c.is_file()), None)
    if not ckpt:
        print(f"[REPRO GATE ERROR] Checkpoint for {REPRO_CHECK_RUN['run']} not found at {ckpt_candidates[0]}")
        return False

    # 2. Evaluate on split=val
    print(f"\n[REPRO GATE] Evaluating {ckpt} on validation split...")
    try:
        val_stats = evaluate_checkpoint(
            ckpt_path=ckpt,
            data_yaml=data_yaml,
            split="val",
            device=device,
            batch=4 if smoke else 16,
        )
        observed_mAP50 = float(val_stats.get("mAP50", 0.0))
    except Exception as e:
        print(f"[REPRO GATE ERROR] Validation failed: {e}")
        return False

    diff_pp = (observed_mAP50 - target_mAP50) * 100.0
    abs_diff = abs(diff_pp)
    verdict = "PASS" if abs_diff <= tolerance_pp else "FAIL"

    print("\n" + "=" * 80)
    print(f"  REPRODUCIBILITY GATE VERDICT: [{verdict}]")
    print("=" * 80)
    print("  Target Run   : e5p__baseline-ep40__s0")
    print(f"  Repro Run    : {REPRO_CHECK_RUN['run']}")
    print(f"  Target mAP50 : {target_mAP50:.4f} ({target_mAP50 * 100:.2f}%)")
    print(f"  Observed mAP : {observed_mAP50:.4f} ({observed_mAP50 * 100:.2f}%)")
    print(f"  Difference   : {diff_pp:+.2f} pp (tolerance: +-{tolerance_pp:.2f} pp)")
    print(f"  Verdict      : {verdict}")
    print("=" * 80 + "\n")

    if verdict != "PASS":
        if smoke:
            print("[REPRO GATE] In smoke mode (tiny fixture), discrepancy is expected. Proceeding...")
            return True
        return False

    return True


def main():
    args = parse_args()

    # Initialize NMS backend (validates torchvision in production mode, enables pure-torch fallback in smoke mode)
    if not args.dry_run or args.smoke:
        init_nms_backend(smoke=args.smoke)

    # Determine results directory
    results_dir = DEFAULT_SMOKE_DIR if args.smoke else DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Resolve device for smoke test
    device = "cpu" if args.smoke else args.device

    # Resolve dataset yaml
    if args.smoke:
        fixture_dir = results_dir / "fixture"
        data_yaml = ensure_smoke_fixture(fixture_dir)
    else:
        try:
            data_yaml = resolve_data_yaml(args.data)
        except FileNotFoundError:
            data_yaml = DEFAULT_DATA_YAML

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_log = logs_dir / f"master_{args.stage}_{timestamp}.log"
    tee = TeeLogger(master_log)
    sys.stdout = tee
    sys.stderr = tee

    start_all = time.time()

    print("=" * 80)
    print("  PIERE-2026 REVISION MASTER RUNNER")
    print("=" * 80)
    print(f"Timestamp    : {datetime.now().isoformat()}")
    print(f"Stage        : {args.stage}")
    print(f"Device       : {device}")
    print(f"Dataset      : {data_yaml}")
    print(f"Dry Run      : {args.dry_run}")
    print(f"Smoke Mode   : {args.smoke}")
    print(f"Force Re-run : {args.force}")
    print(f"With Mosaic  : {args.with_mosaic}")
    print(f"With EP40SGD : {args.with_ep40sgd}")
    print(f"Skip Repro   : {args.skip_repro_gate}")
    print(f"Results Dir  : {results_dir}")
    print(f"Log File     : {master_log}")
    print("=" * 80 + "\n")

    stages_to_run = (
        ["info", "train", "eval", "corr", "illum", "stats", "report"]
        if args.stage == "all"
        else [s.strip() for s in args.stage.split(",") if s.strip()]
    )

    for st in stages_to_run:
        stage_start = time.time()
        print(f"\n>>> ENTERING STAGE: {st.upper()} [{datetime.now().strftime('%H:%M:%S')}]")

        if st == "info":
            run_stage_info(results_dir=results_dir, data_yaml=data_yaml, dry_run=args.dry_run)

        elif st == "train":
            runs_dir = results_dir / "runs" if args.smoke else REPO_ROOT / "runs" / "detect" / "runs"

            if not args.skip_repro_gate:
                gate_passed = check_reproducibility_gate(
                    data_yaml=data_yaml,
                    device=device,
                    runs_dir=runs_dir,
                    smoke=args.smoke,
                    force=args.force,
                    dry_run=args.dry_run,
                    results_dir=results_dir,
                )
                if not gate_passed:
                    print("\n" + "!" * 80)
                    print("  [FATAL] REPRODUCIBILITY GATE FAILED (|Delta| > 0.3 pp)!")
                    print("  Comparing e9p__repro-baseline-ep40__s0 vs historical e5p__baseline-ep40__s0.")
                    print("  Bulk training has been HALTED to prevent wasting GPU compute on an unverified setup.")
                    print("  What to compare:")
                    print("    1. Ultralytics version & PyTorch / CUDA environment in environment.json")
                    print("    2. ExDark dataset splits integrity (labels/val and images/val)")
                    print("    3. Optimizer settings (ensure optimizer='auto', no manual lr0 drift)")
                    print("  To bypass this gate and continue bulk training anyway, run with:")
                    print("    python scripts/revision/run_all.py --stage all --skip-repro-gate")
                    print("!" * 80 + "\n")
                    sys.exit(1)

            run_stage_train(
                data_yaml=data_yaml,
                device=device,
                with_mosaic=args.with_mosaic,
                with_ep40sgd=args.with_ep40sgd,
                repro_check=False,
                dry_run=args.dry_run,
                smoke=args.smoke,
                force=args.force,
                runs_dir=runs_dir,
                results_dir=results_dir,
            )

        elif st == "eval":
            run_stage_eval(
                results_dir=results_dir,
                data_yaml=data_yaml,
                device=device,
                with_mosaic=args.with_mosaic,
                with_ep40sgd=args.with_ep40sgd,
                repro_check=True,
                dry_run=args.dry_run,
                smoke=args.smoke,
                force=args.force,
            )

        elif st == "corr":
            run_stage_corr(
                results_dir=results_dir,
                data_yaml=data_yaml,
                device=device,
                dry_run=args.dry_run,
                smoke=args.smoke,
                force=args.force,
            )

        elif st == "illum":
            run_stage_illum(
                results_dir=results_dir,
                data_yaml=data_yaml,
                device=device,
                dry_run=args.dry_run,
                smoke=args.smoke,
                force=args.force,
            )

        elif st == "stats":
            run_stage_stats(results_dir=results_dir)

        elif st == "report":
            run_stage_report(results_dir=results_dir)

        stage_elapsed = round((time.time() - stage_start) / 60.0, 2)
        print(f"<<< COMPLETED STAGE: {st.upper()} in {stage_elapsed} minutes.\n")

    total_elapsed = round((time.time() - start_all) / 60.0, 2)
    print("=" * 80)
    print(f"  ALL REQUESTED STAGES COMPLETED SUCCESSFULLY IN {total_elapsed} MINUTES")
    print(f"  Summary Report : {results_dir / 'SUMMARY.md'}")
    print(f"  Zip Archive    : {results_dir / 'revision_results.zip'}")
    print("=" * 80)

    orig_stdout = tee.terminal
    sys.stdout = orig_stdout
    sys.stderr = orig_stdout
    tee.close()


if __name__ == "__main__":
    main()
