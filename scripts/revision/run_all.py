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

# Ensure ultralytics is in sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.revision.config import (  # noqa: E402
    DEFAULT_DATA_YAML,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SMOKE_DIR,
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
        self.terminal.write(message)
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


def main():
    args = parse_args()

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
    print(f"Results Dir  : {results_dir}")
    print(f"Log File     : {master_log}")
    print("=" * 80 + "\n")

    stages_to_run = (
        ["info", "train", "eval", "corr", "illum", "stats", "report"] if args.stage == "all" else [args.stage]
    )

    for st in stages_to_run:
        stage_start = time.time()
        print(f"\n>>> ENTERING STAGE: {st.upper()} [{datetime.now().strftime('%H:%M:%S')}]")

        if st == "info":
            run_stage_info(results_dir=results_dir, data_yaml=data_yaml, dry_run=args.dry_run)

        elif st == "train":
            run_stage_train(
                data_yaml=data_yaml,
                device=device,
                with_mosaic=args.with_mosaic,
                dry_run=args.dry_run,
                smoke=args.smoke,
                force=args.force,
            )

        elif st == "eval":
            run_stage_eval(
                results_dir=results_dir,
                data_yaml=data_yaml,
                device=device,
                with_mosaic=args.with_mosaic,
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
