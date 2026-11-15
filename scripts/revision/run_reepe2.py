# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# -*- coding: utf-8 -*-
"""REEPE 2027 Unified Overnight Runner (Stage A + Stage B).

Single-entry overnight execution:
    1. Quick pre-flight check (2-3 min):
        python -m scripts.revision.run_reepe2 --device 0 --check

    2. Full overnight execution (Stage A Latency + Stage B Seeds training & eval):
        python -m scripts.revision.run_reepe2 --device 0

All raw results are written to results/reepe2/ and packaged into results/reepe2_bundle.zip.
Execution is fully resumable: if any step or run is already completed, it is skipped automatically.
"""

import argparse
from pathlib import Path
import time

from scripts.revision.config import DEFAULT_DATA_YAML, REPO_ROOT, resolve_data_yaml
from scripts.revision.reepe_latency import pack_bundle, run_benchmark, update_notes
from scripts.revision.reepe_seeds import run_smoke_check, run_stage_seeds

OUT = REPO_ROOT / "results" / "reepe2"


def log(msg: str):
    """Log message to console and results/reepe2/run_all.log."""
    OUT.mkdir(parents=True, exist_ok=True)
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(OUT / "run_all.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_pipeline(
    data_yaml: Path,
    device: str = "0",
    skip_latency: bool = False,
    skip_seeds: bool = False,
    rounds: int = 10,
    iters: int = 300,
):
    """Run full overnight pipeline sequentially: Stage A then Stage B."""
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    update_notes(f"Started unified overnight pipeline on device={device}")

    log("=" * 70)
    log("=== REEPE 2027 UNIFIED OVERNIGHT PIPELINE ===")
    log(f"Device: {device} | Data: {data_yaml}")
    log("=" * 70)

    # ---------------------------------------------------------
    # STAGE A: Pure Latency Benchmark (~40-45 min)
    # ---------------------------------------------------------
    if not skip_latency:
        log("\n>>> [1/2] STAGE A: Pure Latency Benchmark (10 rounds, randomized order) <<<")
        run_benchmark(device_str=device, num_rounds=rounds, warmup_iters=50, bench_iters=iters)
        log(">>> [1/2] STAGE A: Latency Benchmark Completed! <<<\n")
    else:
        log("\n>>> [1/2] STAGE A: Skipped (--skip-latency) <<<")

    # ---------------------------------------------------------
    # STAGE B: Additional Seeds Training & Evaluation (~6-7 hrs)
    # ---------------------------------------------------------
    if not skip_seeds:
        log("\n>>> [2/2] STAGE B: Additional Seeds Training & Evaluation (8 runs) <<<")
        run_stage_seeds(data_yaml=data_yaml, device=device)
        log(">>> [2/2] STAGE B: Additional Seeds Completed! <<<\n")
    else:
        log("\n>>> [2/2] STAGE B: Skipped (--skip-seeds) <<<")

    # ---------------------------------------------------------
    # Final Bundle & Notes
    # ---------------------------------------------------------
    elapsed_total_h = (time.time() - t_start) / 3600.0
    update_notes(f"Unified overnight pipeline completed successfully in {elapsed_total_h:.2f} hours.")
    pack_bundle()

    log("=" * 70)
    log(f"=== ALL EXPERIMENTS COMPLETED IN {elapsed_total_h:.2f} HOURS ===")
    log(f"Raw results directory : {OUT}")
    log(f"Final archive package : {REPO_ROOT / 'results' / 'reepe2_bundle.zip'}")
    log("=" * 70)


def main():
    """CLI entrypoint for unified overnight pipeline."""
    ap = argparse.ArgumentParser(description="REEPE 2027 Unified Overnight Runner")
    ap.add_argument("--data", default=str(DEFAULT_DATA_YAML), help="Path to exdark.yaml")
    ap.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    ap.add_argument("--check", action="store_true", help="Quick 2-min sanity check of both stages")
    ap.add_argument("--skip-latency", action="store_true", help="Skip Stage A latency benchmark")
    ap.add_argument("--skip-seeds", action="store_true", help="Skip Stage B training & evaluation")
    ap.add_argument("--rounds", type=int, default=10, help="Stage A rounds (default: 10)")
    ap.add_argument("--iters", type=int, default=300, help="Stage A iterations per block (default: 300)")
    args = ap.parse_args()

    data_yaml = resolve_data_yaml(args.data)

    if args.check:
        log("=================================================================")
        log("=== RUNNING PRE-FLIGHT CHECK (Stage A & Stage B Smoke Tests) ===")
        log("=================================================================")
        log("\n[CHECK 1/2] Testing Stage A (Latency benchmark: 1 round, 20 iters)...")
        run_benchmark(device_str=args.device, num_rounds=1, warmup_iters=5, bench_iters=20)
        log("[CHECK 1/2 PASS] Stage A latency check passed.")

        log("\n[CHECK 2/2] Testing Stage B (Smoke training & eval: 1 epoch, fraction=0.02)...")
        run_smoke_check(data_yaml=data_yaml, device=args.device)
        log("[CHECK 2/2 PASS] Stage B smoke check passed.")

        log("\n" + "=" * 65)
        log("[SUCCESS] ALL PRE-FLIGHT CHECKS PASSED!")
        log("You can now safely run the full overnight job with:")
        log(f"    python -m scripts.revision.run_reepe2 --device {args.device}")
        log("=" * 65)
        return

    run_pipeline(
        data_yaml=data_yaml,
        device=args.device,
        skip_latency=args.skip_latency,
        skip_seeds=args.skip_seeds,
        rounds=args.rounds,
        iters=args.iters,
    )


if __name__ == "__main__":
    main()
