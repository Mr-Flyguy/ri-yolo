# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# -*- coding: utf-8 -*-
"""REEPE 2027 pure latency benchmark (Stage A).

Measures latency across all architectures in ONE unified session, in randomized order over 10 rounds:
    python -m scripts.revision.reepe_latency --check   # 1 round, 20 iters sanity check
    python -m scripts.revision.reepe_latency           # full benchmark: 10 rounds x 300 iters

Outputs: results/reepe2/{latency_raw.csv, env_latency.txt, latency.log, NOTES.md}
"""
import argparse
import copy
import csv
import os
from pathlib import Path
import platform
import random
import subprocess
import time
import zipfile

import numpy as np
import torch

from scripts.revision.config import REPO_ROOT
from scripts.revision.reepe_overnight import ckpt_of, done_keys

OUT = REPO_ROOT / "results" / "reepe2"

MODEL_SPECS = [
    ("baseline", "e2p__baseline__s0", 0),
    ("p3", "e3p__pos-p3__s0", 128),
    ("p4", "e3p__pos-p4__s0", 256),
    ("presppf", "e3p__pos-presppf__s0", 512),
    ("postsppf", "e2p__postsppf__s0", 512),
    ("capctrl", "e3p__capctrl__s0", None),
]


def log(msg: str):
    """Log message to console and results/reepe2/latency.log."""
    OUT.mkdir(parents=True, exist_ok=True)
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(OUT / "latency.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def append_row(path: Path, row: dict):
    """Append a dictionary row to CSV, writing header if file is new."""
    new = not path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def get_gpu_telemetry():
    """Query GPU temperature (C) and SM clock (MHz) via nvidia-smi."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.current.sm", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            if len(parts) >= 2:
                return parts[0], parts[1]
    except Exception:
        pass
    return "NA", "NA"


def capture_env_latency(stage_label="START"):
    """Capture system, GPU, driver, and performance telemetry to env_latency.txt."""
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"=== GPU & SYSTEM ENVIRONMENT ({stage_label}) ===")
    lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"OS: {platform.platform()} ({platform.system()} {platform.release()})")
    lines.append(f"Python: {platform.python_version()}")
    lines.append(f"PyTorch: {torch.__version__}")
    try:
        import ultralytics

        lines.append(f"Ultralytics: {ultralytics.__version__}")
    except Exception as e:
        lines.append(f"Ultralytics: error ({e})")
    lines.append(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        lines.append(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        lines.append(f"CUDA Version (torch): {torch.version.cuda}")
        lines.append(f"Device Count: {torch.cuda.device_count()}")
    lines.append(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    if platform.system() == "Windows":
        try:
            pwr = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=3)
            if pwr.returncode == 0:
                lines.append(f"Windows Power Scheme: {pwr.stdout.strip()}")
        except Exception:
            pass

    try:
        drv = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if drv.returncode == 0:
            lines.append(f"NVIDIA Driver: {drv.stdout.strip()}")

        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if apps.returncode == 0:
            lines.append(f"Active Compute Apps:\n{apps.stdout.strip()}")

        clocks = subprocess.run(
            ["nvidia-smi", "-q", "-d", "CLOCK,PERFORMANCE"], capture_output=True, text=True, timeout=5
        )
        if clocks.returncode == 0:
            lines.append(f"NVIDIA Clock & Performance Query:\n{clocks.stdout.strip()}")
    except Exception as e:
        lines.append(f"nvidia-smi query failed/unavailable: {e}")

    lines.append("=" * 60 + "\n")
    env_file = OUT / "env_latency.txt"
    mode = "a" if env_file.exists() else "w"
    with open(env_file, mode, encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def pack_bundle():
    """Package results/reepe2/ contents into results/reepe2_bundle.zip."""
    bundle_path = REPO_ROOT / "results" / "reepe2_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(OUT):
            for file in files:
                full_p = Path(root) / file
                rel_p = full_p.relative_to(REPO_ROOT)
                zipf.write(full_p, arcname=str(rel_p))
    log(f"[PACK] Created archive {bundle_path} ({bundle_path.stat().st_size / 1024:.1f} KB)")


def update_notes(entry: str):
    """Append factual log entry to results/reepe2/NOTES.md."""
    OUT.mkdir(parents=True, exist_ok=True)
    notes_file = OUT / "NOTES.md"
    mode = "a" if notes_file.exists() else "w"
    with open(notes_file, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write("# REEPE 2027 REVISION EXPERIMENT NOTES\n\n")
        f.write(f"- **{time.strftime('%Y-%m-%d %H:%M:%S')}**: {entry}\n")


def run_benchmark(device_str="0", num_rounds=10, warmup_iters=50, bench_iters=300):
    """Execute clean latency benchmark across models, graph fusion, batches, and precisions."""
    torch.backends.cudnn.benchmark = False
    device = torch.device(f"cuda:{device_str}" if torch.cuda.is_available() and device_str.isdigit() else device_str)

    log(f"=== Stage A: Clean Latency Benchmark (device={device}, rounds={num_rounds}, iters={bench_iters}) ===")
    capture_env_latency("START")

    # 1. Resolve and load checkpoints
    from ultralytics import YOLO

    loaded_unfused = {}
    loaded_fused = {}
    param_counts = {}
    run_names = {}

    log("Resolving checkpoints and initializing models...")
    for m_name, run_name, c_dim in MODEL_SPECS:
        ckpt = ckpt_of(run_name)
        if ckpt is None or not ckpt.is_file():
            if m_name == "capctrl":
                log(f"[WARN] Checkpoint {run_name} for capctrl not found, skipping capctrl.")
                continue
            raise FileNotFoundError(f"Required checkpoint for {m_name} ({run_name}) not found!")

        run_names[m_name] = run_name
        model_obj = YOLO(str(ckpt)).model.eval().to(device)
        for p in model_obj.parameters():
            p.requires_grad = False

        n_p = sum(p.numel() for p in model_obj.parameters())
        param_counts[m_name] = n_p
        loaded_unfused[m_name] = model_obj

        # Fused copy
        fused_copy = copy.deepcopy(model_obj).fuse(verbose=False)
        for p in fused_copy.parameters():
            p.requires_grad = False
        loaded_fused[m_name] = fused_copy

        log(f"Loaded {m_name:<10} ({run_name}) | params = {n_p:,}")

    # 2. Sanity check: parameter count differences
    base_params = param_counts["baseline"]
    log(f"Baseline ({run_names['baseline']}) parameters: {base_params:,}")
    expected_diffs = {
        "p3": 10 * (128**2) + 5 * 128 + 1,  # 164481
        "p4": 10 * (256**2) + 5 * 256 + 1,  # 656641
        "presppf": 10 * (512**2) + 5 * 512 + 1,  # 2624001
        "postsppf": 10 * (512**2) + 5 * 512 + 1,  # 2624001
    }
    for m_name, exp_d in expected_diffs.items():
        if m_name in param_counts:
            act_d = param_counts[m_name] - base_params
            log(f"[SANITY] {m_name:<10}: act_diff = {act_d:,}, exp_diff = {exp_d:,}")
            assert (
                act_d == exp_d
            ), f"Parameter count difference mismatch for {m_name}: got {act_d}, expected {exp_d} (10C^2+5C+1)"

    if "capctrl" in param_counts:
        act_d_cap = param_counts["capctrl"] - base_params
        exp_d_cap = 10 * (512**2) + 4 * 512 + 1  # 2623489
        log(f"[SANITY] capctrl   : act_diff = {act_d_cap:,} (reference 10C^2+4C+1 = {exp_d_cap:,})")

    # 3. Fixed input tensors (seed 0)
    torch.manual_seed(0)
    inputs_fp32 = {
        1: torch.rand(1, 3, 640, 640, device=device, dtype=torch.float32),
        16: torch.rand(16, 3, 640, 640, device=device, dtype=torch.float32),
    }
    inputs_fp16 = {
        1: inputs_fp32[1].half(),
        16: inputs_fp32[16].half(),
    }

    # 4. Multi-round measurement loop
    csv_path = OUT / "latency_raw.csv"
    done = done_keys(csv_path, ["round", "model", "fused", "batch", "precision"])
    models_available = list(loaded_unfused.keys())

    t_start = time.time()
    for r in range(num_rounds):
        # Shuffled model order per round
        rng = random.Random(r)
        shuffled_order = list(models_available)
        rng.shuffle(shuffled_order)
        log(f"--- Round {r + 1}/{num_rounds} | Shuffled order: {shuffled_order} ---")

        for order_idx, m_name in enumerate(shuffled_order):
            gpu_temp, sm_clock = get_gpu_telemetry()
            run_name = run_names[m_name]
            n_p = param_counts[m_name]

            for fused in (0, 1):
                base_m = loaded_fused[m_name] if fused == 1 else loaded_unfused[m_name]

                for bs in (1, 16):
                    for prec in ("fp32", "fp16"):
                        key = (str(r), m_name, str(fused), str(bs), prec)
                        if key in done:
                            continue

                        # Setup model precision and input
                        if prec == "fp16":
                            active_m = base_m.half()
                            x = inputs_fp16[bs]
                        else:
                            active_m = base_m.float()
                            x = inputs_fp32[bs]

                        # Warmup
                        with torch.no_grad():
                            for _ in range(warmup_iters):
                                active_m(x)
                            if device.type == "cuda":
                                torch.cuda.synchronize()

                            # Timed passes
                            timings = []
                            for _ in range(bench_iters):
                                if device.type == "cuda":
                                    ev_start = torch.cuda.Event(enable_timing=True)
                                    ev_end = torch.cuda.Event(enable_timing=True)
                                    ev_start.record()
                                    active_m(x)
                                    ev_end.record()
                                    torch.cuda.synchronize()
                                    timings.append(ev_start.elapsed_time(ev_end))
                                else:
                                    t0 = time.perf_counter()
                                    active_m(x)
                                    timings.append((time.perf_counter() - t0) * 1000.0)

                        t_arr = np.array(timings)
                        med_ms = round(float(np.median(t_arr)), 3)
                        p95_ms = round(float(np.percentile(t_arr, 95)), 3)
                        mean_ms = round(float(np.mean(t_arr)), 3)
                        min_ms = round(float(np.min(t_arr)), 3)

                        row = {
                            "round": r,
                            "order_index": order_idx,
                            "model": m_name,
                            "run": run_name,
                            "fused": fused,
                            "batch": bs,
                            "precision": prec,
                            "median_ms": med_ms,
                            "p95_ms": p95_ms,
                            "mean_ms": mean_ms,
                            "min_ms": min_ms,
                            "n_params": n_p,
                            "gpu_temp_c": gpu_temp,
                            "sm_clock_mhz": sm_clock,
                        }
                        append_row(csv_path, row)
                        log(
                            f"[R{r + 1:02d}|ord{order_idx}] {m_name:<9} fused={fused} bs={bs:<2} prec={prec:<4} "
                            f"| med={med_ms:.2f}ms p95={p95_ms:.2f}ms min={min_ms:.2f}ms | temp={gpu_temp}C sm={sm_clock}MHz"
                        )

    capture_env_latency("END")
    elapsed_total_min = (time.time() - t_start) / 60.0
    update_notes(f"reepe_latency benchmark completed ({num_rounds} rounds) in {elapsed_total_min:.1f} min.")
    pack_bundle()
    log(f"=== Latency benchmark complete ({elapsed_total_min:.1f} min total) ===")


def main():
    ap = argparse.ArgumentParser(description="REEPE 2027 Pure Latency Benchmark")
    ap.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    ap.add_argument("--check", action="store_true", help="Quick check: 1 round, 5 warmup, 20 iters")
    ap.add_argument("--rounds", type=int, default=10, help="Number of benchmark rounds (default: 10)")
    ap.add_argument("--iters", type=int, default=300, help="Iterations per block (default: 300)")
    args = ap.parse_args()

    if args.check:
        log("Running reepe_latency check (1 round, 20 iterations)...")
        run_benchmark(device_str=args.device, num_rounds=1, warmup_iters=5, bench_iters=20)
        log("[PASS] reepe_latency --check completed successfully!")
        return

    run_benchmark(device_str=args.device, num_rounds=args.rounds, warmup_iters=50, bench_iters=args.iters)


if __name__ == "__main__":
    main()
