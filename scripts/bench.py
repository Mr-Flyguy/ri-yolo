# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Hardware inference latency, FPS, and computational complexity benchmark.

Measures parameter count, GFLOPs, median/p95 latency (ms), and FPS
across batch sizes and precisions (FP32, FP16/AMP). Generates Table 2 for Paper C3.
"""

import argparse
import csv
import os
import time

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark YOLO latency, FPS, parameters and GFLOPs.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "yolov8s.yaml",
            "yolov8s-rfd-p3.yaml",
            "yolov8s-rfd-p4.yaml",
            "yolov8s-rfd-presppf.yaml",
            "yolov8s-rfd-postsppf.yaml",
            "yolov8s-capctrl.yaml",
        ],
        help="List of model config files or checkpoint paths",
    )
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 16], help="Batch sizes to evaluate")
    parser.add_argument("--half", nargs="+", type=int, default=[0, 1], help="Precision modes: 0=FP32, 1=FP16")
    parser.add_argument("--warmup", type=int, default=50, help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=300, help="Number of benchmark iterations")
    parser.add_argument("--imgsz", type=int, default=640, help="Image spatial resolution (default: 640)")
    parser.add_argument("--out", default="artifacts/phase0/bench.csv", help="Output CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[WARNING] CUDA is not available! Benchmark will run on CPU and may not reflect GPU latency/FPS.")

    rows = []
    print("\n" + "=" * 95, flush=True)
    print(
        f"{'Model':<28} | {'Params (M)':<10} | {'GFLOPs':<8} | {'BS':<4} | {'FP16':<5} | {'Median (ms)':<11} | {'p95 (ms)':<9} | {'FPS':<7}",
        flush=True,
    )
    print("=" * 95, flush=True)

    for cfg in args.models:
        try:
            base_model = YOLO(cfg).model
            npar = sum(p.numel() for p in base_model.parameters())
            gflops = get_flops(base_model, imgsz=args.imgsz)
        except Exception as e:
            print(f"[ERROR] Could not initialize model {cfg}: {e}")
            continue

        for bs in args.batches:
            for hf in args.half:
                try:
                    m = YOLO(cfg).model.eval().to(device)
                    if hf and device == "cuda":
                        m = m.half()

                    dtype = torch.half if (hf and device == "cuda") else torch.float
                    x = torch.randn(bs, 3, args.imgsz, args.imgsz, device=device, dtype=dtype)

                    # Warmup
                    with torch.no_grad():
                        for _ in range(args.warmup):
                            m(x)
                        if device == "cuda":
                            torch.cuda.synchronize()

                        timings = []
                        for _ in range(args.iters):
                            if device == "cuda":
                                torch.cuda.synchronize()
                            t0 = time.perf_counter()

                            m(x)

                            if device == "cuda":
                                torch.cuda.synchronize()
                            timings.append(time.perf_counter() - t0)

                    t_arr = np.array(timings) * 1000.0  # convert to ms
                    lat_med = float(np.median(t_arr))
                    lat_p95 = float(np.percentile(t_arr, 95))
                    fps = float(1000.0 * bs / lat_med)

                    row = {
                        "model": os.path.basename(cfg),
                        "params_M": round(npar / 1e6, 3),
                        "gflops": round(gflops, 2),
                        "batch": bs,
                        "half": hf,
                        "lat_med_ms": round(lat_med, 2),
                        "lat_p95_ms": round(lat_p95, 2),
                        "fps": round(fps, 1),
                    }
                    rows.append(row)
                    print(
                        f"{row['model']:<28} | {row['params_M']:<10.3f} | {row['gflops']:<8.2f} | "
                        f"{bs:<4d} | {hf:<5d} | {lat_med:<11.2f} | {lat_p95:<9.2f} | {fps:<7.1f}",
                        flush=True,
                    )
                    del m
                    if device == "cuda":
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"[ERROR] Benchmark failed for {cfg} (batch={bs}, half={hf}): {e}")

    if rows:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("=" * 95)
        print(f"[INFO] Benchmark results written to {args.out}\n")


if __name__ == "__main__":
    main()
