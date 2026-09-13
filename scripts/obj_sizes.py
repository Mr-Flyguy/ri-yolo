# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Object size distribution analysis and NWD constant C calibration script.

Analyzes bounding box area distributions across the ExDark dataset,
computes scale proportions (small/medium/large), and derives dataset-grounded
calibration values for NWD normalization constant C and size-gating parameter tau.
"""

import argparse
import csv
import glob
import os
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze bounding box sizes and calibrate NWD constants.")
    parser.add_argument("--labels", required=True, help="Directory containing YOLO format label text files")
    parser.add_argument("--imgsz", type=int, default=640, help="Standard input image resolution (default: 640)")
    parser.add_argument("--out", default="artifacts/phase0/obj_sizes.csv", help="Output CSV path for raw box sizes")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    label_files = glob.glob(os.path.join(args.labels, "**", "*.txt"), recursive=True)
    if not label_files:
        label_files = glob.glob(os.path.join(args.labels, "*.txt"))

    if not label_files:
        raise FileNotFoundError(f"No .txt label files found in: {args.labels}")

    sizes = []
    total_boxes = 0
    for f in label_files:
        try:
            with open(f) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        w = float(parts[3])
                        h = float(parts[4])
                        # Equivalent box dimension in pixels
                        side = np.sqrt(max(w * h, 1e-8)) * args.imgsz
                        sizes.append(side)
                        total_boxes += 1
        except Exception as e:
            print(f"[WARNING] Could not read {f}: {e}")

    if not sizes:
        raise RuntimeError("No valid bounding boxes parsed from label files!")

    s = np.array(sizes)
    mean_sz = float(s.mean())
    median_sz = float(np.median(s))
    p10, p50, p90 = np.percentile(s, [10, 50, 90])

    pct_small = float(100.0 * (s < 32).mean())
    pct_medium = float(100.0 * ((s >= 32) & (s < 96)).mean())
    pct_large = float(100.0 * (s >= 96).mean())

    c_p3 = mean_sz / 8.0
    c_p4 = mean_sz / 16.0
    c_p5 = mean_sz / 32.0
    rec_tau = c_p4 / 2.0

    print("\n" + "=" * 60)
    print(f"  EXDARK BOUNDING BOX CALIBRATION SUMMARY (imgsz={args.imgsz})")
    print("=" * 60)
    print(f"Total objects parsed   : {len(s):,}")
    print(f"Mean dimension         : {mean_sz:.2f} px")
    print(f"Median dimension       : {median_sz:.2f} px")
    print(f"Percentiles [10/50/90] : [{p10:.1f} / {p50:.1f} / {p90:.1f}] px")
    print("-" * 60)
    print(f"Scale Categories (COCO definition):")
    print(f"  Small  (< 32 px)     : {pct_small:.1f}%")
    print(f"  Medium (32 - 96 px)  : {pct_medium:.1f}%")
    print(f"  Large  (>= 96 px)    : {pct_large:.1f}%")
    print("-" * 60)
    print(f"Feature-stride normalized mean box scale:")
    print(f"  P3 (stride  8) : {c_p3:.2f} grid units (C_P3 = {c_p3:.2f})")
    print(f"  P4 (stride 16) : {c_p4:.2f} grid units (C_P4 = {c_p4:.2f})")
    print(f"  P5 (stride 32) : {c_p5:.2f} grid units (C_P5 = {c_p5:.2f})")
    print("-" * 60)
    print(f">>> RECOMMENDED nwd_c (P4-anchored) = {c_p4:.2f}   (YOLO baseline default: 12.8)")
    print(f">>> RECOMMENDED size_tau (sizegate) = {rec_tau:.2f}")
    print("=" * 60 + "\n")

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["size_px"])
        w.writerows([[round(x, 3)] for x in s])
    print(f"[INFO] Raw sizes saved to {args.out}")


if __name__ == "__main__":
    main()
