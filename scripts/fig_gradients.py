# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Analytical localization gradient curves for IoU and NWD losses."""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot analytical localization gradient curves.")
    parser.add_argument("--C", type=float, default=12.8, help="Standard baseline NWD constant C")
    parser.add_argument("--C-calib", type=float, default=10.24, help="Calibrated ExDark NWD constant C")
    parser.add_argument("--out", type=str, default="figs/C2_fig1_gradients.png", help="Output PNG path")
    return parser.parse_args()


def main():
    args = parse_args()
    A = np.linspace(0.5, 40, 400)
    d = 0.1 * A
    g_iou = 2 * A / ((A + d) ** 2)

    plt.figure(figsize=(6, 4))
    plt.plot(A, g_iou, label="|d(IoU)/dδ|", color="black", linewidth=1.8)
    for C, st, col, lab in [
        (args.C, "--", "blue", f"|d(NWD)/dδ|, C={args.C:.1f} (baseline)"),
        (args.C_calib, "-.", "red", f"|d(NWD)/dδ|, C={args.C_calib:.2f} (calibrated)"),
    ]:
        plt.plot(A, (1 / C) * np.exp(-d / C), st, color=col, label=lab, linewidth=1.6)

    plt.xlabel("Object size, grid units", fontsize=11)
    plt.ylabel("Localization gradient magnitude", fontsize=11)
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3, linestyle=":")
    plt.legend(fontsize=9)
    plt.tight_layout()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.out, dpi=300)
    print(f"[SUCCESS] Saved gradient figure to {args.out}")


if __name__ == "__main__":
    main()
