# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Training curves plotting and metrics extraction script for RI-YOLO Phase 0.

Plots specified columns from results.csv across multiple training runs,
saves figure, exports concatenated dataframe, and prints max/last metric checkpoints.
"""

import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training curves and compare runs.")
    parser.add_argument("--runs", nargs="+", required=True, help="List of run directories to compare")
    parser.add_argument(
        "--cols",
        nargs="+",
        default=["metrics/mAP50(B)", "metrics/mAP50-95(B)", "train/box_loss", "val/box_loss"],
        help="List of columns to plot",
    )
    parser.add_argument("--out", default="artifacts/phase0/curves_comparison.png", help="Output PNG file path")
    parser.add_argument("--csv", default="artifacts/phase0/curves_comparison.csv", help="Output aggregated CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)

    n_cols = len(args.cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.5))
    if n_cols == 1:
        axes = [axes]

    frames = []
    for r in args.runs:
        csv_path = os.path.join(r, "results.csv")
        if not os.path.exists(csv_path):
            print(f"[WARNING] Skipping {r}: {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        run_name = os.path.basename(os.path.normpath(r))
        df["run"] = run_name
        frames.append(df)

        for ax, c in zip(axes, args.cols):
            if c in df.columns:
                ax.plot(df["epoch"], df[c], label=run_name)
            else:
                print(f"[WARNING] Column '{c}' not found in {r}")

    for ax, c in zip(axes, args.cols):
        ax.set_title(c, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"[INFO] Comparison plot saved to {args.out}")

    if frames and args.csv:
        combined_df = pd.concat(frames, ignore_index=True)
        combined_df.to_csv(args.csv, index=False)
        print(f"[INFO] Combined metrics exported to {args.csv}")

    print("\n--- Summary Checkpoints ---")
    m_col = "metrics/mAP50(B)"
    for r in args.runs:
        csv_path = os.path.join(r, "results.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        if m_col in df.columns:
            max_val = df[m_col].max()
            best_epoch = int(df[m_col].idxmax()) + 1
            last_val = df[m_col].iloc[-1]
            print(f"Run: {r:35s} | max mAP50 = {max_val:.4f} (@epoch {best_epoch:3d}) | last mAP50 = {last_val:.4f}")


if __name__ == "__main__":
    main()
