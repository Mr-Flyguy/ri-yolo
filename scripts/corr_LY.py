# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Correlation analysis between Retinex illumination map L and physical luminance Y.

Evaluates whether the learned illumination map L physically correlates with image brightness
(justifying Retinex terminology) or acts as a spatial attention/saliency mask.
"""

import argparse
import csv
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Compute correlation between illumination map L and luminance Y.")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--images", required=True, help="Path to text file with image paths (e.g. splits/val.txt) or image directory")
    parser.add_argument("--n", type=int, default=300, help="Number of images to process")
    parser.add_argument("--out-csv", default="artifacts/phase0/rho_LY.csv", help="Output CSV path")
    parser.add_argument("--out-fig", default="artifacts/phase0/L_maps.png", help="Output visualization PNG path")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_fig)), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading model {args.ckpt} on {device}...")
    model = YOLO(args.ckpt).model.eval().to(device)

    # Locate RFDBlock
    rfd_modules = [m for m in model.modules() if m.__class__.__name__ == "RFDBlock"]
    if not rfd_modules:
        raise RuntimeError("No RFDBlock module found in the checkpoint model!")
    rfd = rfd_modules[0]
    rfd.train()  # Ensure _illumination_map is recorded in forward pass

    buf = {}
    rfd.register_forward_hook(
        lambda m, i, o: buf.__setitem__("L", getattr(m, "_ill_stat", getattr(m, "_illumination_map", None)))
    )

    # Parse image paths
    if os.path.isfile(args.images):
        with open(args.images) as f:
            paths = [line.strip() for line in f if line.strip() and os.path.exists(line.strip())][: args.n]
    elif os.path.isdir(args.images):
        import glob
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(args.images, ext)))
        paths = paths[: args.n]
    else:
        raise FileNotFoundError(f"Cannot find image source: {args.images}")

    print(f"[INFO] Processing {len(paths)} images...")
    L_all, Y_all, examples = [], [], []

    for idx, path in enumerate(paths):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
        img_resized = cv2.resize(img_bgr, (640, 640))
        x = torch.from_numpy(img_resized[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().to(device) / 255.0

        with torch.no_grad():
            model(x)

        if buf.get("L") is None:
            continue

        L = buf["L"][0, 0].float().cpu().numpy()
        Y = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        Yd = cv2.resize(Y, (L.shape[1], L.shape[0]), interpolation=cv2.INTER_AREA)

        L_all.append(L.ravel())
        Y_all.append(Yd.ravel())

        if len(examples) < 3:
            examples.append((img_resized, L))

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(paths)} images...")

    if not L_all:
        raise RuntimeError("Failed to extract any illumination maps!")

    L_all = np.concatenate(L_all)
    Y_all = np.concatenate(Y_all)

    pr, _ = pearsonr(L_all, Y_all)
    sr, _ = spearmanr(L_all, Y_all)

    res = {
        "pearson": float(pr),
        "spearman": float(sr),
        "L_std": float(L_all.std()),
        "L_mean": float(L_all.mean()),
        "n_images": len(paths),
    }

    print("\n--- Correlation Results ---")
    print(f"Pearson correlation rho(L, Y) : {res['pearson']:+.4f}")
    print(f"Spearman rank correlation     : {res['spearman']:+.4f}")
    print(f"Illumination std dev (spatial): {res['L_std']:.4f}")
    print(f"Illumination mean             : {res['L_mean']:.4f}")

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res.keys()))
        w.writeheader()
        w.writerow(res)
    print(f"[INFO] Saved metrics to {args.out_csv}")

    # Plot figure
    fig, ax = plt.subplots(2, 4, figsize=(16, 7))
    for i, (img, L) in enumerate(examples):
        ax[0, i].imshow(img[:, :, ::-1])
        ax[0, i].set_title(f"Input {i + 1}", fontsize=10)
        ax[0, i].axis("off")

        im = ax[1, i].imshow(L, cmap="magma", vmin=0, vmax=1)
        ax[1, i].set_title(f"Map L {i + 1}", fontsize=10)
        ax[1, i].axis("off")
        fig.colorbar(im, ax=ax[1, i], fraction=0.046, pad=0.04)

    # Scatter plot on the 4th column
    sample_size = min(25000, len(L_all))
    sample_idx = np.random.choice(len(L_all), sample_size, replace=False)
    ax[0, 3].scatter(Y_all[sample_idx], L_all[sample_idx], s=1, alpha=0.15, c="darkblue")
    ax[0, 3].set_xlabel("Luminance Y (gray)", fontsize=9)
    ax[0, 3].set_ylabel("Illumination L", fontsize=9)
    ax[0, 3].set_title(f"Correlation (rho = {res['pearson']:.3f})", fontsize=10, fontweight="bold")
    ax[0, 3].grid(True, linestyle="--", alpha=0.5)

    ax[1, 3].hist(L_all[sample_idx], bins=50, color="teal", alpha=0.7, density=True)
    ax[1, 3].set_title(f"Dist of L (std={res['L_std']:.3f})", fontsize=10)
    ax[1, 3].set_xlabel("L value", fontsize=9)
    ax[1, 3].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=200)
    print(f"[INFO] Saved figure to {args.out_fig}")


if __name__ == "__main__":
    main()
