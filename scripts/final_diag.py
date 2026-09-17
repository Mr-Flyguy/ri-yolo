# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Stage F.3: Unified Final Diagnostic Script for Paper C2.

Evaluates Retinex illumination map diagnostics across all model checkpoints:
- Fixed set of exactly 300 images explicitly stored in artifacts/phase2/diag_image_list.txt
- Single deterministic luminance downsampling protocol (cv2.INTER_AREA)
- Checkpoint MD5 tracking
- Metrics: L_mean, L_std, pearson, spearman, L_sat_frac, mean(1 - L), gamma_final
- Direct output to tables/C2_final_diag.csv
"""

import argparse
import csv
import glob
import hashlib
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch
import yaml
from ultralytics import YOLO


RUN_SPECS = [
    ("e2p__baseline__s0", [
        "runs/detect/runs/e2p__baseline__s0/weights/best.pt",
        "runs/e2p__baseline__s0/weights/best.pt",
    ]),
    ("e2p__baseline__s1", [
        "runs/detect/runs/e2p__baseline__s1/weights/best.pt",
        "runs/e2p__baseline__s1/weights/best.pt",
    ]),
    ("e2p__baseline__s2", [
        "runs/detect/runs/e2p__baseline__s2/weights/best.pt",
        "runs/e2p__baseline__s2/weights/best.pt",
    ]),
    ("e2p__baseline__s3", [
        "runs/detect/runs/e2p__baseline__s3/weights/best.pt",
        "runs/e2p__baseline__s3/weights/best.pt",
    ]),

    ("e2p__postsppf__s0", [
        "runs/detect/runs/e3p__pos-postsppf__s0/weights/best.pt",
        "runs/detect/runs/e2p__postsppf__s0/weights/best.pt",
        "runs/e3p__pos-postsppf__s0/weights/best.pt",
        "runs/e2p__postsppf__s0/weights/best.pt",
    ]),
    ("e2p__postsppf__s1", [
        "runs/detect/runs/e2p__postsppf__s1/weights/best.pt",
        "runs/e2p__postsppf__s1/weights/best.pt",
    ]),
    ("e2p__postsppf__s2", [
        "runs/detect/runs/e2p__postsppf__s2/weights/best.pt",
        "runs/e2p__postsppf__s2/weights/best.pt",
    ]),
    ("e2p__postsppf__s3", [
        "runs/detect/runs/e2p__postsppf__s3/weights/best.pt",
        "runs/e2p__postsppf__s3/weights/best.pt",
    ]),
    ("e6p__lam-1e-5__s0", [
        "runs/detect/runs/e6p__lam-1e-5__s0/weights/best.pt",
        "runs/e6p__lam-1e-5__s0/weights/best.pt",
    ]),
    ("e6p__lam-1e-4__s0", [
        "runs/detect/runs/e6p__lam-1e-4__s0/weights/best.pt",
        "runs/e6p__lam-1e-4__s0/weights/best.pt",
    ]),
    ("e6p__lam-1e-4__s1", [
        "runs/detect/runs/e6p__lam-1e-4__s1/weights/best.pt",
        "runs/e6p__lam-1e-4__s1/weights/best.pt",
    ]),
    ("e6p__lam-1e-4__s2", [
        "runs/detect/runs/e6p__lam-1e-4__s2/weights/best.pt",
        "runs/e6p__lam-1e-4__s2/weights/best.pt",
    ]),
    ("e6p__lam-1e-3__s0", [
        "runs/detect/runs/e6p__lam-1e-3__s0/weights/best.pt",
        "runs/e6p__lam-1e-3__s0/weights/best.pt",
    ]),
    ("e6p__lam-1e-2__s0", [
        "runs/detect/runs/e6p__lam-1e-2__s0/weights/best.pt",
        "runs/e6p__lam-1e-2__s0/weights/best.pt",
    ]),
    ("e6p__lam-1e-1__s0", [
        "runs/detect/runs/e6p__lam-1e-1__s0/weights/best.pt",
        "runs/e6p__lam-1e-1__s0/weights/best.pt",
    ]),
    ("e7p__nwd-calib__s0", [
        "runs/detect/runs/e7p__nwd-calib__s0/weights/best.pt",
        "runs/e7p__nwd-calib__s0/weights/best.pt",
    ]),
    ("e7p__nwd-scaleinv__s0", [
        "runs/detect/runs/e7p__nwd-scaleinv__s0/weights/best.pt",
        "runs/e7p__nwd-scaleinv__s0/weights/best.pt",
    ]),
    ("e7p__nwd-calib-norsl__s0", [
        "runs/detect/runs/e7p__nwd-calib-norsl__s0/weights/best.pt",
        "runs/e7p__nwd-calib-norsl__s0/weights/best.pt",
    ]),
    ("runs_fixed/e7p__nwd-sizegate__s0", [
        "runs_fixed/e7p__nwd-sizegate__s0/weights/best.pt",
        "runs/detect/runs_fixed/e7p__nwd-sizegate__s0/weights/best.pt",
        "runs_fixed/detect/runs/e7p__nwd-sizegate__s0/weights/best.pt",
    ]),
    ("runs/detect/runs/e7p__nwd-sizegate__s0", [
        "runs/detect/runs/e7p__nwd-sizegate__s0/weights/best.pt",
        "runs/e7p__nwd-sizegate__s0/weights/best.pt",
    ]),
    ("e3p__pos-p3__s0", ["runs/detect/runs/e3p__pos-p3__s0/weights/best.pt", "runs/e3p__pos-p3__s0/weights/best.pt"]),
    ("e3p__pos-p4__s0", ["runs/detect/runs/e3p__pos-p4__s0/weights/best.pt", "runs/e3p__pos-p4__s0/weights/best.pt"]),
    ("e3p__pos-presppf__s0", ["runs/detect/runs/e3p__pos-presppf__s0/weights/best.pt", "runs/e3p__pos-presppf__s0/weights/best.pt"]),
    ("e3p__capctrl__s0", ["runs/detect/runs/e3p__capctrl__s0/weights/best.pt", "runs/e3p__capctrl__s0/weights/best.pt"]),
    ("e3p__capctrl__s1", ["runs/detect/runs/e3p__capctrl__s1/weights/best.pt", "runs/e3p__capctrl__s1/weights/best.pt"]),
    ("e3p__capctrl__s2", ["runs/detect/runs/e3p__capctrl__s2/weights/best.pt", "runs/e3p__capctrl__s2/weights/best.pt"]),
    ("e5p__baseline-ep40__s0", ["runs/detect/runs/e5p__baseline-ep40__s0/weights/best.pt", "runs/e5p__baseline-ep40__s0/weights/best.pt"]),
    ("e5p__baseline-ep60__s0", ["runs/detect/runs/e5p__baseline-ep60__s0/weights/best.pt", "runs/e5p__baseline-ep60__s0/weights/best.pt"]),
    ("e5p__postsppf-ep40__s0", ["runs/detect/runs/e5p__postsppf-ep40__s0/weights/best.pt", "runs/e5p__postsppf-ep40__s0/weights/best.pt"]),
    ("e5p__postsppf-ep60__s0", ["runs/detect/runs/e5p__postsppf-ep60__s0/weights/best.pt", "runs/e5p__postsppf-ep60__s0/weights/best.pt"]),
    ("e4p__mosaic-close0__s0", ["runs/detect/runs/e4p__mosaic-close0__s0/weights/best.pt", "runs/e4p__mosaic-close0__s0/weights/best.pt"]),
    ("e4p__mosaic-close50__s0", ["runs/detect/runs/e4p__mosaic-close50__s0/weights/best.pt", "runs/e4p__mosaic-close50__s0/weights/best.pt"]),
]


def compute_md5(file_path: Path) -> str:
    """Compute MD5 checksum for a file."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_checkpoint(candidates, run_name: str) -> Path:
    """Resolve checkpoint file path from candidates or filesystem search."""
    for c in candidates:
        p = Path(c)
        if p.is_file():
            return p
    clean_name = run_name.split("/")[-1]
    matches = list(Path(".").glob(f"**/{clean_name}/weights/best.pt"))
    if matches:
        if "fixed" in run_name:
            fixed_matches = [m for m in matches if "fixed" in str(m)]
            if fixed_matches:
                return fixed_matches[0]
        else:
            non_fixed = [m for m in matches if "fixed" not in str(m)]
            if non_fixed:
                return non_fixed[0]
        return matches[0]
    return None


def get_or_create_image_list(data_yaml: str, list_file: Path, n_images: int = 300) -> list:
    """Read existing image list or generate deterministically and save."""
    if list_file.is_file():
        with open(list_file, "r", encoding="utf-8") as f:
            paths = [line.strip() for line in f if line.strip() and os.path.exists(line.strip())]
        if len(paths) >= n_images:
            print(f"[INFO] Using existing image list from {list_file} ({len(paths)} images)")
            return paths[:n_images]

    # Discover validation images from dataset yaml
    val_images = []
    if os.path.exists(data_yaml):
        with open(data_yaml, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        root = Path(d.get("path", "."))
        val_spec = d.get("val", "")
        val_path = root / val_spec if not Path(val_spec).is_absolute() else Path(val_spec)

        if val_path.is_file():  # text file of paths
            with open(val_path, "r", encoding="utf-8") as f:
                val_images = [line.strip() for line in f if line.strip()]
        elif val_path.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                val_images.extend(glob.glob(str(val_path / ext)))
                val_images.extend(glob.glob(str(val_path / "**" / ext), recursive=True))

    if not val_images:
        # Fallback to local splits/val.txt or datasets/ExDark
        for cand in ["splits/val.txt", "datasets/ExDark/images/val"]:
            cp = Path(cand)
            if cp.is_file():
                with open(cp, "r", encoding="utf-8") as f:
                    val_images = [line.strip() for line in f if line.strip()]
                break
            elif cp.is_dir():
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                    val_images.extend(glob.glob(str(cp / ext)))
                break

    val_images = sorted(list(set([str(Path(p).resolve()) for p in val_images if os.path.exists(p)])))
    selected = val_images[:n_images]

    list_file.parent.mkdir(parents=True, exist_ok=True)
    with open(list_file, "w", encoding="utf-8") as f:
        for p in selected:
            f.write(p + "\n")
    print(f"[INFO] Saved deterministic image list to {list_file} ({len(selected)} images)")
    return selected


def find_gamma_final(run_name: str, ckpt_path: Path, model) -> float:
    """Find final gamma value from model parameter or rfd_log.csv."""
    # Check model parameter first
    for n, p in model.named_parameters():
        if "gamma" in n:
            try:
                return float(p.detach().cpu().item())
            except Exception:
                pass

    # Check rfd_log.csv in checkpoint directory or run directory
    run_dir = ckpt_path.parent.parent
    for rfd_cand in [
        run_dir / "rfd_log.csv",
        run_dir.parent / run_name / "rfd_log.csv",
        Path(f"runs/detect/runs/{run_name}/rfd_log.csv"),
    ]:
        if rfd_cand.is_file():
            try:
                df = pd.read_csv(rfd_cand)
                if "gamma" in df.columns and len(df) > 0:
                    return float(df["gamma"].iloc[-1])
            except Exception:
                pass
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Unified final diagnostic evaluation for Paper C2.")
    parser.add_argument("--data", default="exdark.yaml", help="Path to dataset yaml (exdark.yaml)")
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--n", type=int, default=300, help="Number of validation images to evaluate")
    parser.add_argument("--image-list", default="artifacts/phase2/diag_image_list.txt", help="Deterministic image list file")
    parser.add_argument("--out-csv", default="tables/C2_final_diag.csv", help="Output CSV path")
    parser.add_argument("--out-art", default="artifacts/phase2/C2_final_diag.csv", help="Artifact CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda:" + args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    if not torch.cuda.is_available():
        device = "cpu"

    image_list_path = Path(args.image_list)
    images = get_or_create_image_list(args.data, image_list_path, n_images=args.n)
    if not images:
        raise RuntimeError(f"No validation images found using data={args.data}")

    print("=" * 80)
    print("  RI-YOLO: UNIFIED FINAL ILLUMINATION MAP DIAGNOSTICS (Stage F.3)")
    print("=" * 80)
    print(f"[IMAGES]  Evaluating {len(images)} images from: {image_list_path}")
    print(f"[DEVICE]  {device}")
    print("=" * 80)

    rows = []

    for run_name, candidates in RUN_SPECS:
        ckpt_path = resolve_checkpoint(candidates, run_name)
        if ckpt_path is None or not ckpt_path.is_file():
            print(f"[WARN] Checkpoint for '{run_name}' NOT FOUND. Skipping.")
            continue

        ckpt_md5 = compute_md5(ckpt_path)
        print(f"\n[RUN] Diagnosing '{run_name}'")
        print(f"      Path: {ckpt_path}")
        print(f"      MD5 : {ckpt_md5}")

        model_wrapper = YOLO(str(ckpt_path))
        model = model_wrapper.model.eval().to(device)

        rfd_modules = [m for m in model.modules() if m.__class__.__name__ in ("RFDBlock", "RFDBlockNoGate")]
        if not rfd_modules or rfd_modules[0].__class__.__name__ == "RFDBlockNoGate":
            print(f"      [INFO] Model has no RFDBlock (or is NoGate). Writing NA for illumination map.")
            gamma_val = find_gamma_final(run_name, ckpt_path, model)
            row = {
                "run": run_name,
                "ckpt_path": str(ckpt_path).replace("\\", "/"),
                "ckpt_md5": ckpt_md5,
                "L_mean": "NA",
                "L_std": "NA",
                "pearson": "NA",
                "spearman": "NA",
                "L_sat_frac": "NA",
                "mean_one_minus_L": "NA",
                "gamma_final": round(float(gamma_val), 6) if gamma_val is not None else "NA",
            }
            rows.append(row)
            continue

        rfd = rfd_modules[0]
        # Keep entire model strictly in eval() mode. Do NOT call rfd.train()
        # because rfd.cv1 (BatchNorm2d) would recompute stats on single image batches.
        # Instead, hook directly onto ill_estimator output!
        buf = {}
        hook = rfd.ill_estimator.register_forward_hook(
            lambda m, i, o: buf.__setitem__("L", o.detach())
        )

        L_all, Y_all = [], []

        for idx, img_p in enumerate(images):
            img_bgr = cv2.imread(img_p)
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

        hook.remove()

        if not L_all:
            print(f"      [WARN] No illumination maps captured for {run_name}!")
            continue

        L_flat = np.concatenate(L_all)
        Y_flat = np.concatenate(Y_all)

        pr, _ = pearsonr(L_flat, Y_flat)
        sr, _ = spearmanr(L_flat, Y_flat)
        l_mean = float(L_flat.mean())
        l_std = float(L_flat.std())
        l_sat_frac = float(np.mean(L_flat >= 0.95))
        mean_one_minus_L = float(np.mean(1.0 - L_flat))
        gamma_final = find_gamma_final(run_name, ckpt_path, model)

        row = {
            "run": run_name,
            "ckpt_path": str(ckpt_path).replace("\\", "/"),
            "ckpt_md5": ckpt_md5,
            "L_mean": round(l_mean, 4),
            "L_std": round(l_std, 4),
            "pearson": round(float(pr), 4),
            "spearman": round(float(sr), 4),
            "L_sat_frac": round(l_sat_frac, 4),
            "mean_one_minus_L": round(mean_one_minus_L, 4),
            "gamma_final": round(float(gamma_final), 6) if gamma_final is not None else "NA",
        }
        rows.append(row)
        print(f"[RESULT] L_mean={row['L_mean']}, L_std={row['L_std']}, "
              f"rho(L,Y)={row['pearson']}, spearman={row['spearman']}, "
              f"L_sat_frac={row['L_sat_frac']}, mean(1-L)={row['mean_one_minus_L']}, "
              f"gamma={row['gamma_final']}")

    # Save to CSV
    for out_p in [args.out_csv, args.out_art]:
        p = Path(out_p)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        print(f"[SUCCESS] Wrote final diagnostics table to {p} ({len(rows)} runs)")

    print("\n" + "=" * 80)
    print("                      C2 FINAL DIAGNOSTICS TABLE")
    print("=" * 80)
    if rows:
        df_res = pd.DataFrame(rows)
        print(df_res.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
