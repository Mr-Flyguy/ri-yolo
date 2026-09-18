# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Stage F.5: Ground Truth Object Count by COCO Size Category.

Calculates exact bounding box area distributions directly from dataset label files:
- Standard COCO area definition at imgsz=640:
    Small : area < 32^2 (1024 px^2)
    Medium: 32^2 <= area < 96^2 (1024 <= area < 9216 px^2)
    Large : area >= 96^2 (9216 px^2)
- Percentiles (10, 25, 50, 75, 90) of bounding box area and dimensions
- Evaluates discrepancy between C_calib.txt and COCO definition
"""

import argparse
import glob
import os
from pathlib import Path
import numpy as np
import yaml


def find_label_dirs(data_yaml: str):
    label_dirs = {}
    if os.path.exists(data_yaml):
        with open(data_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        root = Path(cfg.get("path", "."))
        for split in ["val", "train", "test"]:
            spec = cfg.get(split)
            if spec:
                p = root / spec if not Path(spec).is_absolute() else Path(spec)
                # Labels directory corresponds to replacing 'images' with 'labels'
                lp = Path(str(p).replace("images", "labels"))
                if lp.is_dir():
                    label_dirs[split] = lp
                elif p.is_dir():
                    label_dirs[split] = p

    # Fallback search
    for split in ["val", "train", "test"]:
        if split not in label_dirs:
            for cand in [
                Path(f"datasets/ExDark/labels/{split}"),
                Path(f"datasets/ExDark/images/{split}"),
                Path(f"ExDark/labels/{split}"),
            ]:
                if cand.is_dir():
                    label_dirs[split] = cand
                    break
    return label_dirs


def parse_labels_from_dir(lbl_dir: Path, imgsz: int = 640):
    files = sorted(list(set(lbl_dir.rglob("*.txt"))))
    boxes = []  # list of (w_px, h_px, area)

    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        w = float(parts[3])
                        h = float(parts[4])
                        w_px = w * imgsz
                        h_px = h * imgsz
                        area = w_px * h_px
                        boxes.append((w_px, h_px, area))
                    except ValueError:
                        continue
    return boxes


def analyze_boxes(boxes, name: str, imgsz: int = 640):
    if not boxes:
        print(f"[WARN] No boxes found for {name}.")
        return

    n_total = len(boxes)
    w_arr = np.array([b[0] for b in boxes])
    h_arr = np.array([b[1] for b in boxes])
    area_arr = np.array([b[2] for b in boxes])
    max_dim_arr = np.maximum(w_arr, h_arr)

    # 1. Standard COCO Area Definition
    small_coco = area_arr < 1024.0
    med_coco = (area_arr >= 1024.0) & (area_arr < 9216.0)
    large_coco = area_arr >= 9216.0

    n_s_coco = int(small_coco.sum())
    n_m_coco = int(med_coco.sum())
    n_l_coco = int(large_coco.sum())

    # 2. Linear Dimension Definition (e.g. C_calib.txt max_dim < 32 px)
    small_dim = max_dim_arr < 32.0
    med_dim = (max_dim_arr >= 32.0) & (max_dim_arr < 96.0)
    large_dim = max_dim_arr >= 96.0

    n_s_dim = int(small_dim.sum())
    n_m_dim = int(med_dim.sum())
    n_l_dim = int(large_dim.sum())

    pcts = [10, 25, 50, 75, 90]
    area_pcts = np.percentile(area_arr, pcts)
    sqrt_area_pcts = np.percentile(np.sqrt(area_arr), pcts)
    max_dim_pcts = np.percentile(max_dim_arr, pcts)

    print("=" * 80)
    print(f"  EXDARK OBJECT SIZE BREAKDOWN: {name} (imgsz={imgsz})")
    print("=" * 80)
    print(f"Total objects parsed: {n_total}")
    print("\n--- 1. STANDARD COCO DEFINITION (by Area: px^2) ---")
    print(f"  Small  (area < 32^2 = 1024 px^2)         : {n_s_coco:>5} ({n_s_coco / n_total * 100:.2f}%)")
    print(f"  Medium (1024 <= area < 96^2 = 9216 px^2) : {n_m_coco:>5} ({n_m_coco / n_total * 100:.2f}%)")
    print(f"  Large  (area >= 9216 px^2)               : {n_l_coco:>5} ({n_l_coco / n_total * 100:.2f}%)")
    print(f"  Sum check                                : {n_s_coco + n_m_coco + n_l_coco} / {n_total}")

    print("\n--- 2. LINEAR DIMENSION DEFINITION (by max(w, h): px) [Used in C_calib.txt] ---")
    print(f"  Small  (max_dim < 32 px)                 : {n_s_dim:>5} ({n_s_dim / n_total * 100:.2f}%)")
    print(f"  Medium (32 <= max_dim < 96 px)           : {n_m_dim:>5} ({n_m_dim / n_total * 100:.2f}%)")
    print(f"  Large  (max_dim >= 96 px)                : {n_l_dim:>5} ({n_l_dim / n_total * 100:.2f}%)")
    print(f"  Sum check                                : {n_s_dim + n_m_dim + n_l_dim} / {n_total}")

    print("\n--- 3. PERCENTILES [10 / 25 / 50 / 75 / 90] ---")
    print(f"  Area (px^2)       : [{area_pcts[0]:.1f}, {area_pcts[1]:.1f}, {area_pcts[2]:.1f}, {area_pcts[3]:.1f}, {area_pcts[4]:.1f}]")
    print(f"  sqrt(Area) (px)   : [{sqrt_area_pcts[0]:.1f}, {sqrt_area_pcts[1]:.1f}, {sqrt_area_pcts[2]:.1f}, {sqrt_area_pcts[3]:.1f}, {sqrt_area_pcts[4]:.1f}]")
    print(f"  max(w, h) (px)    : [{max_dim_pcts[0]:.1f}, {max_dim_pcts[1]:.1f}, {max_dim_pcts[2]:.1f}, {max_dim_pcts[3]:.1f}, {max_dim_pcts[4]:.1f}]")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="Count ground truth objects by size category")
    parser.add_argument("--data", default="exdark.yaml", help="Path to dataset yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution for scaling")
    return parser.parse_args()


def main():
    args = parse_args()
    label_dirs = find_label_dirs(args.data)
    if not label_dirs:
        raise RuntimeError(f"Could not find label directories from {args.data}")

    all_boxes = {}
    for split, p in label_dirs.items():
        b = parse_labels_from_dir(p, imgsz=args.imgsz)
        all_boxes[split] = b
        analyze_boxes(b, name=f"Split '{split}' ({p})", imgsz=args.imgsz)

    if "val" in all_boxes and "train" in all_boxes:
        comb = all_boxes["train"] + all_boxes["val"]
        if "test" in all_boxes:
            comb += all_boxes["test"]
            analyze_boxes(comb, name="Combined (Train + Val + Test)", imgsz=args.imgsz)
        else:
            analyze_boxes(comb, name="Combined (Train + Val)", imgsz=args.imgsz)


if __name__ == "__main__":
    main()
