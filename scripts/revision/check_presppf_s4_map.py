# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# -*- coding: utf-8 -*-
"""Diagnostic script to recalculate pearson_rec601_area for a checkpoint and inspect per-frame correlations."""

import argparse
from pathlib import Path
import tempfile

import numpy as np

from scripts.revision.config import DEFAULT_DATA_YAML, REPO_ROOT, resolve_data_yaml
from scripts.revision.reepe_overnight import ckpt_of, map_stats


def main():
    """Recalculate map_stats into a temporary file and inspect NaN / 0 correlations."""
    ap = argparse.ArgumentParser(description="Check per-frame correlations for RFD illumination map")
    ap.add_argument("--run", default="e3p__pos-presppf__s4", help="Run name (default: e3p__pos-presppf__s4)")
    ap.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    ap.add_argument("--data", default=str(DEFAULT_DATA_YAML), help="Path to exdark.yaml")
    ap.add_argument("--out", default=None, help="Path to temporary output CSV (default: auto tempfile)")
    args = ap.parse_args()

    data_yaml = resolve_data_yaml(args.data)

    # 1. Resolve checkpoint
    ckpt = ckpt_of(args.run)
    if ckpt is None or not Path(ckpt).is_file():
        candidates = [
            REPO_ROOT / "runs" / "detect" / "runs" / args.run / "weights" / "best.pt",
            REPO_ROOT / "runs" / args.run / "weights" / "best.pt",
            REPO_ROOT / "runs" / "detect" / "runs" / args.run / "weights" / "last.pt",
        ]
        for c in candidates:
            if c.is_file():
                ckpt = c
                break

    if ckpt is None or not Path(ckpt).is_file():
        print(f"[ERROR] Checkpoint for '{args.run}' not found!")
        print("Checked candidate paths:")
        print(f"  - runs/detect/runs/{args.run}/weights/best.pt")
        print(f"  - runs/{args.run}/weights/best.pt")
        return

    # 2. Setup temporary output path
    if args.out:
        temp_csv = Path(args.out)
    else:
        temp_dir = Path(tempfile.gettempdir()) / "ri_yolo_diag"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_csv = temp_dir / f"map_stats_{args.run}.csv"

    print("=" * 70)
    print(f"Recalculating map_stats for: {args.run}")
    print(f"Checkpoint: {ckpt}")
    print(f"Data YAML : {data_yaml}")
    print(f"Temp CSV  : {temp_csv}")
    print("=" * 70)

    # 3. Call map_stats
    row, rs = map_stats(args.run, ckpt, data_yaml, args.device, out_path=temp_csv)

    if not rs:
        print("[ERROR] No correlation results returned!")
        return

    # 4. Analyze per-frame correlations
    rs_arr = np.array(rs, dtype=np.float64)
    total_frames = len(rs_arr)
    nan_mask = np.isnan(rs_arr)
    nan_count = int(np.sum(nan_mask))

    # Exactly zero (or zero assigned due to std < 1e-8)
    zero_mask = (~nan_mask) & (rs_arr == 0.0)
    zero_count = int(np.sum(zero_mask))

    valid_mask = (~nan_mask) & (rs_arr != 0.0)
    valid_rs = rs_arr[valid_mask]
    valid_count = len(valid_rs)

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ АНАЛИЗА ПОКАДРОВЫХ КОРРЕЛЯЦИЙ")
    print("=" * 70)
    print(f"Временный CSV файл: {temp_csv}")
    print(f"Итоговый pearson_rec601_area (из CSV) : {row.get('pearson_rec601_area')}")
    print(f"L_mean: {row.get('L_mean')} | L_sd_pooled: {row.get('L_sd_pooled')} | map_hw: {row.get('map_hw')}")
    print("-" * 70)
    print(f"Всего обработано кадров (n_images) : {total_frames}")
    print(f"Корреляций равных NaN              : {nan_count}")
    print(f"Корреляций равных 0 (L.std < 1e-8) : {zero_count}")
    print(f"Ненулевых корреляций               : {valid_count}")

    if valid_count > 0:
        print("-" * 70)
        print(f"Статистика ненулевых корреляций ({valid_count} кадров):")
        print(f"  Min   : {np.min(valid_rs):.4f}")
        print(f"  Mean  : {np.mean(valid_rs):.4f}")
        print(f"  Median: {np.median(valid_rs):.4f}")
        print(f"  Max   : {np.max(valid_rs):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
