# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Evaluation of checkpoints on held-out test split for generalisation assessment."""

import argparse
import csv
import glob
import os
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on held-out test split.")
    parser.add_argument("--data", default="exdark.yaml", help="Path to dataset YAML (default: exdark.yaml)")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate (default: test)")
    parser.add_argument("--ckpts", nargs="+", required=True, help="Checkpoint paths or glob patterns")
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu' (default: 0)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--out", default="tables/C3_test_generalization.csv", help="Output CSV path")
    return parser.parse_args()


def main():
    args = parse_args()

    # Expand any glob patterns in ckpts
    expanded_ckpts = []
    for c in args.ckpts:
        matches = glob.glob(c)
        if matches:
            expanded_ckpts.extend(matches)
        elif Path(c).exists():
            expanded_ckpts.append(c)

    expanded_ckpts = sorted(list(set(expanded_ckpts)))
    if not expanded_ckpts:
        print(f"[ERROR] No checkpoints matched: {args.ckpts}")
        return

    rows = []
    for ck in expanded_ckpts:
        run_name = Path(ck).parent.parent.name if Path(ck).parent.name == "weights" else Path(ck).stem
        print(f"\n{'='*70}\n[INFO] Evaluating {run_name} ({ck}) on split '{args.split}'...\n{'='*70}")

        try:
            model = YOLO(ck)
            res = model.val(
                data=args.data,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                split=args.split,
                plots=False,
            )
            d = res.results_dict if hasattr(res, "results_dict") else {}
            map50 = float(d.get("metrics/mAP50(B)", getattr(res.box, "map50", 0.0)))
            map50_95 = float(d.get("metrics/mAP50-95(B)", getattr(res.box, "map", 0.0)))

            row = {
                "run": run_name,
                "split": args.split,
                "mAP50": round(map50, 4),
                "mAP50_95": round(map50_95, 4),
            }
            rows.append(row)
            print(f"[RESULT] {row}")
        except Exception as e:
            print(f"[ERROR] Failed evaluation on {ck}: {e}")

    if rows:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[SUCCESS] Test generalisation saved to {args.out}")


if __name__ == "__main__":
    main()
