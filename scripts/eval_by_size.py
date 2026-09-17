# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Evaluation breakdown across object scale categories (small, medium, large)."""

import argparse
import csv
import os
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate detection metrics broken down by object size.")
    parser.add_argument("--ckpts", nargs="+", required=True, help="List of model checkpoint paths (.pt)")
    parser.add_argument("--data", default="exdark.yaml", help="Path to dataset YAML (default: exdark.yaml)")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate (default: val)")
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu' (default: 0)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--out", default="tables/C2_table2_bysize.csv", help="Output CSV path")
    return parser.parse_args()


def resolve_ckpt(ck_str):
    p = Path(ck_str)
    if p.exists():
        return str(p)
    target_dir = "runs_fixed" if "runs_fixed" in str(ck_str) else "runs"
    # Check direct alternatives
    for base in [target_dir, "runs_fixed" if target_dir == "runs" else "runs"]:
        alt = Path(base) / ck_str
        if alt.exists():
            return str(alt)
        alt_detect = Path(base) / "detect" / ck_str
        if alt_detect.exists():
            return str(alt_detect)
        alt_nested = Path(base) / "detect" / "runs" / p.name
        if alt_nested.exists():
            return str(alt_nested)
        matches = list(Path(base).glob(f"**/{p.name}"))
        if matches:
            return str(matches[0])
    return ck_str


def main():
    args = parse_args()
    rows = []

    for ck in args.ckpts:
        resolved = resolve_ckpt(ck)
        run_name = Path(resolved).parent.parent.name if Path(resolved).parent.name == "weights" else Path(resolved).stem
        print(f"\n{'='*70}\n[INFO] Evaluating {run_name} from {resolved} on split '{args.split}'...\n{'='*70}")

        model = YOLO(resolved)
        res = model.val(
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            split=args.split,
            plots=False,
            save_json=True,
        )
        d = res.results_dict if hasattr(res, "results_dict") else {}

        map50 = float(d.get("metrics/mAP50(B)", getattr(res.box, "map50", 0.0)))
        map50_95 = float(d.get("metrics/mAP50-95(B)", getattr(res.box, "map", 0.0)))

        # Extract size metrics (from faster-coco-eval if available)
        ap_s, ap_m, ap_l = None, None, None
        try:
            if hasattr(res, "validator") and getattr(res.validator, "gdict", None) and getattr(res.validator, "jdict", None):
                from faster_coco_eval import COCO, COCOeval_faster
                anno = COCO(res.validator.gdict)
                pred = anno.loadRes(res.validator.jdict)
                val_eval = COCOeval_faster(anno, pred, iouType="bbox")
                val_eval.evaluate()
                val_eval.accumulate()
                val_eval.summarize()
                if ap_s is None and hasattr(val_eval, "stats") and len(val_eval.stats) >= 6:
                    ap_s = val_eval.stats[3]
                    ap_m = val_eval.stats[4]
                    ap_l = val_eval.stats[5]
        except Exception as e:
            print(f"[WARN] faster-coco-eval extraction failed: {e}")

        if ap_s is None:
            ap_s = d.get("metrics/mAP_small(B)", None)
            ap_m = d.get("metrics/mAP_medium(B)", None)
            ap_l = d.get("metrics/mAP_large(B)", None)

        if ap_s is None and hasattr(res.box, "aps") and len(res.box.aps) >= 3:
            ap_s = res.box.aps[0]
            ap_m = res.box.aps[1]
            ap_l = res.box.aps[2]

        row = {
            "ckpt": run_name,
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "AP_small": round(float(ap_s), 4) if ap_s is not None else "NA",
            "AP_medium": round(float(ap_m), 4) if ap_m is not None else "NA",
            "AP_large": round(float(ap_l), 4) if ap_l is not None else "NA",
        }
        rows.append(row)
        print(f"[RESULT] {row}")

    existing_rows = {}
    if os.path.exists(args.out):
        try:
            with open(args.out, "r", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    existing_rows[r["ckpt"]] = r
        except Exception as e:
            print(f"[WARN] Failed to read existing {args.out}: {e}")

    for r in rows:
        existing_rows[r["ckpt"]] = r

    final_rows = list(existing_rows.values())
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()))
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"\n[SUCCESS] Table saved to {args.out}:")
    for r in final_rows:
        print(r)


if __name__ == "__main__":
    main()
