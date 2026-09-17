# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Stage F.2: Unified Final Validation Protocol for Paper C2.

Executes a single, standardized validation protocol across all model checkpoints:
- Explicit validation parameters (no implicit defaults)
- MD5 checksum recorded for each checkpoint
- COCO size breakdown (AP_small, AP_medium, AP_large, n_small, n_medium, n_large)
- Verbatim configuration logging
- Direct output to tables/C2_final_metrics.csv
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import torch
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
    # Fallback to search
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


def parse_args():
    parser = argparse.ArgumentParser(description="Unified final evaluation for Paper C2.")
    parser.add_argument("--data", default="exdark.yaml", help="Path to dataset yaml (exdark.yaml)")
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size")
    parser.add_argument("--out-csv", default="tables/C2_final_metrics.csv", help="Output CSV path")
    parser.add_argument("--out-art", default="artifacts/phase2/C2_final_metrics.csv", help="Artifact CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device
    if torch.cuda.is_available() and device != "cpu":
        device = int(device) if device.isdigit() else device
    else:
        device = "cpu"

    # Standardized validation protocol parameters explicitly stated
    val_params_dict = {
        "data": args.data,
        "imgsz": 640,
        "batch": args.batch,
        "conf": 0.001,
        "iou": 0.6,
        "max_det": 300,
        "rect": False,
        "split": "val",
        "device": str(device),
        "half": False,
        "plots": False,
        "save_json": True,
    }
    val_protocol = "Ultralytics DetectionValidator (faster-coco-eval)"
    val_params_str = (
        f"data={val_params_dict['data']}, imgsz={val_params_dict['imgsz']}, "
        f"batch={val_params_dict['batch']}, conf={val_params_dict['conf']}, "
        f"iou={val_params_dict['iou']}, max_det={val_params_dict['max_det']}, "
        f"rect={val_params_dict['rect']}, split={val_params_dict['split']}, "
        f"device={val_params_dict['device']}, half={val_params_dict['half']}"
    )

    print("=" * 80)
    print("  RI-YOLO: UNIFIED FINAL VALIDATION PROTOCOL (Stage F.2)")
    print("=" * 80)
    print(f"[PROTOCOL] {val_protocol}")
    print(f"[PARAMS]   {val_params_str}")
    print("=" * 80)

    rows = []

    for run_name, candidates in RUN_SPECS:
        ckpt_path = resolve_checkpoint(candidates, run_name)
        if ckpt_path is None or not ckpt_path.is_file():
            print(f"[WARN] Checkpoint for '{run_name}' NOT FOUND. Skipping.")
            continue

        ckpt_md5 = compute_md5(ckpt_path)
        print(f"\n[RUN] Evaluating '{run_name}'")
        print(f"      Path: {ckpt_path}")
        print(f"      MD5 : {ckpt_md5}")

        model = YOLO(str(ckpt_path))
        val_args = {
            **model.overrides,
            "mode": "val",
            **val_params_dict,
        }

        # Direct execution of DetectionValidator to capture full stats and object counts
        validator = model._smart_load("validator")(args=val_args, _callbacks=model.callbacks)
        val_stats = validator(model=model.model)
        model.metrics = validator.metrics
        res = validator.metrics
        d = res.results_dict if hasattr(res, "results_dict") else {}

        map50 = float(d.get("metrics/mAP50(B)", getattr(res.box, "map50", 0.0)))
        map50_95 = float(d.get("metrics/mAP50-95(B)", getattr(res.box, "map", 0.0)))

        # Extract size metrics (AP_small, AP_medium, AP_large)
        ap_s, ap_m, ap_l = None, None, None
        if isinstance(val_stats, dict):
            ap_s = val_stats.get("metrics/mAP_small(B)")
            ap_m = val_stats.get("metrics/mAP_medium(B)")
            ap_l = val_stats.get("metrics/mAP_large(B)")

        if hasattr(validator, "stats") and isinstance(validator.stats, dict):
            if ap_s is None:
                ap_s = validator.stats.get("metrics/mAP_small(B)")
            if ap_m is None:
                ap_m = validator.stats.get("metrics/mAP_medium(B)")
            if ap_l is None:
                ap_l = validator.stats.get("metrics/mAP_large(B)")

        # Extract ground truth object counts from validator.gdict
        n_s, n_m, n_l = None, None, None
        gdict = getattr(validator, "gdict", None)
        if gdict and isinstance(gdict, dict) and "annotations" in gdict:
            anns = gdict["annotations"]
            n_s = len([a for a in anns if a.get("area", 0) < 1024 and not a.get("iscrowd", 0)])
            n_m = len([a for a in anns if 1024 <= a.get("area", 0) < 9216 and not a.get("iscrowd", 0)])
            n_l = len([a for a in anns if a.get("area", 0) >= 9216 and not a.get("iscrowd", 0)])

        # Fallback to direct faster-coco-eval if needed
        if (ap_s is None or n_s is None) and getattr(validator, "gdict", None) and getattr(validator, "jdict", None):
            try:
                from faster_coco_eval import COCO, COCOeval_faster
                anno = COCO(validator.gdict)
                if n_s is None:
                    anns = list(anno.anns.values())
                    n_s = len([a for a in anns if a.get("area", 0) < 1024 and not a.get("iscrowd", 0)])
                    n_m = len([a for a in anns if 1024 <= a.get("area", 0) < 9216 and not a.get("iscrowd", 0)])
                    n_l = len([a for a in anns if a.get("area", 0) >= 9216 and not a.get("iscrowd", 0)])
                pred = anno.loadRes(validator.jdict)
                val_eval = COCOeval_faster(anno, pred, iouType="bbox")
                val_eval.params.imgIds = (
                    anno.getImgIds()
                    if validator.gdict
                    else [int(Path(x).stem) for x in validator.dataloader.dataset.im_files]
                )
                val_eval.evaluate()
                val_eval.accumulate()
                val_eval.summarize()
                if hasattr(val_eval, "stats_as_dict"):
                    ap_s = val_eval.stats_as_dict.get("AP_small", ap_s)
                    ap_m = val_eval.stats_as_dict.get("AP_medium", ap_m)
                    ap_l = val_eval.stats_as_dict.get("AP_large", ap_l)
            except Exception as e:
                print(f"[WARN] faster-coco-eval direct fallback error: {e}")

        row = {
            "run": run_name,
            "ckpt_path": str(ckpt_path).replace("\\", "/"),
            "ckpt_md5": ckpt_md5,
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "AP_small": round(float(ap_s), 4) if ap_s is not None else "NA",
            "AP_medium": round(float(ap_m), 4) if ap_m is not None else "NA",
            "AP_large": round(float(ap_l), 4) if ap_l is not None else "NA",
            "n_small": n_s if n_s is not None else "NA",
            "n_medium": n_m if n_m is not None else "NA",
            "n_large": n_l if n_l is not None else "NA",
            "val_protocol": val_protocol,
            "val_params": val_params_str,
        }
        rows.append(row)
        print(f"[RESULT] mAP50={row['mAP50']}, mAP50-95={row['mAP50_95']}, "
              f"AP_s={row['AP_small']}, AP_m={row['AP_medium']}, AP_l={row['AP_large']} "
              f"(n_s={row['n_small']}, n_m={row['n_medium']}, n_l={row['n_large']})")

    # Save to outputs
    for out_p in [args.out_csv, args.out_art]:
        p = Path(out_p)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        print(f"[SUCCESS] Wrote final metrics table to {p} ({len(rows)} runs)")

    print("\n" + "=" * 80)
    print("                      C2 FINAL METRICS TABLE")
    print("=" * 80)
    if rows:
        import pandas as pd
        df_res = pd.DataFrame(rows)
        print(df_res.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
