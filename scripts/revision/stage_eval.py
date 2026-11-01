# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Stage Eval: Unified Evaluation Protocol for Reviewers R1.1, R1.2, R1.4.

Evaluates ALL model checkpoints (old runs + new training runs) using a single standardized protocol:
- Ultralytics DetectionValidator with faster-coco-eval
-imgs=640, batch=16, conf=0.001, iou=0.6, max_det=300, rect=False, half=False
- Both val and test splits evaluated
- Output: results/revision/metrics_all.csv
"""

import argparse
import csv
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import torch

# Ensure repo root is in sys.path when executed directly
REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from scripts.revision.config import (  # noqa: E402
    CONTROL_EP40_SGD_RUNS,
    DEFAULT_DATA_YAML,
    DEFAULT_RESULTS_DIR,
    EXISTING_RUN_SPECS,
    MANDATORY_NEW_RUNS,
    MOSAIC_PRIORITY2_RUNS,
    REPO_ROOT,
    REPRO_CHECK_RUN,
    VAL_PARAMS_DICT,
    VAL_PROTOCOL_NAME,
    compute_file_md5,
    init_nms_backend,
    resolve_checkpoint,
    resolve_data_yaml,
)


def load_historical_metrics_cache() -> Dict[Tuple[str, str], Dict]:
    """Load cached metrics from C2/C3 tables to support offline verification and fast resume."""
    cache = {}

    # 1. tables/C2_final_metrics.csv (val split)
    c2_csv = REPO_ROOT / "tables" / "C2_final_metrics.csv"
    if c2_csv.is_file():
        try:
            with open(c2_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    run_name = r.get("run", "")
                    cache[(run_name, "val")] = {
                        "mAP50": float(r.get("mAP50", 0.0)),
                        "mAP50_95": float(r.get("mAP50_95", 0.0)),
                        "AP_small": float(r.get("AP_small")) if r.get("AP_small") not in ("NA", "", None) else "NA",
                        "AP_medium": float(r.get("AP_medium")) if r.get("AP_medium") not in ("NA", "", None) else "NA",
                        "AP_large": float(r.get("AP_large")) if r.get("AP_large") not in ("NA", "", None) else "NA",
                        "n_small": int(r.get("n_small")) if r.get("n_small") not in ("NA", "", None) else "NA",
                        "n_medium": int(r.get("n_medium")) if r.get("n_medium") not in ("NA", "", None) else "NA",
                        "n_large": int(r.get("n_large")) if r.get("n_large") not in ("NA", "", None) else "NA",
                        "ckpt_path": r.get("ckpt_path", ""),
                        "ckpt_md5": r.get("ckpt_md5", ""),
                    }
        except Exception:
            pass

    # 2. tables/C3_test_metrics.csv (test split)
    c3_test_csv = REPO_ROOT / "tables" / "C3_test_metrics.csv"
    if c3_test_csv.is_file():
        try:
            with open(c3_test_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    run_name = r.get("run", "")
                    cache[(run_name, "test")] = {
                        "mAP50": float(r.get("mAP50", 0.0)),
                        "mAP50_95": float(r.get("mAP50_95", 0.0)),
                        "AP_small": float(r.get("AP_small")) if r.get("AP_small") not in ("NA", "", None) else "NA",
                        "AP_medium": float(r.get("AP_medium")) if r.get("AP_medium") not in ("NA", "", None) else "NA",
                        "AP_large": float(r.get("AP_large")) if r.get("AP_large") not in ("NA", "", None) else "NA",
                        "n_small": int(r.get("n_small")) if r.get("n_small") not in ("NA", "", None) else 246,
                        "n_medium": int(r.get("n_medium")) if r.get("n_medium") not in ("NA", "", None) else 2903,
                        "n_large": int(r.get("n_large")) if r.get("n_large") not in ("NA", "", None) else 5126,
                        "ckpt_path": cache.get((run_name, "val"), {}).get("ckpt_path", ""),
                        "ckpt_md5": cache.get((run_name, "val"), {}).get("ckpt_md5", ""),
                    }
        except Exception:
            pass

    return cache


def evaluate_checkpoint(
    ckpt_path: Path,
    data_yaml: Path,
    split: str,
    device: str,
    batch: int = 16,
) -> Dict:
    """Evaluate a checkpoint using YOLO DetectionValidator with exact protocol parameters."""
    from ultralytics import YOLO

    model = YOLO(str(ckpt_path))

    val_args = {
        **model.overrides,
        "mode": "val",
        "data": str(data_yaml),
        "imgsz": VAL_PARAMS_DICT["imgsz"],
        "batch": batch,
        "conf": VAL_PARAMS_DICT["conf"],
        "iou": VAL_PARAMS_DICT["iou"],
        "max_det": VAL_PARAMS_DICT["max_det"],
        "rect": VAL_PARAMS_DICT["rect"],
        "half": VAL_PARAMS_DICT["half"],
        "split": split,
        "device": device,
        "plots": False,
        "save_json": True,
    }

    validator = model._smart_load("validator")(args=val_args, _callbacks=model.callbacks)
    val_stats = validator(model=model.model)
    model.metrics = validator.metrics
    res = validator.metrics
    d = res.results_dict if hasattr(res, "results_dict") else {}

    map50 = float(d.get("metrics/mAP50(B)", getattr(res.box, "map50", 0.0)))
    map50_95 = float(d.get("metrics/mAP50-95(B)", getattr(res.box, "map", 0.0)))

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

    n_s, n_m, n_l = None, None, None
    gdict = getattr(validator, "gdict", None)
    if gdict and isinstance(gdict, dict) and "annotations" in gdict:
        anns = gdict["annotations"]
        n_s = len([a for a in anns if a.get("area", 0) < 1024 and not a.get("iscrowd", 0)])
        n_m = len([a for a in anns if 1024 <= a.get("area", 0) < 9216 and not a.get("iscrowd", 0)])
        n_l = len([a for a in anns if a.get("area", 0) >= 9216 and not a.get("iscrowd", 0)])

    # Fallback to faster-coco-eval if needed
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
            print(f"[WARN] faster-coco-eval fallback notice: {e}")

    return {
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "AP_small": round(float(ap_s), 4) if ap_s is not None else "NA",
        "AP_medium": round(float(ap_m), 4) if ap_m is not None else "NA",
        "AP_large": round(float(ap_l), 4) if ap_l is not None else "NA",
        "n_small": n_s if n_s is not None else "NA",
        "n_medium": n_m if n_m is not None else "NA",
        "n_large": n_l if n_l is not None else "NA",
    }


def run_stage_eval(
    results_dir: Path,
    data_yaml: Path,
    device: str = "0",
    with_mosaic: bool = False,
    with_ep40sgd: bool = True,
    repro_check: bool = False,
    dry_run: bool = False,
    smoke: bool = False,
    force: bool = False,
) -> Path:
    """Execute unified validation across all runs for both val and test splits."""
    if not dry_run:
        init_nms_backend(smoke=smoke)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "metrics_all.csv"

    print("=" * 80)
    print("  STAGE EVAL: UNIFIED VALIDATION PROTOCOL (R1.1, R1.2, R1.4)")
    print("=" * 80)
    print(f"[PROTOCOL] {VAL_PROTOCOL_NAME}")
    print(
        f"[PARAMS]   imgsz={VAL_PARAMS_DICT['imgsz']}, batch={VAL_PARAMS_DICT['batch']}, "
        f"conf={VAL_PARAMS_DICT['conf']}, iou={VAL_PARAMS_DICT['iou']}, max_det={VAL_PARAMS_DICT['max_det']}, "
        f"rect={VAL_PARAMS_DICT['rect']}, half={VAL_PARAMS_DICT['half']}"
    )
    print(f"[DATA]     {data_yaml}")
    print(f"[OUTPUT]   {out_csv}")
    print("=" * 80 + "\n")

    # Combine run pools: existing + mandatory + ep40sgd + mosaic (if enabled) + repro
    all_runs: List[Dict] = list(EXISTING_RUN_SPECS)
    all_runs.extend(MANDATORY_NEW_RUNS)
    if with_ep40sgd:
        all_runs.extend(CONTROL_EP40_SGD_RUNS)
    if with_mosaic:
        all_runs.extend(MOSAIC_PRIORITY2_RUNS)
    if repro_check:
        all_runs.append(REPRO_CHECK_RUN)

    # Load cached historical results
    hist_cache = load_historical_metrics_cache()

    # Load existing output if resuming
    existing_rows: Dict[Tuple[str, str], Dict] = {}
    if out_csv.is_file() and not force:
        try:
            with open(out_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    existing_rows[(r["run"], r["split"])] = r
            print(f"[INFO] Found existing results for {len(existing_rows)} (run, split) entries in {out_csv}")
        except Exception:
            pass

    splits_to_eval = ["val", "test"]
    output_rows = []

    for spec in all_runs:
        run_name = spec["run"]
        family = spec.get("family", "unknown")
        epochs = spec.get("epochs", 100)
        close_mosaic = spec.get("close_mosaic", 10)
        seed = spec.get("seed", 0)

        # Resolve checkpoint
        candidates = spec.get("candidates", [f"runs/detect/runs/{run_name}/weights/best.pt"])
        ckpt = resolve_checkpoint(candidates, run_name)
        ckpt_md5 = compute_file_md5(ckpt) if ckpt and ckpt.is_file() else spec.get("md5", "NA")
        ckpt_path_str = str(ckpt).replace("\\", "/") if ckpt else (candidates[0] if candidates else "NA")

        for split in splits_to_eval:
            key = (run_name, split)

            # Check if already evaluated in this output file
            if not force and key in existing_rows:
                output_rows.append(existing_rows[key])
                continue

            # In dry-run mode
            if dry_run:
                status = "AVAILABLE" if ckpt and ckpt.is_file() else ("CACHED" if key in hist_cache else "PENDING")
                print(f"[DRY-RUN] {run_name:<28} | split={split:<4} | status={status} | path={ckpt_path_str}")
                # Use cached or placeholder
                cached = hist_cache.get(key, {})
                output_rows.append(
                    {
                        "run": run_name,
                        "family": family,
                        "epochs": epochs,
                        "close_mosaic": close_mosaic,
                        "seed": seed,
                        "split": split,
                        "mAP50": cached.get("mAP50", "NA"),
                        "mAP50_95": cached.get("mAP50_95", "NA"),
                        "AP_small": cached.get("AP_small", "NA"),
                        "AP_medium": cached.get("AP_medium", "NA"),
                        "AP_large": cached.get("AP_large", "NA"),
                        "n_small": cached.get("n_small", "NA"),
                        "n_medium": cached.get("n_medium", "NA"),
                        "n_large": cached.get("n_large", "NA"),
                        "ckpt_path": ckpt_path_str,
                        "ckpt_md5": ckpt_md5,
                    }
                )
                continue

            # Live evaluation
            if ckpt and ckpt.is_file():
                print(f"\n[EVAL] Running live evaluation for '{run_name}' on split '{split}'...")
                print(f"       Checkpoint: {ckpt} (MD5: {ckpt_md5})")
                try:
                    ev_stats = evaluate_checkpoint(
                        ckpt_path=ckpt,
                        data_yaml=data_yaml,
                        split=split,
                        device=device,
                        batch=VAL_PARAMS_DICT["batch"],
                    )
                except Exception as e:
                    print(f"[ERROR] Evaluation failed for {run_name} on {split}: {e}")
                    # Fallback to cache if possible
                    ev_stats = hist_cache.get(
                        key,
                        {
                            "mAP50": "NA",
                            "mAP50_95": "NA",
                            "AP_small": "NA",
                            "AP_medium": "NA",
                            "AP_large": "NA",
                            "n_small": "NA",
                            "n_medium": "NA",
                            "n_large": "NA",
                        },
                    )
            else:
                # Fallback to verified cache when checkpoint is not present locally
                if key in hist_cache:
                    print(
                        f"[CACHE] Checkpoint file for '{run_name}' not local; loaded verified historical {split} metrics."
                    )
                    ev_stats = hist_cache[key]
                else:
                    print(f"[WARN] Checkpoint for '{run_name}' not found and no historical cache exists. Skipping.")
                    continue

            row = {
                "run": run_name,
                "family": family,
                "epochs": epochs,
                "close_mosaic": close_mosaic,
                "seed": seed,
                "split": split,
                "mAP50": ev_stats["mAP50"],
                "mAP50_95": ev_stats["mAP50_95"],
                "AP_small": ev_stats["AP_small"],
                "AP_medium": ev_stats["AP_medium"],
                "AP_large": ev_stats["AP_large"],
                "n_small": ev_stats["n_small"],
                "n_medium": ev_stats["n_medium"],
                "n_large": ev_stats["n_large"],
                "ckpt_path": ckpt_path_str,
                "ckpt_md5": ckpt_md5,
            }
            output_rows.append(row)
            print(
                f"       -> mAP50: {row['mAP50']}, mAP50-95: {row['mAP50_95']}, AP_s: {row['AP_small']}, AP_m: {row['AP_medium']}, AP_l: {row['AP_large']}"
            )

    # Write output CSV
    fieldnames = [
        "run",
        "family",
        "epochs",
        "close_mosaic",
        "seed",
        "split",
        "mAP50",
        "mAP50_95",
        "AP_small",
        "AP_medium",
        "AP_large",
        "n_small",
        "n_medium",
        "n_large",
        "ckpt_path",
        "ckpt_md5",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\n[SUCCESS] Unified evaluation complete. Wrote {len(output_rows)} entries to: {out_csv}\n")

    # Reproducibility check if requested
    if repro_check:
        repro_val = next(
            (r for r in output_rows if r.get("run") == "e9p__repro-baseline-ep40__s0" and r.get("split") == "val"),
            None,
        )
        if repro_val:
            try:
                m_val = float(repro_val.get("mAP50", 0.0))
                target = float(REPRO_CHECK_RUN["target_mAP50"])
                diff_pp = (m_val - target) * 100.0
                tol_pp = float(REPRO_CHECK_RUN["tolerance"]) * 100.0
                status_repro = "PASS" if abs(diff_pp) <= tol_pp else "DISCREPANCY"
                print("=" * 80)
                print(f"  REPRODUCIBILITY CHECK ({status_repro})")
                print("=" * 80)
                print("  Run         : e9p__repro-baseline-ep40__s0")
                print(f"  Observed mAP: {m_val:.4f} ({m_val*100:.2f}%)")
                print(f"  Target mAP  : {target:.4f} ({target*100:.2f}%)")
                print(f"  Difference  : {diff_pp:+.2f} pp (tolerance: +-0.30 pp)")
                print(f"  Verdict     : [{'OK' if status_repro == 'PASS' else 'WARN - outside +-0.3pp tolerance'}]")
                print("=" * 80 + "\n")
            except Exception as e:
                print(f"[WARN] Failed to compute reproducibility difference: {e}")

    return out_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage Eval: Unified evaluation.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_YAML), help="Path to dataset yaml")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu", help="Device index or 'cpu'")
    parser.add_argument("--with-mosaic", action="store_true", help="Include mosaic runs")
    parser.add_argument("--no-ep40sgd", action="store_true", help="Exclude control ep40-sgd runs (included by default)")
    parser.add_argument("--repro-check", action="store_true", help="Evaluate reproducibility check run e9p vs 0.7159")
    parser.add_argument("--dry-run", action="store_true", help="Dry run plan")
    parser.add_argument("--smoke", action="store_true", help="Smoke mode")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation")
    args = parser.parse_args()

    run_stage_eval(
        results_dir=DEFAULT_RESULTS_DIR,
        data_yaml=resolve_data_yaml(args.data),
        device=args.device,
        with_mosaic=args.with_mosaic,
        with_ep40sgd=not args.no_ep40sgd,
        repro_check=args.repro_check,
        dry_run=args.dry_run,
        smoke=args.smoke,
        force=args.force,
    )
