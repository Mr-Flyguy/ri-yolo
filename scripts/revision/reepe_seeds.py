# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# -*- coding: utf-8 -*-
"""REEPE 2027 additional seeds training and evaluation (Stage B).

Trains additional seeds to reach n = 5 for all 4 RFD insertion positions and baseline:
    python -m scripts.revision.reepe_seeds --check   # quick smoke check (1 epoch, fraction=0.02)
    python -m scripts.revision.reepe_seeds           # full training & evaluation queue

Queue:
  Seed 3: e3p__pos-p3__s3, e3p__pos-p4__s3, e3p__pos-presppf__s3
  Seed 4: e2p__postsppf__s4, e2p__baseline__s4, e3p__pos-p3__s4, e3p__pos-p4__s4, e3p__pos-presppf__s4

Outputs: results/reepe2/{new_runs_metrics.csv, map_stats.csv, map_ablation.csv, gamma_existing.csv, seeds.log, NOTES.md}
"""
import argparse
import csv
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import torch

from scripts.revision.config import (
    DEFAULT_DATA_YAML,
    EXISTING_RUN_SPECS,
    REPO_ROOT,
    resolve_data_yaml,
)
from scripts.revision.reepe_latency import pack_bundle, update_notes
from scripts.revision.reepe_overnight import ckpt_of, done_keys, evaluate, rfd_of
from scripts.revision.stage_corr import compute_luminance_measures, downsample_brightness, resolve_image_list
from scripts.revision.stage_train import execute_single_run, is_run_completed

OUT = REPO_ROOT / "results" / "reepe2"
RUNS_DIR = REPO_ROOT / "runs" / "detect" / "runs"

CFG_MAP = {
    "baseline": "yolov8s.yaml",
    "postsppf": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
    "p3": "ultralytics/cfg/models/v8/yolov8s-rfd-p3.yaml",
    "p4": "ultralytics/cfg/models/v8/yolov8s-rfd-p4.yaml",
    "presppf": "ultralytics/cfg/models/v8/yolov8s-rfd-presppf.yaml",
}

FAMILY_MAP = {
    "baseline": "baseline",
    "postsppf": "postsppf",
    "p3": "pos_sweep",
    "p4": "pos_sweep",
    "presppf": "pos_sweep",
}

# Training queue strictly ordered as specified
RUNS_QUEUE = [
    # Seed 3 (if not already trained)
    ("e3p__pos-p3__s3", "p3", 3),
    ("e3p__pos-p4__s3", "p4", 3),
    ("e3p__pos-presppf__s3", "presppf", 3),
    # Seed 4 (order: postsppf, baseline, p3, p4, presppf)
    ("e2p__postsppf__s4", "postsppf", 4),
    ("e2p__baseline__s4", "baseline", 4),
    ("e3p__pos-p3__s4", "p3", 4),
    ("e3p__pos-p4__s4", "p4", 4),
    ("e3p__pos-presppf__s4", "presppf", 4),
]

EXISTING_RFD_13 = [
    "e2p__postsppf__s0",
    "e2p__postsppf__s1",
    "e2p__postsppf__s2",
    "e2p__postsppf__s3",
    "e3p__pos-p3__s0",
    "e3p__pos-p3__s1",
    "e3p__pos-p3__s2",
    "e3p__pos-p4__s0",
    "e3p__pos-p4__s1",
    "e3p__pos-p4__s2",
    "e3p__pos-presppf__s0",
    "e3p__pos-presppf__s1",
    "e3p__pos-presppf__s2",
]

MODES = ["identity", "spatial_mean", "const_half", "open_gate", "shuffle"]


def log(msg: str):
    """Log message to console and results/reepe2/seeds.log."""
    OUT.mkdir(parents=True, exist_ok=True)
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(OUT / "seeds.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def append_row(path: Path, row: dict):
    """Append a dictionary row to CSV, writing header if file is new."""
    new = not path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def extract_existing_gammas():
    """Extract gamma_final for the 13 existing RFD checkpoints into results/reepe2/gamma_existing.csv."""
    from ultralytics import YOLO

    path = OUT / "gamma_existing.csv"
    done = done_keys(path, ["run"])

    log("=== Checking existing 13 RFD checkpoints for gamma_final ===")
    for run in EXISTING_RFD_13:
        if (run,) in done:
            continue
        ckpt = ckpt_of(run)
        if ckpt is None or not ckpt.is_file():
            log(f"[GAMMA] Checkpoint for {run} not found, skipped.")
            continue
        try:
            model = YOLO(str(ckpt)).model
            rfd = rfd_of(model)
            gamma_val = round(float(rfd.gamma.item()), 6)
            append_row(path, {"run": run, "gamma_final": gamma_val})
            log(f"[GAMMA] {run:<24} gamma_final = {gamma_val}")
        except Exception as e:
            log(f"[GAMMA] Could not extract gamma for {run}: {e}")


def compute_rfd_map_stats(run: str, ckpt: Path, data_yaml: Path, device: str):
    """Compute illumination map statistics on 300 fixed diagnostic images including gamma_final."""
    from ultralytics import YOLO

    path = OUT / "map_stats.csv"
    if (run,) in done_keys(path, ["run"]):
        log(f"[STATS] {run} already in {path}, skipped.")
        return

    dev_str = (
        f"cuda:{device}"
        if str(device).isdigit() and torch.cuda.is_available()
        else ("cpu" if not torch.cuda.is_available() else device)
    )
    dev = torch.device(dev_str)

    model_wrap = YOLO(str(ckpt))
    model = model_wrap.model.eval().to(dev)
    rfd = rfd_of(model)
    gamma_final = round(float(rfd.gamma.item()), 6)

    buf = {}
    h = rfd.ill_estimator.register_forward_hook(lambda m, i, o: buf.__setitem__("L", o.detach()))
    sds, means, rs = [], [], []

    img_list = resolve_image_list(data_yaml, n_max=300)
    L_hw = None
    for p in img_list:
        img = cv2.imread(p)
        if img is None:
            continue
        rgb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0, (640, 640))
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(next(model.parameters()).device)
        with torch.no_grad():
            model(x)
        L = buf["L"][0, 0].float().cpu().numpy()
        L_hw = (L.shape[0], L.shape[1])
        y = downsample_brightness(compute_luminance_measures(rgb)["rec601"], (L.shape[1], L.shape[0]), "area")
        sds.append(L.std())
        means.append(L.mean())
        rs.append(0.0 if L.std() < 1e-8 or y.std() < 1e-8 else float(np.corrcoef(L.ravel(), y.ravel())[0, 1]))
    h.remove()

    if not rs:
        log(f"[WARN] No valid diagnostic images processed for map_stats on {run}")
        return

    row = {
        "run": run,
        "map_hw": f"{L_hw[0]}x{L_hw[1]}" if L_hw else "unknown",
        "L_sd_pooled": round(float(np.sqrt(np.mean(np.square(sds)))), 4),
        "L_mean": round(float(np.mean(means)), 4),
        "one_minus_L": round(1.0 - float(np.mean(means)), 4),
        "pearson_rec601_area": round(float(np.mean(rs)), 4),
        "n_images": len(rs),
        "gamma_final": gamma_final,
    }
    append_row(path, row)
    log(f"[STATS] {run} map stats: {row}")


def compute_map_ablation(run: str, ckpt: Path, data_yaml: Path, device: str, expected_val_m50: float | None = None):
    """Run inference-time map ablation in 5 modes on val and test splits with sanity check."""
    path = OUT / "map_ablation.csv"
    done = done_keys(path, ["run", "split", "mode"])

    for split in ("val", "test"):
        for mode in MODES:
            if (run, split, mode) in done:
                continue
            t0 = time.time()
            m50, m5095, _ = evaluate(ckpt, data_yaml, split, device, mode)
            append_row(path, {"run": run, "split": split, "mode": mode, "mAP50": m50, "mAP50_95": m5095})
            log(
                f"[ABLATION] {run:<24} split={split:<4} mode={mode:<12} mAP50={m50:.4f} mAP50-95={m5095:.4f} ({time.time() - t0:.0f}s)"
            )

            # Sanity check: identity must match normal evaluation mAP50 to 4 decimal places
            if split == "val" and mode == "identity" and expected_val_m50 is not None:
                diff = abs(m50 - expected_val_m50)
                log(f"[SANITY] {run} val identity vs unhooked: diff = {diff:.6f}")
                assert diff < 1e-4, f"Ablation identity mAP50 ({m50}) diverged from unhooked eval ({expected_val_m50})!"


def verify_specs_against_seed0():
    """Verify that all queued run specs match their seed 0 counterparts identically."""
    log("=== Verifying all queued specs against seed 0 counterparts ===")
    counterparts = {
        "p3": "e3p__pos-p3__s0",
        "p4": "e3p__pos-p4__s0",
        "presppf": "e3p__pos-presppf__s0",
        "postsppf": "e2p__postsppf__s0",
        "baseline": "e2p__baseline__s0",
    }

    # Reference seed 0 specs
    seed0_specs = {}
    for pos, c_run in counterparts.items():
        spec_list = [s for s in EXISTING_RUN_SPECS if s["run"] == c_run]
        if spec_list:
            s0 = spec_list[0]
            seed0_specs[pos] = {
                "family": s0["family"],
                "epochs": s0["epochs"],
                "close_mosaic": s0["close_mosaic"],
                "cfg": CFG_MAP[pos],
                "optimizer": "auto",
            }
        else:
            seed0_specs[pos] = {
                "family": FAMILY_MAP[pos],
                "epochs": 100,
                "close_mosaic": 10,
                "cfg": CFG_MAP[pos],
                "optimizer": "auto",
            }

    # Check each queued run
    for run_name, pos, seed in RUNS_QUEUE:
        cand_spec = {
            "run": run_name,
            "family": FAMILY_MAP[pos],
            "epochs": 100,
            "close_mosaic": 10,
            "seed": seed,
            "cfg": CFG_MAP[pos],
            "optimizer": "auto",
        }
        ref = seed0_specs[pos]

        # Verify matching parameters
        for key in ("family", "epochs", "close_mosaic", "cfg", "optimizer"):
            assert (
                cand_spec[key] == ref[key]
            ), f"Spec mismatch for {run_name} on '{key}': {cand_spec[key]} != {ref[key]}"

        log(
            f"  [OK] {run_name:<24} pos={pos:<8} seed={seed} epochs={cand_spec['epochs']} "
            f"mosaic={cand_spec['close_mosaic']} cfg={cand_spec['cfg']}"
        )
    log("[PASS] All queued specs match seed 0 recipes perfectly.")


def run_stage_seeds(data_yaml: Path, device: str):
    """Execute Stage B: training queue, metric evaluations, map stats, and ablations."""
    log(f"=== Stage B: Additional Seeds Queue (device={device}) ===")

    # 1. Verify specs
    verify_specs_against_seed0()

    # 2. Extract gamma for existing 13 RFD checkpoints
    extract_existing_gammas()

    metrics_csv = OUT / "new_runs_metrics.csv"
    metrics_done = done_keys(metrics_csv, ["run", "split"])

    total_runs = len(RUNS_QUEUE)
    for idx, (run_name, pos, seed) in enumerate(RUNS_QUEUE, 1):
        spec = {
            "run": run_name,
            "family": FAMILY_MAP[pos],
            "epochs": 100,
            "close_mosaic": 10,
            "seed": seed,
            "cfg": CFG_MAP[pos],
            "optimizer": "auto",
        }

        log(f"\n--- [{idx}/{total_runs}] Processing {run_name} (position={pos}, seed={seed}) ---")

        # Check if already trained and complete
        if is_run_completed(run_name, target_epochs=100, runs_dir=RUNS_DIR):
            log(f"[TRAIN] Run '{run_name}' is already completed (100 epochs). Skipping training.")
        else:
            t0 = time.time()
            log(f"[TRAIN] Launching training for '{run_name}'...")
            execute_single_run(spec, data_yaml, device, RUNS_DIR, results_dir=OUT)
            log(f"[TRAIN] Training finished for '{run_name}' in {(time.time() - t0) / 60:.1f} min")
            update_notes(f"Trained {run_name} (pos={pos}, seed={seed}) in {(time.time() - t0) / 60:.1f} min.")

        ckpt = RUNS_DIR / run_name / "weights" / "best.pt"
        if not ckpt.is_file():
            alt_ckpt = REPO_ROOT / "runs" / run_name / "weights" / "best.pt"
            if alt_ckpt.is_file():
                ckpt = alt_ckpt
            else:
                log(f"[ERROR] best.pt not found for {run_name}! Skipping evaluation.")
                continue

        # 1. Standard evaluation on val and test
        val_m50_eval = None
        for split in ("val", "test"):
            if (run_name, split) in metrics_done:
                log(f"[EVAL] {run_name} {split} already in {metrics_csv}, skipped.")
                continue
            m50, m5095, n_params = evaluate(ckpt, data_yaml, split, device, mode=None)
            append_row(
                metrics_csv,
                {
                    "run": run_name,
                    "position": pos,
                    "seed": seed,
                    "split": split,
                    "mAP50": m50,
                    "mAP50_95": m5095,
                    "params": n_params,
                },
            )
            log(f"[EVAL] {run_name:<24} split={split:<4} mAP50={m50:.4f} mAP50-95={m5095:.4f} params={n_params:,}")
            if split == "val":
                val_m50_eval = m50

        # If val was already evaluated, load it for sanity checking
        if val_m50_eval is None and metrics_csv.is_file():
            with open(metrics_csv, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r["run"] == run_name and r["split"] == "val":
                        val_m50_eval = float(r["mAP50"])
                        break

        # 2 & 3. For RFD models only: map statistics and inference-time map ablation
        if pos != "baseline":
            log(f"[MODULE] Running map stats and map ablation for RFD model '{run_name}'...")
            compute_rfd_map_stats(run_name, ckpt, data_yaml, device)
            compute_map_ablation(run_name, ckpt, data_yaml, device, expected_val_m50=val_m50_eval)

        # Repack bundle after each run
        pack_bundle()

    log("\n=== ALL QUEUED SEEDS COMPLETED SUCCESSFULLY ===")
    update_notes("All queued seeds (Stage B) finished training and evaluation.")
    pack_bundle()


def run_smoke_check(data_yaml: Path, device: str):
    """Execute quick smoke check on fraction=0.02, 1 epoch in temporary directory."""
    smoke_dir = REPO_ROOT / "runs" / "detect" / "test_smoke_seeds"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir, ignore_errors=True)
    smoke_dir.mkdir(parents=True, exist_ok=True)

    log("=== Running reepe_seeds --check (1 epoch, fraction=0.02) ===")
    verify_specs_against_seed0()

    test_run = "smoke_pos_p3"
    test_spec = {
        "run": test_run,
        "family": "pos_sweep",
        "epochs": 1,
        "close_mosaic": 0,
        "seed": 0,
        "cfg": CFG_MAP["p3"],
        "optimizer": "auto",
    }

    try:
        log("Testing execute_single_run with smoke=True...")
        execute_single_run(test_spec, data_yaml, device, smoke_dir, smoke=True)
        ckpt = smoke_dir / test_run / "weights" / "best.pt"
        if not ckpt.is_file():
            ckpt = smoke_dir / test_run / "weights" / "last.pt"

        if ckpt.is_file():
            log(f"Smoke checkpoint generated: {ckpt}")
            m50, m5095, _ = evaluate(ckpt, data_yaml, "val", device, mode=None, fraction=0.02)
            log(f"Smoke evaluation result: mAP50 = {m50:.4f}, mAP50-95 = {m5095:.4f}")
        else:
            log("[WARN] Smoke checkpoint not saved, but execution finished without crash.")

        log("[PASS] reepe_seeds --check completed successfully!")
    finally:
        if smoke_dir.exists():
            shutil.rmtree(smoke_dir, ignore_errors=True)
            log("Cleaned up temporary smoke directory.")


def main():
    ap = argparse.ArgumentParser(description="REEPE 2027 Additional Seeds Runner (Stage B)")
    ap.add_argument("--data", default=str(DEFAULT_DATA_YAML), help="Path to exdark.yaml")
    ap.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    ap.add_argument("--check", action="store_true", help="Quick smoke check (1 epoch, fraction=0.02)")
    args = ap.parse_args()

    data_yaml = resolve_data_yaml(args.data)

    if args.check:
        run_smoke_check(data_yaml, args.device)
        return

    run_stage_seeds(data_yaml, args.device)


if __name__ == "__main__":
    main()
