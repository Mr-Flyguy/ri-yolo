# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Post-run verification and diagnostics extraction."""

import argparse
import json
import os
import sys
import pandas as pd
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Run post-training sanity checks.")
    parser.add_argument("--run", type=str, required=True, help="Path to run directory (e.g., runs/e2p__baseline__s0)")
    return parser.parse_args()


def main():
    args = parse_args()
    r = args.run
    results_csv = os.path.join(r, "results.csv")
    if not os.path.exists(results_csv):
        # Try alternate run directories (e.g., runs/detect/runs/<name> vs runs/<name>)
        bname = os.path.basename(os.path.normpath(r))
        alts = [
            os.path.join("runs", "detect", "runs", bname),
            os.path.join("runs", bname),
        ]
        for alt in alts:
            if os.path.exists(os.path.join(alt, "results.csv")):
                r = alt
                results_csv = os.path.join(r, "results.csv")
                break

    if not os.path.exists(results_csv):
        print(f"[ERROR] results.csv not found in {r}", file=sys.stderr)
        sys.exit(1)

    d = pd.read_csv(results_csv)
    d.columns = [c.strip() for c in d.columns]
    m50, m5095 = "metrics/mAP50(B)", "metrics/mAP50-95(B)"
    if m5095 not in d.columns or m50 not in d.columns:
        print(f"[ERROR] Required metric columns missing in {results_csv}", file=sys.stderr)
        sys.exit(1)

    best_i = d[m5095].idxmax()
    out = {
        "run": os.path.basename(os.path.normpath(r)),
        "epochs": int(d["epoch"].max()),
        "best_epoch": int(d["epoch"].iloc[best_i]),
        "best_mAP50": float(d[m50].iloc[best_i]),
        "best_mAP50_95": float(d[m5095].iloc[best_i]),
        "last_mAP50": float(d[m50].iloc[-1]),
        "last_mAP50_95": float(d[m5095].iloc[-1]),
        "train_box_first": float(d["train/box_loss"].iloc[0]),
        "train_box_last": float(d["train/box_loss"].iloc[-1]),
        "val_box_min": float(d["val/box_loss"].min()),
        "val_box_last": float(d["val/box_loss"].iloc[-1]),
    }
    ck = os.path.join(r, "weights", "best.pt")
    if os.path.exists(ck):
        try:
            sd = torch.load(ck, map_location="cpu", weights_only=False)["model"].state_dict()
            g = [float(v.reshape(-1)[0]) for k, v in sd.items() if k.endswith("gamma")]
            out["gamma"] = g[0] if g else None
        except Exception as e:
            out["gamma_error"] = str(e)

    rl = os.path.join(r, "rfd_log.csv")
    if os.path.exists(rl):
        try:
            rd = pd.read_csv(rl)
            out["L_std_final"] = float(rd["L_std_spatial"].iloc[-1]) if "L_std_spatial" in rd else None
            out["L_sat_final"] = float(rd["L_sat_frac"].iloc[-1]) if "L_sat_frac" in rd else None
        except Exception as e:
            out["rfd_log_error"] = str(e)

    # Sanity checks
    if out["train_box_last"] >= out["train_box_first"]:
        print(f"[WARN] train loss did not decrease ({out['train_box_first']} -> {out['train_box_last']})")
    if out["best_mAP50"] <= 0.50:
        print(f"[WARN] suspiciously low mAP50: {out['best_mAP50']}")

    summary_json_path = os.path.join(r, "summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
