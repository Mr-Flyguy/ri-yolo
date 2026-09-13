# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Extract and inspect gamma gating parameters from checkpoint state_dict.

Checks whether RFDBlock gamma parameters learned meaningful non-zero weights
or remained inert/collapsed during training.
"""

import argparse
import csv
import glob
import os
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Extract gamma gate parameters from model checkpoints.")
    parser.add_argument("--ckpt", nargs="+", required=True, help="Checkpoint paths or glob patterns (e.g. runs/*/weights/best.pt)")
    parser.add_argument("--out", default="artifacts/phase0/gamma_existing.csv", help="Output CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    rows = []
    matched_files = []
    for pat in args.ckpt:
        matched_files.extend(sorted(glob.glob(pat)))

    if not matched_files:
        print(f"[ERROR] No checkpoint files matched pattern: {args.ckpt}")
        return

    print(f"\n{'Checkpoint':<60} | {'Layer Key':<35} | {'Gamma':<10}")
    print("-" * 115)

    for f in matched_files:
        try:
            ck = torch.load(f, map_location="cpu", weights_only=False)
            model_obj = ck.get("model") or ck
            sd = model_obj.state_dict() if hasattr(model_obj, "state_dict") else model_obj

            found_gamma = False
            for k, v in sd.items():
                if k.endswith("gamma"):
                    g_val = float(v.reshape(-1)[0])
                    rows.append({"ckpt": f, "key": k, "gamma": g_val})
                    print(f"{f:<60} | {k:<35} | {g_val:+.6f}")
                    found_gamma = True

            if not found_gamma:
                print(f"{f:<60} | [NO GAMMA FOUND IN CHECKPOINT]")
        except Exception as e:
            print(f"{f:<60} | ERROR: {e}")

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ckpt", "key", "gamma"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n[INFO] Saved {len(rows)} gamma parameter entries to {args.out}")


if __name__ == "__main__":
    main()
