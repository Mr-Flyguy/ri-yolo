# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Collect Phase 2 diagnostic metrics into C2_diag.csv."""

import argparse
import glob
import os
import re
import pandas as pd
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Collect Phase 2 diagnostics table.")
    parser.add_argument("--diag-dir", default="artifacts/phase2/diag", help="Directory with rho_*.csv")
    parser.add_argument("--out", default="tables/C2_diag.csv", help="Output table path")
    return parser.parse_args()


def resolve_ckpt(run_tag):
    if run_tag == "lam0":
        candidates = [
            "runs/detect/runs/e3p__pos-postsppf__s0/weights/best.pt",
            "runs/e3p__pos-postsppf__s0/weights/best.pt",
        ]
    else:
        candidates = [
            f"runs/detect/runs/{run_tag}/weights/best.pt",
            f"runs/{run_tag}/weights/best.pt",
        ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def main():
    args = parse_args()
    pattern = os.path.join(args.diag_dir, "rho_*.csv")
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        print(f"[WARN] No rho_*.csv files found matching {pattern}")
        return

    rows = []
    for f in csv_files:
        tag = os.path.basename(f)
        if tag.startswith("rho_"):
            tag = tag[4:]
        if tag.endswith(".csv"):
            tag = tag[:-4]

        d = pd.read_csv(f).iloc[0].to_dict()
        lam = 0.0 if tag == "lam0" else None
        m = re.search(r"lam-([\d.e-]+)", tag)
        if m:
            lam = float(m.group(1))

        ck = resolve_ckpt(tag)
        gamma = None
        if os.path.exists(ck):
            try:
                sd = torch.load(ck, map_location="cpu", weights_only=False)["model"].state_dict()
                g = [float(v.reshape(-1)[0]) for k, v in sd.items() if k.endswith("gamma")]
                if g:
                    gamma = round(g[0], 6)
            except Exception as e:
                print(f"[WARN] Failed to load gamma from {ck}: {e}")

        rows.append(
            {
                "run": tag,
                "lambda_tv": lam,
                "pearson": round(float(d.get("pearson", 0.0)), 4),
                "spearman": round(float(d.get("spearman", 0.0)), 4),
                "L_mean": round(float(d.get("L_mean", 0.0)), 4),
                "L_std": round(float(d.get("L_std", 0.0)), 4),
                "gamma_final": gamma,
            }
        )

    df = pd.DataFrame(rows).sort_values(["lambda_tv", "run"], na_position="last")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[SUCCESS] Saved {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
