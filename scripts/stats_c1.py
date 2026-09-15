# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Computes statistical comparison metrics for Paper C1 (RSCI)."""

import argparse
from math import comb
import os
import numpy as np
import pandas as pd
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate C1 statistics.")
    parser.add_argument("--summary", default="artifacts/phase1/summary.csv", help="Summary CSV path")
    parser.add_argument("--out", default="tables/C1_stats.csv", help="Output stats CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    d = pd.read_csv(args.summary)
    d.columns = [c.strip() for c in d.columns]
    base = d[d.run.str.contains("baseline")]
    post = d[d.run.str.contains("postsppf")]
    rows = []
    for col in ["mAP50", "mAP50_95"]:
        x, y = base[col].astype(float).values, post[col].astype(float).values
        vx, vy = x.var(ddof=1), y.var(ddof=1)
        dfw = (vy / len(y) + vx / len(x)) ** 2 / (
            (vy / len(y)) ** 2 / (len(y) - 1) + (vx / len(x)) ** 2 / (len(x) - 1)
        )
        t, pv = stats.ttest_ind(y, x, equal_var=False)
        u, pu = stats.mannwhitneyu(y, x, alternative="two-sided")
        se = np.sqrt(vx / len(x) + vy / len(y))
        tc = stats.t.ppf(0.975, dfw)
        diff = y.mean() - x.mean()
        sp = np.sqrt(((len(x) - 1) * vx + (len(y) - 1) * vy) / (len(x) + len(y) - 2))
        rows.append(
            {
                "metric": col,
                "n_base": len(x),
                "n_rfd": len(y),
                "base_mean": round(float(x.mean()), 4),
                "base_sd": round(float(x.std(ddof=1)), 4),
                "rfd_mean": round(float(y.mean()), 4),
                "rfd_sd": round(float(y.std(ddof=1)), 4),
                "delta_pp": round(float(100 * diff), 3),
                "welch_t": round(float(t), 3),
                "welch_p": round(float(pv), 4),
                "df": round(float(dfw), 2),
                "ci95_lo_pp": round(float(100 * (diff - tc * se)), 3),
                "ci95_hi_pp": round(float(100 * (diff + tc * se)), 3),
                "mw_U": float(u),
                "mw_p": round(float(pu), 4),
                "mw_p_min_possible": round(float(2 / comb(len(x) + len(y), len(x))), 4),
                "cohen_d": round(float(diff / sp), 2),
            }
        )
    out = pd.DataFrame(rows)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
