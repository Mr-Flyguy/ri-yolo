# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Phase 3 Data Aggregator: Component Ablation, Efficiency, and Training Regime.

Generates tables and figures for Article C3 (IEEE / Scopus Q1):
- Table 1: Component Ablation (Baseline -> RFD Module -> TV-loss -> NWD Metric vs Capacity Control)
- Table 2: Efficiency and Computational Complexity (from Phase 0 benchmark)
- Table 3: Training Regime and Sample Efficiency (Epochs & Mosaic duration sensitivity)
- Figure 3: Regime Curves (Duration & Clean Epochs %)
"""

import argparse
import csv
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Collect and generate Tables and Figures for Paper C3")
    parser.add_argument(
        "--p1-summary",
        type=str,
        default="artifacts/phase1/summary.csv",
        help="Path to Phase 1 summary.csv",
    )
    parser.add_argument(
        "--p2-summary",
        type=str,
        default="artifacts/phase2/summary.csv",
        help="Path to Phase 2 summary.csv",
    )
    parser.add_argument(
        "--p3-summary",
        type=str,
        default="artifacts/phase3/summary.csv",
        help="Path to Phase 3 summary.csv",
    )
    parser.add_argument(
        "--bench",
        type=str,
        default="artifacts/phase0/bench_gpu_rtx3070ti.csv",
        help="Path to Phase 0 benchmark CSV",
    )
    parser.add_argument(
        "--out-t1",
        type=str,
        default="artifacts/phase3/C3_table1.csv",
        help="Output path for Table 1 (Component Ablation)",
    )
    parser.add_argument(
        "--out-t2",
        type=str,
        default="artifacts/phase3/C3_table2.csv",
        help="Output path for Table 2 (Efficiency)",
    )
    parser.add_argument(
        "--out-regime",
        type=str,
        default="artifacts/phase3/C3_regime.csv",
        help="Output path for Regime Table",
    )
    parser.add_argument(
        "--out-fig",
        type=str,
        default="figs/C3_fig3_regime.png",
        help="Output path for Figure 3 (Regime Plot)",
    )
    return parser.parse_args()


def load_csv_rows(path_str):
    p = Path(path_str)
    if not p.exists():
        return []
    with open(p, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_table1(p1_rows, p2_rows, out_path):
    """Build Table 1: Component Ablation."""
    print("\n--- Building Table 1: Component Ablation ---")

    # Group specifications
    specs = [
        {
            "config": "Baseline (YOLOv8s)",
            "sources": [r for r in p1_rows if r.get("group") == "E2" and "baseline" in r.get("run", "")],
        },
        {
            "config": "+ Spatial Retinex Decoupling (RFD post-SPPF)",
            "sources": [r for r in p1_rows if "postsppf" in r.get("run", "")],
        },
        {
            "config": "+ Retinex Smoothness Loss (RSL, lambda=0.01)",
            "sources": [r for r in p2_rows if r.get("run") == "e6p__lam-1e-2__s0"],
        },
        {
            "config": "+ Physical Bounding Box Metric (NWD abs, C=10.24)",
            "sources": [r for r in p2_rows if r.get("run") == "e7p__nwd-calib__s0"],
        },
        {
            "config": "Capacity Control (RFDBlockNoGate, 2.6M params)",
            "sources": [r for r in p1_rows if "capctrl" in r.get("run", "")],
        },
    ]

    out_rows = []
    base_m50 = None
    base_m95 = None

    for spec in specs:
        items = spec["sources"]
        if not items:
            print(f"[WARN] No records found for '{spec['config']}'")
            continue

        m50_vals = [float(x["mAP50"]) for x in items]
        m95_vals = [float(x["mAP50_95"]) for x in items]
        gammas = [
            float(x["gamma_final"])
            for x in items
            if x.get("gamma_final") and str(x["gamma_final"]).strip() not in ("NA", "None", "")
        ]

        n = len(items)
        m50_mean = float(np.mean(m50_vals))
        m50_sd = float(np.std(m50_vals, ddof=1)) if n > 1 else None
        ci95 = float(1.96 * m50_sd / np.sqrt(n)) if n > 1 and m50_sd is not None else None

        m95_mean = float(np.mean(m95_vals))
        m95_sd = float(np.std(m95_vals, ddof=1)) if n > 1 else None

        gamma_mean = float(np.mean(gammas)) if gammas else None

        if base_m50 is None:
            base_m50 = m50_mean
            base_m95 = m95_mean

        delta_m50 = 100.0 * (m50_mean - base_m50)
        delta_m95 = 100.0 * (m95_mean - base_m95)

        out_rows.append(
            {
                "config": spec["config"],
                "n_runs": n,
                "mAP50": round(m50_mean, 4),
                "mAP50_sd": round(m50_sd, 4) if m50_sd is not None else "",
                "mAP50_ci95": round(ci95, 4) if ci95 is not None else "",
                "mAP50_95": round(m95_mean, 4),
                "mAP50_95_sd": round(m95_sd, 4) if m95_sd is not None else "",
                "gamma_final": f"{gamma_mean:+.6f}" if gamma_mean is not None else "NA",
                "delta_mAP50_pp": f"{delta_m50:+.2f}",
                "delta_mAP50_95_pp": f"{delta_m95:+.2f}",
            }
        )

    if out_rows:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"[SUCCESS] Table 1 saved to {out_p} ({len(out_rows)} rows)")

        # Also mirror to tables/
        tables_p = Path("tables/C3_table1.csv")
        tables_p.parent.mkdir(parents=True, exist_ok=True)
        with open(tables_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)

        df = pd.DataFrame(out_rows)
        print(df.to_string(index=False))


def build_table2(bench_path, out_path):
    """Build Table 2: Efficiency and Computational Complexity."""
    print("\n--- Building Table 2: Efficiency and Latency ---")
    bp = Path(bench_path)
    if not bp.exists():
        print(f"[WARN] Benchmark file {bp} not found. Skipping Table 2.")
        return

    bdf = pd.read_csv(bp)
    target_models = ["yolov8s.yaml", "yolov8s-rfd-postsppf.yaml", "yolov8s-capctrl.yaml"]

    # Filter for batch=1, half=0 (FP32 single-image latency)
    t2_df = bdf[(bdf["batch"] == 1) & (bdf["half"] == 0) & (bdf["model"].isin(target_models))].copy()

    if t2_df.empty:
        print("[WARN] No matching records found in benchmark CSV.")
        return

    base_lat = float(t2_df[t2_df["model"] == "yolov8s.yaml"]["lat_med_ms"].iloc[0])

    t2_df["overhead_ms"] = (t2_df["lat_med_ms"] - base_lat).round(2)
    t2_df["overhead_pct"] = ((t2_df["lat_med_ms"] - base_lat) / base_lat * 100.0).round(1)

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    t2_df.to_csv(out_p, index=False)
    print(f"[SUCCESS] Table 2 saved to {out_p}")

    # Mirror to tables/
    tables_p = Path("tables/C3_table2.csv")
    tables_p.parent.mkdir(parents=True, exist_ok=True)
    t2_df.to_csv(tables_p, index=False)
    print(
        t2_df[["model", "params_M", "gflops", "lat_med_ms", "fps", "overhead_ms", "overhead_pct"]].to_string(
            index=False
        )
    )


def build_regime_and_plot(p1_rows, p3_rows, out_csv, out_fig):
    """Build Regime Table and Figure 3 (Epochs & Mosaic ablation)."""
    print("\n--- Building Regime Analysis (Figure 3 & Regime Table) ---")
    if not p3_rows:
        print(
            "[INFO] Phase 3 summary.csv is empty or not yet created. Skipping regime plot until Phase 3 runs complete."
        )
        return

    # 1. Epochs Panel (Sample Efficiency: 40, 60, 100 epochs)
    ep_rows = []

    # E5 runs from Phase 3
    for r in p3_rows:
        if r.get("group") == "E5":
            fam = "baseline" if "baseline" in r.get("run", "") else "postsppf"
            ep_rows.append(
                {
                    "panel": "epochs",
                    "family": fam,
                    "run": r["run"],
                    "epochs": int(r["epochs"]),
                    "close_mosaic": int(r.get("close_mosaic", 10)),
                    "clean_pct": float(r.get("clean_epochs_pct", 0.0)),
                    "mAP50": float(r["mAP50"]),
                    "mAP50_95": float(r["mAP50_95"]),
                }
            )

    # 100-epoch anchors from Phase 1
    base_100_runs = [r for r in p1_rows if r.get("group") == "E2" and "baseline" in r.get("run", "")]
    if base_100_runs:
        ep_rows.append(
            {
                "panel": "epochs",
                "family": "baseline",
                "run": "e2p__baseline__mean100",
                "epochs": 100,
                "close_mosaic": 10,
                "clean_pct": 10.0,
                "mAP50": float(np.mean([float(x["mAP50"]) for x in base_100_runs])),
                "mAP50_95": float(np.mean([float(x["mAP50_95"]) for x in base_100_runs])),
            }
        )

    rfd_100_runs = [r for r in p1_rows if "postsppf" in r.get("run", "")]
    if rfd_100_runs:
        ep_rows.append(
            {
                "panel": "epochs",
                "family": "postsppf",
                "run": "e3p_e2p__postsppf__mean100",
                "epochs": 100,
                "close_mosaic": 10,
                "clean_pct": 10.0,
                "mAP50": float(np.mean([float(x["mAP50"]) for x in rfd_100_runs])),
                "mAP50_95": float(np.mean([float(x["mAP50_95"]) for x in rfd_100_runs])),
            }
        )

    # 2. Mosaic Panel (Mosaic sensitivity: 0%, 10%, 50% clean epochs)
    mosaic_rows = []
    for r in p3_rows:
        if r.get("group") == "E4":
            mosaic_rows.append(
                {
                    "panel": "mosaic",
                    "family": "postsppf",
                    "run": r["run"],
                    "epochs": int(r["epochs"]),
                    "close_mosaic": int(r.get("close_mosaic", 0)),
                    "clean_pct": float(r.get("clean_epochs_pct", 0.0)),
                    "mAP50": float(r["mAP50"]),
                    "mAP50_95": float(r["mAP50_95"]),
                }
            )

    # Add 10% clean mosaic anchor from Phase 1 postsppf
    postsppf_s0 = [r for r in p1_rows if r.get("run") == "e3p__pos-postsppf__s0"]
    if postsppf_s0:
        mosaic_rows.append(
            {
                "panel": "mosaic",
                "family": "postsppf",
                "run": "e3p__pos-postsppf__s0",
                "epochs": 100,
                "close_mosaic": 10,
                "clean_pct": 10.0,
                "mAP50": float(postsppf_s0[0]["mAP50"]),
                "mAP50_95": float(postsppf_s0[0]["mAP50_95"]),
            }
        )

    all_regime = ep_rows + mosaic_rows
    if not all_regime:
        return

    regime_df = pd.DataFrame(all_regime)
    out_csv_p = Path(out_csv)
    out_csv_p.parent.mkdir(parents=True, exist_ok=True)
    regime_df.to_csv(out_csv_p, index=False)
    print(f"[SUCCESS] Regime data written to {out_csv_p}")

    # Mirror to tables/
    tables_p = Path("tables/C3_regime.csv")
    tables_p.parent.mkdir(parents=True, exist_ok=True)
    regime_df.to_csv(tables_p, index=False)

    # Plot Figure 3
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)

        # Panel A: Epochs
        ep_df = pd.DataFrame(ep_rows)
        if not ep_df.empty:
            for fam, mark, col, lbl in [
                ("baseline", "o", "tab:gray", "YOLOv8s baseline"),
                ("postsppf", "s", "tab:blue", "RI-YOLO (RFD post-SPPF)"),
            ]:
                sub = ep_df[ep_df["family"] == fam].sort_values("epochs")
                if not sub.empty:
                    ax[0].plot(
                        sub["epochs"], sub["mAP50"], marker=mark, color=col, linewidth=2, markersize=8, label=lbl
                    )

            ax[0].set_xlabel("Training Epochs", fontsize=11)
            ax[0].set_ylabel("mAP@0.5", fontsize=11)
            ax[0].set_title("(a) Convergence & Sample Efficiency", fontsize=12, fontweight="bold")
            ax[0].grid(alpha=0.3, linestyle="--")
            ax[0].legend(frameon=True)

        # Panel B: Mosaic Clean Epochs %
        mo_df = pd.DataFrame(mosaic_rows)
        if not mo_df.empty:
            mo_df = mo_df.sort_values("clean_pct")
            ax[1].plot(
                mo_df["clean_pct"],
                mo_df["mAP50"],
                marker="^",
                color="tab:green",
                linewidth=2,
                markersize=8,
                label="RI-YOLO (100 epochs)",
            )
            ax[1].set_xlabel("Epochs without Mosaic (% of schedule)", fontsize=11)
            ax[1].set_ylabel("mAP@0.5", fontsize=11)
            ax[1].set_title("(b) Sensitivity to Mosaic Augmentation", fontsize=12, fontweight="bold")
            ax[1].grid(alpha=0.3, linestyle="--")
            ax[1].legend(frameon=True)

        plt.tight_layout()
        out_fig_p = Path(out_fig)
        out_fig_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_fig_p)
        plt.close()
        print(f"[SUCCESS] Figure 3 saved to {out_fig_p}")

    except Exception as e:
        print(f"[WARN] Error generating plot: {e}")


def main():
    args = parse_args()

    p1_rows = load_csv_rows(args.p1_summary)
    p2_rows = load_csv_rows(args.p2_summary)
    p3_rows = load_csv_rows(args.p3_summary)

    print(f"[INFO] Loaded {len(p1_rows)} P1 rows, {len(p2_rows)} P2 rows, {len(p3_rows)} P3 rows.")

    build_table1(p1_rows, p2_rows, args.out_t1)
    build_table2(args.bench, args.out_t2)
    build_regime_and_plot(p1_rows, p3_rows, args.out_regime, args.out_fig)

    print("\n[ALL DONE] Phase 3 data collection finished.")


if __name__ == "__main__":
    main()
