# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Stage Stats: Rigorous Statistical Analysis for Reviewers R1.1 and R1.2.

Executes complete statistical testing on both val and test splits:
- Comparisons:
  1. module vs base (at 100 epochs)
  2. capctrl vs base (at 100 epochs)
  3. capctrl vs module (at 100 epochs)
  4. ep40 vs ep100 for baseline
  5. ep40 vs ep100 for module (postsppf)
  6. ep60 vs ep100 for baseline
  7. ep60 vs ep100 for module (postsppf)
- Metrics: difference in pp, Welch's t (df, p, 95% CI), Mann-Whitney U, Cohen's d
- Equivalence testing: TOST with bounds +-0.5 pp and +-1.0 pp, 90% CI, and minimal equivalence bound
- Schedule vs Module effect ratio with 10,000 bootstrap resamples
- Control verification check (+0.09 pp, p=0.82, 95% CI [-0.92, +1.10]; +0.59 pp, p=0.092)
- Outputs results/revision/stats_val.csv and results/revision/stats_test.csv
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, t as t_dist

# Ensure repo root is in sys.path when executed directly
REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from scripts.revision.config import DEFAULT_RESULTS_DIR, REPO_ROOT  # noqa: E402


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cohen's d effect size between two groups."""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    if dof <= 0:
        return 0.0
    s_pooled = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    if s_pooled < 1e-12:
        return 0.0
    return float((np.mean(y) - np.mean(x)) / s_pooled)


def compute_welch_stats(x: np.ndarray, y: np.ndarray) -> Dict:
    """Compute Welch's t-test with Welch-Satterthwaite degrees of freedom and 95% CI."""
    nx, ny = len(x), len(y)
    mean_x, mean_y = float(np.mean(x)), float(np.mean(y))
    diff = mean_y - mean_x
    diff_pp = diff * 100.0

    if nx <= 1 or ny <= 1:
        return {
            "mean_g1": mean_x,
            "mean_g2": mean_y,
            "diff_pp": round(diff_pp, 4),
            "welch_t": "NA",
            "welch_df": "NA",
            "welch_p": "NA",
            "ci95_lower_pp": "NA",
            "ci95_upper_pp": "NA",
            "mwu_U": "NA",
            "mwu_p": "NA",
            "cohens_d": "NA",
        }

    var_x = float(np.var(x, ddof=1))
    var_y = float(np.var(y, ddof=1))
    se_sq = var_x / nx + var_y / ny
    se = float(np.sqrt(se_sq))

    t_stat = diff / se if se > 1e-12 else 0.0

    # Welch-Satterthwaite degrees of freedom
    df_num = se_sq**2
    df_den = (var_x / nx) ** 2 / (nx - 1) + (var_y / ny) ** 2 / (ny - 1)
    df = df_num / df_den if df_den > 0 else 1.0

    # Two-sided p-value
    p_val = 2.0 * (1.0 - t_dist.cdf(abs(t_stat), df=df))

    # 95% CI
    t_crit = float(t_dist.ppf(0.975, df=df))
    margin_pp = (t_crit * se) * 100.0
    ci_lower_pp = diff_pp - margin_pp
    ci_upper_pp = diff_pp + margin_pp

    # Mann-Whitney U test
    try:
        u_stat, mwu_p = mannwhitneyu(x, y, alternative="two-sided")
    except Exception:
        u_stat, mwu_p = "NA", "NA"

    # Cohen's d
    d_val = cohens_d(x, y)

    return {
        "mean_g1": round(mean_x, 4),
        "mean_g2": round(mean_y, 4),
        "diff_pp": round(diff_pp, 4),
        "welch_t": round(float(t_stat), 4),
        "welch_df": round(float(df), 2),
        "welch_p": round(float(p_val), 4),
        "ci95_lower_pp": round(float(ci_lower_pp), 4),
        "ci95_upper_pp": round(float(ci_upper_pp), 4),
        "mwu_U": float(u_stat) if u_stat != "NA" else "NA",
        "mwu_p": round(float(mwu_p), 4) if mwu_p != "NA" else "NA",
        "cohens_d": round(float(d_val), 4),
        "_se": se,
        "_df": df,
        "_diff_pp": diff_pp,
    }


def compute_tost_equivalence(diff_pp: float, se_pp: float, df: float, bounds: List[float] = [0.5, 1.0]) -> Dict:
    """Compute Two One-Sided Tests (TOST) for equivalence and minimal equivalence bound."""
    res = {}
    if df <= 0 or se_pp < 1e-12:
        for b in bounds:
            res[f"tost_p_bound_{b}pp"] = "NA"
        res["tost_ci90_lower_pp"] = "NA"
        res["tost_ci90_upper_pp"] = "NA"
        res["min_equiv_bound_pp"] = "NA"
        return res

    # 90% CI for equivalence (corresponds to two one-sided tests at alpha=0.05)
    t_crit_90 = float(t_dist.ppf(0.95, df=df))
    margin_90 = t_crit_90 * se_pp
    ci90_lower = diff_pp - margin_90
    ci90_upper = diff_pp + margin_90

    res["tost_ci90_lower_pp"] = round(ci90_lower, 4)
    res["tost_ci90_upper_pp"] = round(ci90_upper, 4)

    # For each bound delta
    for delta in bounds:
        # t_lower: test H01: diff <= -delta
        t1 = (diff_pp - (-delta)) / se_pp
        p1 = 1.0 - t_dist.cdf(t1, df=df)
        # t_upper: test H02: diff >= +delta
        t2 = (diff_pp - delta) / se_pp
        p2 = t_dist.cdf(t2, df=df)
        p_tost = max(p1, p2)
        res[f"tost_p_bound_{delta}pp"] = round(float(p_tost), 4)

    # Minimal bound delta_min where 90% CI is fully inside [-delta_min, +delta_min]
    delta_min = abs(diff_pp) + margin_90
    res["min_equiv_bound_pp"] = round(float(delta_min), 4)

    return res


def compute_schedule_vs_module_ratio(
    sched_series: np.ndarray,
    base_ref: float,
    mod_100: np.ndarray,
    base_100: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Calculate ratio of schedule effect (relative to base_ref) to module effect with 10,000 bootstrap resamples."""
    np.random.seed(seed)
    if len(sched_series) == 0 or len(mod_100) == 0 or len(base_100) == 0:
        return {
            "sched_diff_pp": "NA",
            "mod_diff_pp": "NA",
            "ratio_point": "NA",
            "bootstrap_ci95_lower": "NA",
            "bootstrap_ci95_upper": "NA",
        }

    diff_sched = (float(np.mean(sched_series)) - float(base_ref)) * 100.0
    diff_mod = (float(np.mean(mod_100)) - float(np.mean(base_100))) * 100.0
    ratio_point = diff_sched / diff_mod if abs(diff_mod) > 1e-6 else float("nan")

    if len(sched_series) <= 1 or len(mod_100) <= 1:
        return {
            "sched_diff_pp": round(diff_sched, 2),
            "mod_diff_pp": round(diff_mod, 2),
            "ratio_point": round(ratio_point, 2) if not np.isnan(ratio_point) else "NA",
            "bootstrap_ci95_lower": "NA",
            "bootstrap_ci95_upper": "NA",
        }

    boot_ratios = []
    n_s = len(sched_series)
    n_m = len(mod_100)
    n_b = len(base_100)

    for _ in range(n_boot):
        s_samp = np.random.choice(sched_series, size=n_s, replace=True)
        m_samp = np.random.choice(mod_100, size=n_m, replace=True)
        b_samp = np.random.choice(base_100, size=n_b, replace=True)

        ds = (np.mean(s_samp) - base_ref) * 100.0
        dm = (np.mean(m_samp) - np.mean(b_samp)) * 100.0
        if abs(dm) > 1e-6:
            boot_ratios.append(ds / dm)

    if boot_ratios:
        ci_low = np.percentile(boot_ratios, 2.5)
        ci_up = np.percentile(boot_ratios, 97.5)
    else:
        ci_low, ci_up = float("nan"), float("nan")

    return {
        "sched_diff_pp": round(diff_sched, 2),
        "mod_diff_pp": round(diff_mod, 2),
        "ratio_point": round(ratio_point, 2) if not np.isnan(ratio_point) else "NA",
        "bootstrap_ci95_lower": round(float(ci_low), 2) if not np.isnan(ci_low) else "NA",
        "bootstrap_ci95_upper": round(float(ci_up), 2) if not np.isnan(ci_up) else "NA",
    }


def run_stage_stats_for_split(
    metrics_df: pd.DataFrame,
    split_name: str,
    results_dir: Path,
) -> Tuple[pd.DataFrame, List[str]]:
    """Run full statistical suite for a specific dataset split (val or test)."""
    df = metrics_df[metrics_df["split"] == split_name].copy()
    for col in ["mAP50", "mAP50_95", "AP_small"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Define groups
    groups = {
        "base_100": df[(df["family"] == "baseline") & (df["epochs"] == 100)],
        "module_100": df[(df["family"] == "postsppf") & (df["epochs"] == 100) & (df["close_mosaic"] == 10)],
        "capctrl_100": df[(df["family"] == "capctrl") & (df["epochs"] == 100)],
        "base_40": df[(df["family"] == "baseline") & (df["epochs"] == 40) & (~df["run"].str.contains("sgd"))],
        "base_60": df[(df["family"] == "baseline") & (df["epochs"] == 60)],
        "module_40": df[(df["family"] == "postsppf") & (df["epochs"] == 40) & (~df["run"].str.contains("sgd"))],
        "module_60": df[(df["family"] == "postsppf") & (df["epochs"] == 60)],
        # Control series ep40-sgd (SGD with lr0=0.01)
        "base_40_sgd": df[(df["family"] == "baseline") & (df["run"].str.contains("ep40sgd"))],
        "module_40_sgd": df[(df["family"] == "postsppf") & (df["run"].str.contains("ep40sgd"))],
    }

    comparisons = [
        ("module_vs_base", groups["base_100"], groups["module_100"]),
        ("capctrl_vs_base", groups["base_100"], groups["capctrl_100"]),
        ("capctrl_vs_module", groups["module_100"], groups["capctrl_100"]),
        ("base_ep40_vs_ep100", groups["base_100"], groups["base_40"]),
        ("base_ep60_vs_ep100", groups["base_100"], groups["base_60"]),
        ("module_ep40_vs_ep100", groups["module_100"], groups["module_40"]),
        ("module_ep60_vs_ep100", groups["module_100"], groups["module_60"]),
    ]

    # Only include control comparisons ep40-sgd if runs were trained/evaluated
    if len(groups["base_40_sgd"]) > 0 or len(groups["module_40_sgd"]) > 0:
        comparisons.extend(
            [
                ("base_ep40sgd_vs_ep100", groups["base_100"], groups["base_40_sgd"]),
                ("module_ep40sgd_vs_ep100", groups["module_100"], groups["module_40_sgd"]),
                ("base_ep40auto_vs_ep40sgd", groups["base_40_sgd"], groups["base_40"]),
                ("module_ep40auto_vs_ep40sgd", groups["module_40_sgd"], groups["module_40"]),
            ]
        )

    stat_rows = []
    discrepancies = []

    for comp_name, g1_df, g2_df in comparisons:
        for metric in ["mAP50", "mAP50_95", "AP_small"]:
            v1 = g1_df[metric].dropna().values
            v2 = g2_df[metric].dropna().values

            n1 = len(v1)
            n2 = len(v2)

            w = compute_welch_stats(v1, v2)
            se_pp = w.get("_se", 0.0) * 100.0
            df_val = w.get("_df", 0.0)
            diff_pp = w.get("_diff_pp", 0.0)

            tost = compute_tost_equivalence(diff_pp, se_pp, df_val, bounds=[0.5, 1.0])

            row = {
                "split": split_name,
                "comparison": comp_name,
                "metric": metric,
                "n_g1": n1,
                "n_g2": n2,
                "mean_g1": w["mean_g1"],
                "mean_g2": w["mean_g2"],
                "diff_pp": w["diff_pp"],
                "welch_t": w["welch_t"],
                "welch_df": w["welch_df"],
                "welch_p": w["welch_p"],
                "ci95_lower_pp": w["ci95_lower_pp"],
                "ci95_upper_pp": w["ci95_upper_pp"],
                "mwu_U": w["mwu_U"],
                "mwu_p": w["mwu_p"],
                "cohens_d": w["cohens_d"],
                "tost_p_0.5pp": tost["tost_p_bound_0.5pp"],
                "tost_p_1.0pp": tost["tost_p_bound_1.0pp"],
                "tost_ci90_lower_pp": tost["tost_ci90_lower_pp"],
                "tost_ci90_upper_pp": tost["tost_ci90_upper_pp"],
                "min_equiv_bound_pp": tost["min_equiv_bound_pp"],
            }
            stat_rows.append(row)

            # Verification of control check values on val split
            if split_name == "val" and metric == "mAP50":
                if comp_name == "module_vs_base":
                    exp_diff = 0.59
                    exp_p = 0.092
                    if abs(w["diff_pp"] - exp_diff) > 0.05 or abs(float(w["welch_p"]) - exp_p) > 0.02:
                        discrepancies.append(
                            f"[CHECK] module vs base val mAP50: expected diff +{exp_diff} pp (p~{exp_p}), got {w['diff_pp']:+.2f} pp (p={w['welch_p']})"
                        )
                    else:
                        print(
                            f"[VERIFIED] module vs base on val reproduces: diff={w['diff_pp']:+.2f} pp, p={w['welch_p']}"
                        )

                elif comp_name in ("module_vs_capctrl", "capctrl_vs_module"):
                    exp_diff = 0.09
                    exp_p = 0.82
                    if abs(abs(w["diff_pp"]) - exp_diff) > 0.05 or abs(float(w["welch_p"]) - exp_p) > 0.05:
                        discrepancies.append(
                            f"[CHECK] module vs capctrl val mAP50: expected diff +- {exp_diff} pp (p~{exp_p}), got {w['diff_pp']:+.2f} pp (p={w['welch_p']})"
                        )
                    else:
                        print(
                            f"[VERIFIED] module vs capctrl on val reproduces: diff={w['diff_pp']:+.2f} pp, p={w['welch_p']}, 95% CI=[{w['ci95_lower_pp']}, {w['ci95_upper_pp']}]"
                        )

    # Dual schedule effect calculation (R1.2, item 8):
    # Variant A: against baseline seed 0 (as in original paper: 0.6816 on val)
    # Variant B: against baseline 4-seed mean (0.6818 on val)
    base_100_mAP = groups["base_100"]["mAP50"].dropna().values if len(groups["base_100"]) > 0 else np.array([])
    mod_100_mAP = groups["module_100"]["mAP50"].dropna().values if len(groups["module_100"]) > 0 else np.array([])

    base_s0_row = groups["base_100"][groups["base_100"]["run"].str.endswith("__s0")]
    base_s0_val = (
        float(base_s0_row["mAP50"].values[0]) if len(base_s0_row) > 0 else (0.6816 if split_name == "val" else 0.6921)
    )
    base_mean_val = (
        float(np.mean(base_100_mAP)) if len(base_100_mAP) > 0 else (0.6818 if split_name == "val" else 0.6922)
    )

    ratio_rows = []
    ratio_targets = [
        ("base_ep40_auto", groups["base_40"]["mAP50"].dropna().values),
        ("base_ep40_sgd", groups["base_40_sgd"]["mAP50"].dropna().values),
        ("base_ep60_sgd", groups["base_60"]["mAP50"].dropna().values),
        ("module_ep40_auto", groups["module_40"]["mAP50"].dropna().values),
        ("module_ep40_sgd", groups["module_40_sgd"]["mAP50"].dropna().values),
        ("module_ep60_sgd", groups["module_60"]["mAP50"].dropna().values),
    ]

    print(f"\n[RATIO] Schedule effect vs Module effect on {split_name.upper()} (dual baseline reference):")
    print(f"        Ref A (Seed 0 baseline):     {base_s0_val * 100:.2f}% ({base_s0_val:.4f})")
    print(f"        Ref B (4-seed mean baseline): {base_mean_val * 100:.2f}% ({base_mean_val:.4f})")

    for series_name, series_vals in ratio_targets:
        if len(series_vals) == 0:
            continue
        # Variant A (seed 0)
        res_a = compute_schedule_vs_module_ratio(series_vals, base_s0_val, mod_100_mAP, base_100_mAP)
        ratio_rows.append(
            {
                "split": split_name,
                "series": series_name,
                "baseline_ref_type": "seed_0 (paper)",
                "baseline_ref_val": round(base_s0_val, 4),
                "sched_mean": round(float(np.mean(series_vals)), 4),
                "sched_gain_pp": res_a["sched_diff_pp"],
                "mod_gain_pp": res_a["mod_diff_pp"],
                "ratio_point": res_a["ratio_point"],
                "bootstrap_ci95_lower": res_a["bootstrap_ci95_lower"],
                "bootstrap_ci95_upper": res_a["bootstrap_ci95_upper"],
            }
        )

        # Variant B (4-seed mean)
        res_b = compute_schedule_vs_module_ratio(series_vals, base_mean_val, mod_100_mAP, base_100_mAP)
        ratio_rows.append(
            {
                "split": split_name,
                "series": series_name,
                "baseline_ref_type": "4seed_mean",
                "baseline_ref_val": round(base_mean_val, 4),
                "sched_mean": round(float(np.mean(series_vals)), 4),
                "sched_gain_pp": res_b["sched_diff_pp"],
                "mod_gain_pp": res_b["mod_diff_pp"],
                "ratio_point": res_b["ratio_point"],
                "bootstrap_ci95_lower": res_b["bootstrap_ci95_lower"],
                "bootstrap_ci95_upper": res_b["bootstrap_ci95_upper"],
            }
        )

        print(f"  * {series_name} (n={len(series_vals)}):")
        print(
            f"      vs Seed 0:    gain={res_a['sched_diff_pp']} pp -> ratio={res_a['ratio_point']}x [95% CI: {res_a['bootstrap_ci95_lower']}, {res_a['bootstrap_ci95_upper']}]"
        )
        print(
            f"      vs 4-seed mean: gain={res_b['sched_diff_pp']} pp -> ratio={res_b['ratio_point']}x [95% CI: {res_b['bootstrap_ci95_lower']}, {res_b['bootstrap_ci95_upper']}]"
        )

    if ratio_rows:
        ratio_df = pd.DataFrame(ratio_rows)
        ratio_csv = results_dir / f"schedule_ratios_{split_name}.csv"
        ratio_df.to_csv(ratio_csv, index=False)
        print(f"[SUCCESS] Wrote schedule effect ratios to: {ratio_csv}")

    out_df = pd.DataFrame(stat_rows)
    out_csv = results_dir / f"stats_{split_name}.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Wrote {len(stat_rows)} statistical rows to: {out_csv}\n")

    return out_df, discrepancies


def run_stage_stats(results_dir: Path) -> Dict:
    """Execute complete statistical analysis on both splits."""
    metrics_csv = results_dir / "metrics_all.csv"
    if not metrics_csv.is_file():
        # Fallback to C2/C3 tables
        print(
            f"[INFO] metrics_all.csv not found at {metrics_csv}; building from tables/C2_final_metrics.csv and tables/C3_test_metrics.csv"
        )
        from scripts.revision.stage_eval import run_stage_eval

        metrics_csv = run_stage_eval(results_dir=results_dir, data_yaml=REPO_ROOT / "exdark.yaml", dry_run=True)

    df_metrics = pd.read_csv(metrics_csv)

    print("=" * 80)
    print("  STAGE STATS: FORMAL STATISTICAL TESTING (R1.1, R1.2)")
    print("=" * 80)

    val_df, val_disc = run_stage_stats_for_split(df_metrics, "val", results_dir)
    test_df, test_disc = run_stage_stats_for_split(df_metrics, "test", results_dir)

    all_disc = val_disc + test_disc
    if all_disc:
        print("\n[WARNINGS / DISCREPANCIES]")
        for d in all_disc:
            print(f"  {d}")
    else:
        print("\n[SUCCESS] All control verification checks passed with zero discrepancies!")

    return {
        "val_stats_csv": str(results_dir / "stats_val.csv"),
        "test_stats_csv": str(results_dir / "stats_test.csv"),
        "discrepancies": all_disc,
    }


if __name__ == "__main__":
    run_stage_stats(DEFAULT_RESULTS_DIR)
