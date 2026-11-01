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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, t as t_dist

from scripts.revision.config import DEFAULT_RESULTS_DIR, REPO_ROOT


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
    sched_ep: np.ndarray,
    sched_base: np.ndarray,
    mod_100: np.ndarray,
    base_100: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """Calculate ratio of schedule effect to module effect with 10,000 bootstrap resamples."""
    np.random.seed(seed)
    diff_sched = (float(np.mean(sched_ep)) - float(np.mean(sched_base))) * 100.0
    diff_mod = (float(np.mean(mod_100)) - float(np.mean(base_100))) * 100.0

    ratio_point = diff_sched / diff_mod if abs(diff_mod) > 1e-6 else float("nan")

    if len(sched_ep) <= 1 or len(mod_100) <= 1:
        return {
            "ratio_point": round(ratio_point, 4) if not np.isnan(ratio_point) else "NA",
            "bootstrap_ci95_lower": "NA",
            "bootstrap_ci95_upper": "NA",
        }

    boot_ratios = []
    n_se = len(sched_ep)
    n_sb = len(sched_base)
    n_m = len(mod_100)
    n_b = len(base_100)

    for _ in range(n_boot):
        se_samp = np.random.choice(sched_ep, size=n_se, replace=True)
        sb_samp = np.random.choice(sched_base, size=n_sb, replace=True)
        m_samp = np.random.choice(mod_100, size=n_m, replace=True)
        b_samp = np.random.choice(base_100, size=n_b, replace=True)

        ds = (np.mean(se_samp) - np.mean(sb_samp)) * 100.0
        dm = (np.mean(m_samp) - np.mean(b_samp)) * 100.0
        if abs(dm) > 1e-6:
            boot_ratios.append(ds / dm)

    if boot_ratios:
        ci_low = np.percentile(boot_ratios, 2.5)
        ci_up = np.percentile(boot_ratios, 97.5)
    else:
        ci_low, ci_up = float("nan"), float("nan")

    return {
        "ratio_point": round(ratio_point, 4) if not np.isnan(ratio_point) else "NA",
        "bootstrap_ci95_lower": round(float(ci_low), 4) if not np.isnan(ci_low) else "NA",
        "bootstrap_ci95_upper": round(float(ci_up), 4) if not np.isnan(ci_up) else "NA",
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
        "base_40": df[(df["family"] == "baseline") & (df["epochs"] == 40)],
        "base_60": df[(df["family"] == "baseline") & (df["epochs"] == 60)],
        "module_40": df[(df["family"] == "postsppf") & (df["epochs"] == 40)],
        "module_60": df[(df["family"] == "postsppf") & (df["epochs"] == 60)],
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

    # Ratio of schedule effect vs module effect
    sched_40_base = groups["base_40"]["mAP50"].dropna().values if len(groups["base_40"]) > 0 else np.array([])
    sched_100_base = groups["base_100"]["mAP50"].dropna().values if len(groups["base_100"]) > 0 else np.array([])
    mod_100 = groups["module_100"]["mAP50"].dropna().values if len(groups["module_100"]) > 0 else np.array([])

    if len(sched_40_base) > 0 and len(sched_100_base) > 0 and len(mod_100) > 0:
        ratio_res = compute_schedule_vs_module_ratio(sched_40_base, sched_100_base, mod_100, sched_100_base)
        print(f"\n[RATIO] Schedule effect (40 ep vs 100 ep) vs Module effect on {split_name}:")
        print(
            f"        Point estimate: {ratio_res['ratio_point']}x | 95% Bootstrap CI: [{ratio_res['bootstrap_ci95_lower']}; {ratio_res['bootstrap_ci95_upper']}]"
        )
    else:
        print(
            f"\n[RATIO] Schedule effect (40 ep vs 100 ep) vs Module effect on {split_name}: PENDING full schedule data"
        )

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
