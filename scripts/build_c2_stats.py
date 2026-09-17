import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
import warnings

def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    if dof <= 0: return float('nan')
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    if pooled_std == 0: return float('nan')
    return (np.mean(x) - np.mean(y)) / pooled_std

def get_stats(x, y):
    if len(x) == 0 or len(y) == 0:
        return {"t": "NA", "df": "NA", "p_welch": "NA", "ci_95": "NA", "U": "NA", "p_mw": "NA", "d": "NA", "min_p": "NA"}
    if len(x) == 1 and len(y) == 1:
        return {"t": "NA", "df": "NA", "p_welch": "NA", "ci_95": "NA", "U": "NA", "p_mw": "NA", "d": "NA", "min_p": "NA"}
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(x) > 1 and len(y) > 1:
            # Welch's t-test
            t, p_welch = ttest_ind(x, y, equal_var=False)
            df = len(x) + len(y) - 2 # Approx
            # CI
            diff = np.mean(x) - np.mean(y)
            se = np.sqrt(np.var(x, ddof=1)/len(x) + np.var(y, ddof=1)/len(y))
            ci_95 = f"[{diff - 1.96*se:.4f}, {diff + 1.96*se:.4f}]"
            # Mann-Whitney U
            U, p_mw = mannwhitneyu(x, y, alternative='two-sided')
            # Cohen's d
            d = cohens_d(x, y)
            # min_p for mw
            import math
            min_p = 2 / math.comb(len(x) + len(y), len(x))
            
            return {
                "t": round(t, 4) if not np.isnan(t) else "NA",
                "df": df,
                "p_welch": round(p_welch, 4) if not np.isnan(p_welch) else "NA",
                "ci_95": ci_95,
                "U": U,
                "p_mw": round(p_mw, 4) if not np.isnan(p_mw) else "NA",
                "d": round(d, 4) if not np.isnan(d) else "NA",
                "min_p": round(min_p, 4)
            }
        else:
            return {"t": "NA", "df": "NA", "p_welch": "NA", "ci_95": "NA", "U": "NA", "p_mw": "NA", "d": "NA", "min_p": "NA"}

if __name__ == "__main__":
    df_metrics = pd.read_csv("tables/C2_final_metrics.csv")
    df_diag = pd.read_csv("tables/C2_final_diag.csv")
    
    merged = pd.merge(df_metrics, df_diag, on="run", how="inner")
    
    # groups
    base = merged[merged["run"].str.startswith("e2p__baseline")]
    postsppf = merged[merged["run"].str.startswith("e2p__postsppf") | merged["run"].str.startswith("e3p__pos-postsppf")]
    
    rows = []
    
    def add_comparison(name, g1, g2, n1, n2):
        for metric in ["mAP50", "mAP50_95", "L_mean", "L_std"]:
            x = g1[metric].dropna().astype(float).values
            y = g2[metric].dropna().astype(float).values
            diff_pp = (np.mean(x) - np.mean(y)) * 100 if len(x) > 0 and len(y) > 0 else "NA"
            
            stats = get_stats(x, y)
            rows.append({
                "comparison": name,
                "metric": metric,
                "mean_g1": round(np.mean(x), 4) if len(x)>0 else "NA",
                "std_g1": round(np.std(x, ddof=1), 4) if len(x)>1 else "NA",
                "n_g1": len(x),
                "mean_g2": round(np.mean(y), 4) if len(y)>0 else "NA",
                "std_g2": round(np.std(y, ddof=1), 4) if len(y)>1 else "NA",
                "n_g2": len(y),
                "diff_pp": round(diff_pp, 4) if diff_pp != "NA" else "NA",
                **stats
            })
            
    add_comparison("base_vs_postsppf(lam=0)", base, postsppf, 4, 4)
    
    # against e6p__lam-1e-4
    e6p_1e4 = merged[merged["run"].str.startswith("e6p__lam-1e-4")]
    add_comparison("postsppf_vs_lam-1e-4", postsppf, e6p_1e4, 4, 3)
    
    # single lam points
    for lam in ["1e-5", "1e-3", "1e-2", "1e-1"]:
        g2 = merged[merged["run"].str.startswith(f"e6p__lam-{lam}")]
        if len(g2) > 0:
            add_comparison(f"postsppf_vs_lam-{lam}", postsppf, g2, 4, len(g2))
            
    res_df = pd.DataFrame(rows)
    res_df.to_csv("tables/C2_final_stats.csv", index=False)
    print("[SUCCESS] tables/C2_final_stats.csv generated.")
