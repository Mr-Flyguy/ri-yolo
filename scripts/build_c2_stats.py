import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
import warnings
import math

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
    if len(x) == 1 or len(y) == 1:
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
    
    merged = pd.merge(df_metrics, df_diag, on="run", how="left")
    
    # groups
    base = merged[merged["run"].str.startswith("e2p__baseline")]
    postsppf = merged[merged["run"].str.startswith("e2p__postsppf") | merged["run"].str.startswith("e3p__pos-postsppf")]
    
    rows = []
    
    def add_comparison(name, g1, g2):
        for metric in ["mAP50", "mAP50_95", "AP_small", "L_std"]:
            if metric not in g1.columns or metric not in g2.columns:
                continue
                
            x_raw = g1[metric].dropna()
            y_raw = g2[metric].dropna()
            
            x = pd.to_numeric(x_raw, errors='coerce').dropna().values
            y = pd.to_numeric(y_raw, errors='coerce').dropna().values
            
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
            
    add_comparison("base_vs_postsppf(lam=0)", base, postsppf)
    
    # against e6p__lam-1e-4
    e6p_1e4 = merged[merged["run"].str.startswith("e6p__lam-1e-4")]
    add_comparison("postsppf_vs_lam-1e-4", postsppf, e6p_1e4)
    
    # single lam points
    for lam in ["1e-5", "1e-3", "1e-2", "1e-1"]:
        g2 = merged[merged["run"].str.startswith(f"e6p__lam-{lam}")]
        if len(g2) > 0:
            add_comparison(f"postsppf_vs_lam-{lam}", postsppf, g2)
            
    # nwd-calib-norsl
    nwd_norsl = merged[merged["run"].str.startswith("e7p__nwd-calib-norsl")]
    if len(nwd_norsl) > 0:
        add_comparison("postsppf_vs_nwd-calib-norsl", postsppf, nwd_norsl)
        
    # sizegate broken vs fixed
    broken = merged[merged["run"] == "runs/detect/runs/e7p__nwd-sizegate__s0"]
    fixed = merged[merged["run"] == "runs_fixed/e7p__nwd-sizegate__s0"]
    if len(broken) > 0 and len(fixed) > 0:
        add_comparison("sizegate_broken_vs_fixed", broken, fixed)
        
    res_df = pd.DataFrame(rows)
    res_df.to_csv("tables/C2_final_stats.csv", index=False)
    print("[SUCCESS] tables/C2_final_stats.csv generated.")
