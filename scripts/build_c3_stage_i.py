import csv
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind, mannwhitneyu

def get_runs_for_group(runs_list, metrics_df, diag_df):
    """Filter dataframes for specific runs."""
    m = metrics_df[metrics_df["run"].isin(runs_list)]
    d = diag_df[diag_df["run"].isin(runs_list)]
    return m, d

def stats_row(name, grp1_runs, grp2_runs, m_df, d_df, n1, n2):
    m1, d1 = get_runs_for_group(grp1_runs, m_df, d_df)
    m2, d2 = get_runs_for_group(grp2_runs, m_df, d_df)
    
    row = {"comparison": name, "n_g1": len(m1), "n_g2": len(m2)}
    
    # We must enforce the requested n1 and n2 for correct calculations,
    # or at least acknowledge if the sample size is missing.
    # Note: if n=1, p-values are NA.
    
    metrics_to_calc = [
        ("mAP50", m1, m2),
        ("mAP50_95", m1, m2),
        ("AP_small", m1, m2),
        ("L_std", d1, d2)
    ]
    
    for m_name, df1, df2 in metrics_to_calc:
        if m_name not in df1.columns or m_name not in df2.columns:
            for k in [f"{m_name}_mean_g1", f"{m_name}_mean_g2", f"{m_name}_diff_pp",
                      f"{m_name}_welch_t", f"{m_name}_welch_df", f"{m_name}_welch_p",
                      f"{m_name}_ci_lower_pp", f"{m_name}_ci_upper_pp",
                      f"{m_name}_mwu_U", f"{m_name}_mwu_p", f"{m_name}_cohens_d"]:
                row[k] = "NA"
            continue
            
        v1 = df1[m_name].dropna().astype(float).values
        v2 = df2[m_name].dropna().astype(float).values
        
        if len(v1) == 0 or len(v2) == 0:
            for k in [f"{m_name}_mean_g1", f"{m_name}_mean_g2", f"{m_name}_diff_pp",
                      f"{m_name}_welch_t", f"{m_name}_welch_df", f"{m_name}_welch_p",
                      f"{m_name}_ci_lower_pp", f"{m_name}_ci_upper_pp",
                      f"{m_name}_mwu_U", f"{m_name}_mwu_p", f"{m_name}_cohens_d"]:
                row[k] = "NA"
            continue

        mean1 = v1.mean()
        mean2 = v2.mean()
        diff = mean2 - mean1
        diff_pp = diff * (100 if m_name != "L_std" else 1) # Note: L_std is not % typically, but we scale by 100 for APs
        if m_name != "L_std":
            diff_disp = diff * 100
        else:
            diff_disp = diff
            
        row[f"{m_name}_mean_g1"] = mean1
        row[f"{m_name}_mean_g2"] = mean2
        row[f"{m_name}_diff_pp"] = diff_disp
        
        if len(v1) == 1 or len(v2) == 1:
            row[f"{m_name}_welch_t"] = "NA"
            row[f"{m_name}_welch_df"] = "NA"
            row[f"{m_name}_welch_p"] = "NA"
            row[f"{m_name}_ci_lower_pp"] = "NA"
            row[f"{m_name}_ci_upper_pp"] = "NA"
            row[f"{m_name}_mwu_U"] = "NA"
            row[f"{m_name}_mwu_p"] = "NA"
            row[f"{m_name}_cohens_d"] = "NA"
            continue
            
        # Welch's t-test
        t_stat, p_val = ttest_ind(v1, v2, equal_var=False)
        var1 = np.var(v1, ddof=1)
        var2 = np.var(v2, ddof=1)
        n_1 = len(v1)
        n_2 = len(v2)
        df_num = (var1/n_1 + var2/n_2)**2
        df_den = (var1/n_1)**2/(n_1-1) + (var2/n_2)**2/(n_2-1)
        df = df_num / df_den if df_den > 0 else 1.0
        
        # 95% CI
        import scipy.stats
        t_crit = scipy.stats.t.ppf(0.975, df)
        margin = t_crit * np.sqrt(var1/n_1 + var2/n_2)
        ci_lower = diff - margin
        ci_upper = diff + margin
        if m_name != "L_std":
            ci_lower *= 100
            ci_upper *= 100
            
        # Mann-Whitney
        u_stat, mwu_p = mannwhitneyu(v1, v2, alternative='two-sided')
        
        # Cohen's d
        s_pooled = np.sqrt(((n_1-1)*var1 + (n_2-1)*var2) / (n_1+n_2-2))
        cohens_d = diff / s_pooled if s_pooled > 0 else 0.0
        
        row[f"{m_name}_welch_t"] = t_stat
        row[f"{m_name}_welch_df"] = df
        row[f"{m_name}_welch_p"] = p_val
        row[f"{m_name}_ci_lower_pp"] = ci_lower
        row[f"{m_name}_ci_upper_pp"] = ci_upper
        row[f"{m_name}_mwu_U"] = u_stat
        row[f"{m_name}_mwu_p"] = mwu_p
        row[f"{m_name}_cohens_d"] = cohens_d
        
    return row

def build_c3_stats():
    metrics_p = Path("tables/C2_final_metrics.csv")
    diag_p = Path("tables/C2_final_diag.csv")
    if not metrics_p.exists() or not diag_p.exists():
        return
        
    m_df = pd.read_csv(metrics_p)
    d_df = pd.read_csv(diag_p)
    
    base_runs = [f"e2p__baseline__s{i}" for i in range(4)]
    postsppf_runs = [f"e2p__postsppf__s{i}" for i in range(4)]
    capctrl_runs = [f"e3p__capctrl__s{i}" for i in range(3)]
    
    pos_p3 = ["e3p__pos-p3__s0"]
    pos_p4 = ["e3p__pos-p4__s0"]
    pos_presppf = ["e3p__pos-presppf__s0"]
    
    rows = []
    rows.append(stats_row("base_vs_postsppf", base_runs, postsppf_runs, m_df, d_df, 4, 4))
    rows.append(stats_row("postsppf_vs_capctrl", postsppf_runs, capctrl_runs, m_df, d_df, 4, 3))
    rows.append(stats_row("base_vs_capctrl", base_runs, capctrl_runs, m_df, d_df, 4, 3))
    rows.append(stats_row("base_vs_pos_p3", base_runs, pos_p3, m_df, d_df, 4, 1))
    rows.append(stats_row("base_vs_pos_p4", base_runs, pos_p4, m_df, d_df, 4, 1))
    rows.append(stats_row("base_vs_pos_presppf", base_runs, pos_presppf, m_df, d_df, 4, 1))
    
    df_out = pd.DataFrame(rows)
    out_p = "tables/C3_stats.csv"
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    df_out.to_csv(out_p, index=False)
    print(f"[SUCCESS] {out_p} generated.")

def build_c3_table1():
    metrics_p = Path("tables/C2_final_metrics.csv")
    diag_p = Path("tables/C2_final_diag.csv")
    if not metrics_p.exists() or not diag_p.exists():
        return
    m_df = pd.read_csv(metrics_p)
    d_df = pd.read_csv(diag_p)
    
    configs = [
        ("Base model", [f"e2p__baseline__s{i}" for i in range(4)]),
        ("+ module after SPPF", [f"e2p__postsppf__s{i}" for i in range(4)]),
        ("+ regularizer", ["e6p__lam-1e-2__s0"]),
        ("+ box metric", ["e7p__nwd-calib__s0"]),
        ("Capacity control", [f"e3p__capctrl__s{i}" for i in range(3)])
    ]
    
    # Calculate baseline means for diffs
    base_m, _ = get_runs_for_group([f"e2p__baseline__s{i}" for i in range(4)], m_df, d_df)
    base_map50 = base_m["mAP50"].dropna().astype(float).mean() if len(base_m) > 0 else 0
    base_map50_95 = base_m["mAP50_95"].dropna().astype(float).mean() if len(base_m) > 0 else 0
    
    rows = []
    for cfg_name, runs in configs:
        m, d = get_runs_for_group(runs, m_df, d_df)
        n = len(m)
        if n == 0:
            continue
            
        map50 = m["mAP50"].dropna().astype(float)
        map50_95 = m["mAP50_95"].dropna().astype(float)
        aps = m["AP_small"].dropna().astype(float)
        apm = m["AP_medium"].dropna().astype(float)
        apl = m["AP_large"].dropna().astype(float)
        gamma = d["gamma_final"].dropna().replace("NA", np.nan).astype(float)
        
        row = {
            "config": cfg_name,
            "n_runs": n,
            "mAP50": map50.mean(),
            "mAP50_sd": map50.std(ddof=1) if n > 1 else "NA",
            "mAP50_95": map50_95.mean(),
            "mAP50_95_sd": map50_95.std(ddof=1) if n > 1 else "NA",
            "AP_small": aps.mean() if len(aps)>0 else "NA",
            "AP_medium": apm.mean() if len(apm)>0 else "NA",
            "AP_large": apl.mean() if len(apl)>0 else "NA",
            "delta_mAP50_pp": (map50.mean() - base_map50)*100 if n > 0 else "NA",
            "delta_mAP50_95_pp": (map50_95.mean() - base_map50_95)*100 if n > 0 else "NA",
            "gamma_mean": gamma.mean() if len(gamma)>0 else "NA"
        }
        rows.append(row)
        
    df_out = pd.DataFrame(rows)
    df_out.to_csv("tables/C3_table1.csv", index=False)
    print("[SUCCESS] tables/C3_table1.csv generated.")

def build_c3_table2():
    bench_p = Path("artifacts/phase0/bench_gpu_rtx3070ti.csv")
    if not bench_p.exists():
        return
        
    df = pd.read_csv(bench_p)
    df = df[(df["batch"] == 1) & (df["half"] == 0)]
    
    models = ["yolov8s.yaml", "yolov8s-rfd-postsppf.yaml", "yolov8s-capctrl.yaml", 
              "yolov8s-rfd-p3.yaml", "yolov8s-rfd-p4.yaml", "yolov8s-rfd-presppf.yaml"]
    
    df = df[df["model"].isin(models)].copy()
    
    # Calculate overhead against yolov8s.yaml
    base_row = df[df["model"] == "yolov8s.yaml"]
    if len(base_row) == 0:
        return
    base_lat = base_row.iloc[0]["lat_med_ms"]
    
    df["overhead_ms"] = df["lat_med_ms"] - base_lat
    df["overhead_pct"] = (df["overhead_ms"] / base_lat) * 100
    
    cols = ["model", "params_M", "gflops", "lat_med_ms", "lat_p95_ms", "fps", "overhead_ms", "overhead_pct"]
    df = df[cols]
    
    df.to_csv("tables/C3_table2.csv", index=False)
    print("[SUCCESS] tables/C3_table2.csv generated.")

def build_c3_regime():
    metrics_p = Path("tables/C2_final_metrics.csv")
    if not metrics_p.exists():
        return
    m_df = pd.read_csv(metrics_p)
    
    rows = []
    
    # epochs panel
    ep_map = {
        "e5p__baseline-ep40__s0": ("baseline", 40),
        "e5p__baseline-ep60__s0": ("baseline", 60),
        "e2p__baseline__s0": ("baseline", 100),
        "e5p__postsppf-ep40__s0": ("postsppf", 40),
        "e5p__postsppf-ep60__s0": ("postsppf", 60),
        "e2p__postsppf__s0": ("postsppf", 100),
    }
    for run, (fam, ep) in ep_map.items():
        m = m_df[m_df["run"] == run]
        if len(m) > 0:
            clean_pct = ((ep - 10) / ep) * 100
            row = {
                "panel": "epochs", "family": fam, "run": run, "epochs": ep, "close_mosaic": "NA",
                "clean_epochs_pct": clean_pct, "mAP50": m.iloc[0]["mAP50"], "mAP50_95": m.iloc[0]["mAP50_95"]
            }
            rows.append(row)
            
    # mosaic panel
    mos_map = {
        "e4p__mosaic-close0__s0": 0,
        "e2p__postsppf__s0": 10,
        "e4p__mosaic-close50__s0": 50,
    }
    for run, mos in mos_map.items():
        m = m_df[m_df["run"] == run]
        if len(m) > 0:
            clean_pct = (mos / 100) * 100
            row = {
                "panel": "mosaic", "family": "postsppf", "run": run, "epochs": 100, "close_mosaic": mos,
                "clean_epochs_pct": clean_pct, "mAP50": m.iloc[0]["mAP50"], "mAP50_95": m.iloc[0]["mAP50_95"]
            }
            rows.append(row)
            
    df_out = pd.DataFrame(rows)
    df_out.to_csv("tables/C3_regime.csv", index=False)
    print("[SUCCESS] tables/C3_regime.csv generated.")

if __name__ == "__main__":
    build_c3_table1()
    build_c3_table2()
    build_c3_regime()
    build_c3_stats()
