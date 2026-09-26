"""Quantify the relationship between FE-failure risk and POD/NODE error.

Several analyses on the master per-simulation table: Spearman/Pearson
correlations between failure risk (p_fail, d_fail) and reconstruction
error, a high-risk (top quartile p_fail) vs. low-risk group comparison
(Mann-Whitney U test plus a bootstrap confidence interval on the median
difference), the explicit high-k1/low-mu parameter region, and
kappa-extreme bins (the same 8 quantile bins used in
step3_marginal_panel.py).

Reads: step9_master_per_sim_table.csv (written by step9_master_join.py).

Writes: step10_correlations.csv, step10_highrisk_lowrisk.csv,
step10_high_k1_low_mu_region.csv, step10_kappa_bins.csv -- these supply
the error-vs-risk numbers reported in SI Section 2, and are consumed by
step11_figures.py.

Run from the repository root; all paths resolve relative to this script.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RNG_SEED = 20260901
N_BOOT = 5000

master = pd.read_csv(os.path.join(OUT_DIR, "step9_master_per_sim_table.csv"))
val = master[master["is_validation"]].copy()
assert len(val) == 185

rng = np.random.default_rng(RNG_SEED)


def bootstrap_median_diff_ci(a, b, n=N_BOOT):
    """Bootstrap a 95% CI on median(b) - median(a) by resampling each group
    with replacement n times. Used alongside the Mann-Whitney U test below
    because MWU only tests whether the two distributions differ, not by how
    much; this gives an effect-size estimate directly. Medians (rather than
    means) are used since the error distributions here are right-skewed."""
    diffs = np.empty(n)
    for i in range(n):
        ai = rng.choice(a, size=len(a), replace=True)
        bi = rng.choice(b, size=len(b), replace=True)
        diffs[i] = np.median(bi) - np.median(ai)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.median(b) - np.median(a)), float(lo), float(hi)


def spearman_report(x, y, label):
    """Report both Spearman (rank-based, robust to nonlinearity/outliers)
    and Pearson (linear) correlation, so a purely monotonic relationship
    isn't missed just because it isn't linear."""
    rho, p = stats.spearmanr(x, y)
    pear_r, pear_p = stats.pearsonr(x, y)
    print(f"  {label}: n={len(x)}  Spearman rho={rho:+.4f} (p={p:.4g})  Pearson r={pear_r:+.4f} (p={pear_p:.4g})")
    return dict(label=label, n=len(x), spearman_rho=rho, spearman_p=p, pearson_r=pear_r, pearson_p=pear_p)


results = []

print("=== Section 6/7: Spearman correlation, error vs. p_fail ===")
results.append(spearman_report(master["p_fail_oof"], master["e_pod"], "e_pod vs p_fail (n=927, all converged)"))
results.append(spearman_report(val["p_fail_oof"], val["e_node_A"], "e_node_A vs p_fail (n=185, validation)"))
results.append(spearman_report(val["p_fail_oof"], val["e_node_D"], "e_node_D vs p_fail (n=185, validation)"))

print("\n=== Section 11: Spearman correlation, error vs. d_fail (distance to nearest failed point) ===")
results.append(spearman_report(master["d_fail"], master["e_pod"], "e_pod vs d_fail (n=927)"))
results.append(spearman_report(val["d_fail"], val["e_node_A"], "e_node_A vs d_fail (n=185)"))
results.append(spearman_report(val["d_fail"], val["e_node_D"], "e_node_D vs d_fail (n=185)"))
pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, "step10_correlations.csv"), index=False)

print("\n=== Section 8: high-risk (top 25% p_fail) vs low-risk (bottom 75%) group comparison ===")
group_results = []
for pop, err_col, label in [(master, "e_pod", "e_pod (n=927)"),
                              (val, "e_node_A", "e_node_A (n=185 val)"),
                              (val, "e_node_D", "e_node_D (n=185 val)")]:
    thresh = pop["p_fail_oof"].quantile(0.75)
    low = pop.loc[pop["p_fail_oof"] <= thresh, err_col].values
    high = pop.loc[pop["p_fail_oof"] > thresh, err_col].values
    u_stat, u_p = stats.mannwhitneyu(low, high, alternative="two-sided")
    diff, lo, hi = bootstrap_median_diff_ci(low, high)
    ratio = np.median(high) / np.median(low)
    print(f"  {label}: n_low={len(low)} n_high={len(high)}  median_low={np.median(low):.4f} "
          f"median_high={np.median(high):.4f}  ratio={ratio:.3f}  "
          f"MWU p={u_p:.4g}  bootstrap median diff={diff:+.4f} [{lo:+.4f},{hi:+.4f}]")
    group_results.append(dict(metric=label, n_low=len(low), n_high=len(high),
                                median_low=np.median(low), median_high=np.median(high),
                                ratio_high_over_low=ratio, mwu_p=u_p,
                                bootstrap_diff=diff, ci_lo=lo, ci_hi=hi))
pd.DataFrame(group_results).to_csv(os.path.join(OUT_DIR, "step10_highrisk_lowrisk.csv"), index=False)

print("\n=== Section 9: explicit high-k1/low-mu region (top-tercile k1, bottom-tercile mu) ===")
k1_top = master["k1"].quantile(2 / 3)
mu_bot = master["mu"].quantile(1 / 3)
region_results = []
for pop, err_cols, label in [(master, ["e_pod"], "n=927"), (val, ["e_node_A", "e_node_D"], "n=185 val")]:
    inside = pop["k1"] >= k1_top
    inside &= pop["mu"] <= mu_bot
    for col in err_cols:
        med_in = pop.loc[inside, col].median()
        med_out = pop.loc[~inside, col].median()
        u_stat, u_p = stats.mannwhitneyu(pop.loc[inside, col], pop.loc[~inside, col], alternative="two-sided")
        print(f"  [{label}] {col}: n_inside={inside.sum()} n_outside={(~inside).sum()}  "
              f"median_inside={med_in:.4f}  median_outside={med_out:.4f}  "
              f"ratio={med_in/med_out:.3f}  MWU p={u_p:.4g}")
        region_results.append(dict(population=label, metric=col, n_inside=int(inside.sum()),
                                     n_outside=int((~inside).sum()), median_inside=med_in,
                                     median_outside=med_out, ratio=med_in / med_out, mwu_p=u_p))
pd.DataFrame(region_results).to_csv(os.path.join(OUT_DIR, "step10_high_k1_low_mu_region.csv"), index=False)

print("\n=== Section 10: kappa-extreme bins (same 8 quantile bins as the marginal FE panel) ===")
kappa_bins_results = []
master["kappa_bin"] = pd.qcut(master["kappa"], q=8, duplicates="drop")
for bin_label, g in master.groupby("kappa_bin", observed=True):
    row = dict(kappa_bin=str(bin_label), kappa_mid=g["kappa"].mean(), n=len(g),
                median_e_pod=g["e_pod"].median())
    gv = g[g["is_validation"]]
    row["n_val"] = len(gv)
    row["median_e_node_A"] = gv["e_node_A"].median() if len(gv) > 0 else np.nan
    row["median_e_node_D"] = gv["e_node_D"].median() if len(gv) > 0 else np.nan
    kappa_bins_results.append(row)
kappa_df = pd.DataFrame(kappa_bins_results).sort_values("kappa_mid")
print(kappa_df.to_string(index=False))
kappa_df.to_csv(os.path.join(OUT_DIR, "step10_kappa_bins.csv"), index=False)

print("\nDONE")
