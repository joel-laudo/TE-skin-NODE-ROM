"""Two summary figures for SI Section 2 / Figure 2, built from the master
per-simulation table and the kappa-bin error trends.

Figure 1: 3-panel scatter of e_pod / e_node_A / e_node_D vs. predicted
FE-failure probability (p_fail), with a binned-median trend line rather
than an imposed linear fit -- the risk-error relationship is not assumed
to be linear.

Figure 2: kappa-bin trend for all three error metrics side by side,
testing directly whether the U-shaped kappa relationship found for FE
failure (step3_marginal_panel.py) recurs in POD/NODE reconstruction error.

Reads: step9_master_per_sim_table.csv (from step9_master_join.py) and
step10_kappa_bins.csv (from step10_error_vs_risk_analysis.py).

Writes: fig1_error_vs_failure_risk.{pdf,png} and
fig2_kappa_bin_error_trend.{pdf,png}, the two figures referenced in SI
Section 2 / Figure 2.

Run from the repository root; all paths resolve relative to this script.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

master = pd.read_csv(os.path.join(OUT_DIR, "step9_master_per_sim_table.csv"))
val = master[master["is_validation"]].copy()
kappa_df = pd.read_csv(os.path.join(OUT_DIR, "step10_kappa_bins.csv"))

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                       "savefig.dpi": 600, "savefig.bbox": "tight"})


def binned_median_trend(x, y, n_bins=8):
    """Bin x into n_bins quantile bins and summarize y within each bin by
    its median and interquartile range (25th/75th percentile). Used to
    visualize the error-vs-risk trend without assuming linearity."""
    bins = pd.qcut(x, q=n_bins, duplicates="drop")
    g = pd.DataFrame({"x": x, "y": y, "bin": bins}).groupby("bin", observed=True).agg(
        x_mid=("x", "mean"), y_med=("y", "median"), y_lo=("y", lambda v: np.percentile(v, 25)),
        y_hi=("y", lambda v: np.percentile(v, 75)))
    return g.sort_values("x_mid")


# --- Figure 1: 3-panel scatter vs p_fail ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
panels = [
    (axes[0], master["p_fail_oof"], master["e_pod"], "POD reconstruction error", "e_POD (cm)",
     "All 927 converged sims"),
    (axes[1], val["p_fail_oof"], val["e_node_A"], "Model A NODE displacement error", "e_NODE,A (cm)",
     "185 validation sims"),
    (axes[2], val["p_fail_oof"], val["e_node_D"], "Model D NODE displacement error", "e_NODE,D (cm)",
     "185 validation sims"),
]
for ax, x, y, title, ylabel, subtitle in panels:
    ax.scatter(x, y, s=18, alpha=0.4, color="steelblue")
    trend = binned_median_trend(x.values, y.values)
    ax.plot(trend["x_mid"], trend["y_med"], color="tab:red", linewidth=2.5, marker="o", markersize=5,
            label="binned median")
    ax.fill_between(trend["x_mid"], trend["y_lo"], trend["y_hi"], color="tab:red", alpha=0.15,
                     label="binned IQR")
    ax.set_xlabel("Predicted P(FE non-convergence)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n({subtitle})", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
fig.suptitle("Reconstruction/prediction error vs. out-of-fold predicted FE-failure risk", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT_DIR, "fig1_error_vs_failure_risk.pdf"))
fig.savefig(os.path.join(OUT_DIR, "fig1_error_vs_failure_risk.png"))
plt.close(fig)
print("saved fig1_error_vs_failure_risk.{pdf,png}")

# --- Figure 2: kappa-bin trend, all 3 error metrics -- small multiples
# (NOT a dual/twin y-axis: POD (n=927/bin) and NODE (n~23/bin, validation
# only) are different sample sizes as well as different scales, so sharing
# one axis pair would also be misleading about precision, not just visually
# cluttered).
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
specs = [
    (axes[0], "median_e_pod", "tab:blue", "POD error", "e_POD (cm)", "n=927/bin"),
    (axes[1], "median_e_node_A", "tab:orange", "Model A NODE error", "e_NODE,A (cm)", "n~23/bin, val only"),
    (axes[2], "median_e_node_D", "tab:green", "Model D NODE error", "e_NODE,D (cm)", "n~23/bin, val only"),
]
for ax, col, color, title, ylabel, note in specs:
    ax.plot(kappa_df["kappa_mid"], kappa_df[col], marker="o", color=color, linewidth=2)
    ax.set_xlabel("kappa (fiber dispersion)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n({note})", fontsize=11)
    ax.grid(True, alpha=0.25)
fig.suptitle("Median error by kappa bin (same 8 quantile bins as the FE-failure marginal panel)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUT_DIR, "fig2_kappa_bin_error_trend.pdf"))
fig.savefig(os.path.join(OUT_DIR, "fig2_kappa_bin_error_trend.png"))
plt.close(fig)
print("saved fig2_kappa_bin_error_trend.{pdf,png}")
print("DONE")
