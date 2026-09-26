"""Marginal failure-probability panel, all 8 design parameters.

Cohen's d (used in step1_recover_failed_jobs.py) is a linear/
location-shift statistic and misses non-monotonic relationships --
step2b_investigate_kappa_discrepancy.py found that kappa has a real,
robust, U-shaped (non-monotonic) relationship with non-convergence that
Cohen's d cannot detect. This panel plots binned failure probability (with
Wilson 95% confidence intervals) against each of the 8 parameters, which is
the only way to see that U-shape alongside the more familiar monotonic
k1/mu effects.

Reads: all_1000_jobs_with_convergence_flag.csv (written by
step1_recover_failed_jobs.py).

Writes:
  - fig_marginal_failure_probability.{pdf,png}: the 8-panel marginal figure
  - step3_marginal_bin_data.csv: the underlying binned data (consumed by
    later steps and the four_panel_revision panel builders)
This is the marginal (1D) panel referenced in SI Section 2 / Figure 2.

Run from the repository root; all paths resolve relative to this script.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(OUT_DIR, "all_1000_jobs_with_convergence_flag.csv"))
FEATURES = ["tol", "Tcrit", "Vf", "mu", "kk1", "kk2", "kappa", "k1"]
N_BINS = 8

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                       "savefig.dpi": 600, "savefig.bbox": "tight"})

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
overall_rate = (~df["converged"]).mean()

bin_results = {}
for ax, param in zip(axes.flat, FEATURES):
    bins = pd.qcut(df[param], q=N_BINS, duplicates="drop")
    g = df.groupby(bins, observed=True).apply(
        lambda x: pd.Series({"n": len(x), "n_failed": (~x["converged"]).sum(),
                               "fail_rate": (~x["converged"]).mean(),
                               "mid": x[param].mean()})
    )
    bin_results[param] = g
    # Wilson score interval (rather than the normal approximation) for the
    # 95% CI per bin -- better calibrated than a simple SE-based interval
    # when bin counts are small or the failure rate is near 0.
    n, k = g["n"].values, g["n_failed"].values
    p = k / n
    z = 1.96
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    halfwidth = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    ax.errorbar(g["mid"], g["fail_rate"], yerr=[g["fail_rate"] - (center - halfwidth),
                                                  (center + halfwidth) - g["fail_rate"]],
                marker="o", markersize=5, linewidth=1.5, capsize=3, color="tab:red")
    ax.axhline(overall_rate, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label=f"overall rate ({overall_rate:.1%})")
    ax.set_title(param)
    ax.set_xlabel(param)
    ax.set_ylabel("P(non-convergence)")
    ax.set_ylim(-0.02, max(0.35, g["fail_rate"].max() * 1.3))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

fig.suptitle("Marginal FE non-convergence probability by parameter (8 quantile bins, Wilson 95% CI)",
             fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT_DIR, "fig_marginal_failure_probability.pdf"))
fig.savefig(os.path.join(OUT_DIR, "fig_marginal_failure_probability.png"))
plt.close(fig)
print("saved fig_marginal_failure_probability.{pdf,png}")

# Save the underlying bin data
all_bins = []
for param, g in bin_results.items():
    gg = g.reset_index(drop=True).copy()
    gg["parameter"] = param
    all_bins.append(gg)
pd.concat(all_bins, ignore_index=True).to_csv(
    os.path.join(OUT_DIR, "step3_marginal_bin_data.csv"), index=False)
print("saved step3_marginal_bin_data.csv")
print("DONE")
