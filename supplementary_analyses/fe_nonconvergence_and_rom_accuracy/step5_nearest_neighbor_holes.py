"""Does excluding the 73 non-converged simulations leave actual "holes" in
the training domain, or do the excluded points remain close to retained
(converged) neighbors?

Method: normalize all 8 raw LHS parameters to [0,1] using the FULL
1000-point design's own min/max (not just the 927 converged points' range
-- using the converged-only range would artificially pull failed points
"inward" and bias the result toward "no hole"). For each failed point,
compute nearest-neighbor distance to the 927 converged points (d_fail). For
each converged point, compute nearest-neighbor distance to the OTHER 926
converged points (d_conv) -- this is the background/self-consistency scale.
Report both distributions and the ratio median(d_fail)/median(d_conv).

A permutation test (5000 resamples) turns that descriptive ratio into an
actual hypothesis test: repeatedly draw a random 73-point subset from the
full 1000 as a fake "failed" set, compute the same nearest-neighbor ratio,
and see where the TRUE ratio falls in that null distribution (empirical
p-value). This is needed because the raw ratio alone can't say whether an
observed gap is bigger than what removing any random 73 points would
produce anyway -- the permutation test supplies that baseline.

A sanity check runs first: the same procedure on one genuinely random
73-point subset should give a ratio near 1.0 with p near 0.5, confirming
the method behaves as expected before trusting the real result.

Reads: all_1000_jobs_with_convergence_flag.csv (written by
step1_recover_failed_jobs.py).

Writes:
  - fig_nearest_neighbor_holes.{pdf,png}: nearest-neighbor distance
    distributions and the permutation-test null distribution
  - step5_nearest_neighbor_summary.csv: the summary numbers (ratio,
    p-value, etc.) reported in SI Section 2

Run from the repository root; all paths resolve relative to this script.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RNG_SEED = 20260831
N_PERM = 5000

df = pd.read_csv(os.path.join(OUT_DIR, "all_1000_jobs_with_convergence_flag.csv"))
FEATURES = ["tol", "Tcrit", "Vf", "mu", "kk1", "kk2", "kappa", "k1"]

# Normalize using the FULL 1000-point range (not converged-only)
X_raw = df[FEATURES].values
mins, maxs = X_raw.min(axis=0), X_raw.max(axis=0)
X_norm = (X_raw - mins) / (maxs - mins)

conv_mask = df["converged"].values.astype(bool)
X_conv = X_norm[conv_mask]      # 927 x 8
X_fail = X_norm[~conv_mask]     # 73 x 8


def nn_ratio(fail_idx_mask, conv_idx_mask, X):
    """median(NN distance from 'failed' points to 'converged' points) /
    median(NN distance among 'converged' points to each other)."""
    Xc = X[conv_idx_mask]
    Xf = X[fail_idx_mask]
    tree = cKDTree(Xc)
    d_fail, _ = tree.query(Xf, k=1)
    # leave-one-out NN among converged points
    d_conv, _ = tree.query(Xc, k=2)  # k=1 is self (dist 0), k=2 is true NN
    d_conv = d_conv[:, 1]
    return d_fail, d_conv, np.median(d_fail) / np.median(d_conv)


# --- Real result ---
d_fail, d_conv, real_ratio = nn_ratio(~conv_mask, conv_mask, X_norm)
print(f"Real data: median(d_fail)={np.median(d_fail):.4f}  median(d_conv)={np.median(d_conv):.4f}  "
      f"ratio={real_ratio:.4f}")
print(f"  mean(d_fail)={np.mean(d_fail):.4f}  mean(d_conv)={np.mean(d_conv):.4f}")

# --- Sanity check: one random 73-point subset standing in for "failed" ---
rng = np.random.default_rng(RNG_SEED)
sanity_idx = rng.choice(1000, size=73, replace=False)
sanity_fail_mask = np.zeros(1000, dtype=bool)
sanity_fail_mask[sanity_idx] = True
sanity_conv_mask = ~sanity_fail_mask
_, _, sanity_ratio = nn_ratio(sanity_fail_mask, sanity_conv_mask, X_norm)
print(f"\nSanity check (one random 73-point subset): ratio={sanity_ratio:.4f} (expect near 1.0)")

# --- Permutation test: null distribution of the ratio under random exclusion ---
null_ratios = np.empty(N_PERM)
for i in range(N_PERM):
    idx = rng.choice(1000, size=73, replace=False)
    fmask = np.zeros(1000, dtype=bool)
    fmask[idx] = True
    _, _, r = nn_ratio(fmask, ~fmask, X_norm)
    null_ratios[i] = r

p_value = np.mean(null_ratios >= real_ratio)
print(f"\nPermutation test ({N_PERM} resamples): null ratio mean={null_ratios.mean():.4f}, "
      f"std={null_ratios.std():.4f}")
print(f"  Real ratio={real_ratio:.4f}  ->  empirical p-value (P(null >= real)) = {p_value:.4f}")

# --- Figure: overlaid distributions + null histogram ---
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "savefig.dpi": 600, "savefig.bbox": "tight"})
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
bins = np.linspace(0, max(d_fail.max(), d_conv.max()), 30)
ax.hist(d_conv, bins=bins, alpha=0.6, color="steelblue", density=True,
        label=f"converged NN dist. (n=927, median={np.median(d_conv):.3f})")
ax.hist(d_fail, bins=bins, alpha=0.6, color="tab:red", density=True,
        label=f"non-converged NN dist. to converged set (n=73, median={np.median(d_fail):.3f})")
ax.set_xlabel("Nearest-neighbor distance (normalized parameter space, 8-D)")
ax.set_ylabel("Density")
ax.set_title("Nearest-neighbor distance distributions")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)

ax = axes[1]
ax.hist(null_ratios, bins=40, color="gray", alpha=0.7, density=True,
        label=f"null (random 73-pt exclusion, n={N_PERM})")
ax.axvline(real_ratio, color="tab:red", linewidth=2.5,
           label=f"observed ratio = {real_ratio:.3f}\n(p = {p_value:.4f})")
ax.set_xlabel("median(d_excluded) / median(d_retained-self)")
ax.set_ylabel("Density")
ax.set_title("Permutation test: is the observed ratio\nmore extreme than random exclusion?")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_nearest_neighbor_holes.pdf"))
fig.savefig(os.path.join(OUT_DIR, "fig_nearest_neighbor_holes.png"))
plt.close(fig)
print("\nsaved fig_nearest_neighbor_holes.{pdf,png}")

pd.DataFrame({
    "metric": ["median_d_fail", "median_d_conv", "ratio", "sanity_check_ratio",
               "permutation_p_value", "n_permutations"],
    "value": [np.median(d_fail), np.median(d_conv), real_ratio, sanity_ratio, p_value, N_PERM],
}).to_csv(os.path.join(OUT_DIR, "step5_nearest_neighbor_summary.csv"), index=False)
print("saved step5_nearest_neighbor_summary.csv")
print("DONE")
