"""Representation-structure analysis: having established that h_g encodes
physical growth information (growth_decoding.py) and that the NODE
actually uses it (intervention_sweep.py), this script visualizes
WHAT that representation looks like -- its low-dimensional structure and
how it relates to the mechanical latent state and design parameters.
Produces figures 6A/6B/7/8:

- 6A: PCA of h_g (train-fit), scatter colored by physical/mechanical
  quantities (net area gain, volume, time, growth anisotropy, etc.).
- 6B: representative per-sim temporal trajectories in (PC1,PC2) of h_g vs.
  (z1,z2) of the mechanical latent state, for a low-, mid-, and
  high-final-volume simulation.
- 7: pairwise-distance geometry -- do two snapshots that are close in
  mechanical-latent distance (d_z) also have similar CNN features (d_h)?
  Compared against the growth-PCA distance (d_g) as well.
- 8: parameter-correlation heatmap -- h_g's leading PCs vs. growth,
  mechanical, and design (mu) quantities.

NOTE: must be run with the working directory set to the repository root
(where the FE dataset from data/DATA_AVAILABILITY.md has been placed) --
this script does not chdir. It adds evaluation/ to sys.path relative to its
own file location.
"""
import os
import sys
import json

import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(SCRIPT_DIR, "..", "..", "evaluation")
sys.path.insert(0, EVAL_DIR)
sys.path.insert(0, SCRIPT_DIR)

import eval_model_a as ma  # noqa: E402

OUT_DIR = SCRIPT_DIR
train = np.load(os.path.join(OUT_DIR, "train_features.npz"))
val = np.load(os.path.join(OUT_DIR, "val_features.npz"))

h_train = train["cnn_final_features"]
h_val = val["cnn_final_features"]
h_mean = h_train.mean(axis=0, keepdims=True)
h_std = h_train.std(axis=0, keepdims=True) + 1e-8

U, S, Vt = np.linalg.svd((h_train - h_mean) / h_std, full_matrices=False)
pcs_val = ((h_val - h_mean) / h_std) @ Vt[:2].T  # (n_val, 2)

# ---------------------------------------------------------------------------
# 6A: PCA scatter colored by various quantities
# ---------------------------------------------------------------------------
color_by = {
    "Ag": val["Ag"], "volume": val["volume"], "time": val["time"],
    "mean_lambda_g1": val["mean_lambda_g1"], "mean_lambda_g2": val["mean_lambda_g2"],
    "growth_anisotropy": val["growth_anisotropy"],
    "growth_pca_mode0": val["growth_pca"][:, 0],
    "z1": val["modelA_inputs"][:, 0],
}
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for ax, (name, c) in zip(axes.flat, color_by.items()):
    sc = ax.scatter(pcs_val[:, 0], pcs_val[:, 1], c=c, s=6, cmap="viridis", alpha=0.7)
    ax.set_xlabel("PC1(h_g)"); ax.set_ylabel("PC2(h_g)"); ax.set_title(name)
    plt.colorbar(sc, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pca_scatter_colored.png"), dpi=200)
plt.savefig(os.path.join(OUT_DIR, "pca_scatter_colored.pdf"))
plt.close()

# ---------------------------------------------------------------------------
# 6B: representative per-sim trajectories, PC space vs z space
# ---------------------------------------------------------------------------
sids = val["sim_id"]
final_vol_by_sim = {}
for sid in np.unique(sids):
    mask = sids == sid
    final_vol_by_sim[int(sid)] = val["target_volume"][mask][0]

sorted_sims = sorted(final_vol_by_sim, key=lambda s: final_vol_by_sim[s])
rep_sims = [sorted_sims[0], sorted_sims[len(sorted_sims) // 2], sorted_sims[-1]]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for sid in rep_sims:
    mask = sids == sid
    order = np.argsort(val["snapshot_id"][mask])
    pc = pcs_val[mask][order]
    z12 = val["modelA_inputs"][mask][order][:, :2]
    axes[0].plot(pc[:, 0], pc[:, 1], "-o", markersize=3, label=f"sim {sid} (Vf={final_vol_by_sim[sid]:.0f})")
    axes[1].plot(z12[:, 0], z12[:, 1], "-o", markersize=3, label=f"sim {sid}")
axes[0].set_xlabel("PC1(h_g)"); axes[0].set_ylabel("PC2(h_g)"); axes[0].set_title("CNN-feature trajectories")
axes[1].set_xlabel("z1"); axes[1].set_ylabel("z2"); axes[1].set_title("Mechanical-latent trajectories")
for ax in axes:
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "representative_trajectories.png"), dpi=200)
plt.savefig(os.path.join(OUT_DIR, "representative_trajectories.pdf"))
plt.close()

# ---------------------------------------------------------------------------
# 7: pairwise-distance geometry, d_z vs d_h (and d_g)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
n_val = h_val.shape[0]
n_pairs = 5000
i_idx = rng.integers(0, n_val, n_pairs)
j_idx = rng.integers(0, n_val, n_pairs)
keep = i_idx != j_idx
i_idx, j_idx = i_idx[keep], j_idx[keep]

z_val = val["modelA_inputs"][:, :9]
z_mean_ = train["modelA_inputs"][:, :9].mean(axis=0, keepdims=True)
z_std_ = train["modelA_inputs"][:, :9].std(axis=0, keepdims=True) + 1e-8
zn = (z_val - z_mean_) / z_std_
hn = (h_val - h_mean) / h_std
g_val = val["growth_pca"]
g_mean_ = train["growth_pca"].mean(axis=0, keepdims=True)
g_std_ = train["growth_pca"].std(axis=0, keepdims=True) + 1e-8
gn = (g_val - g_mean_) / g_std_

d_z = np.linalg.norm(zn[i_idx] - zn[j_idx], axis=1)
d_h = np.linalg.norm(hn[i_idx] - hn[j_idx], axis=1)
d_g = np.linalg.norm(gn[i_idx] - gn[j_idx], axis=1)

pear_zh, _ = pearsonr(d_z, d_h)
spear_zh, _ = spearmanr(d_z, d_h)
pear_gh, _ = pearsonr(d_g, d_h)
spear_gh, _ = spearmanr(d_g, d_h)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(d_z, d_h, s=3, alpha=0.3)
axes[0].set_xlabel("d_z (mechanical latent distance)"); axes[0].set_ylabel("d_h (CNN feature distance)")
axes[0].set_title(f"pearson={pear_zh:.3f}  spearman={spear_zh:.3f}")
axes[1].scatter(d_g, d_h, s=3, alpha=0.3, color="tab:orange")
axes[1].set_xlabel("d_g (growth-PCA distance)"); axes[1].set_ylabel("d_h (CNN feature distance)")
axes[1].set_title(f"pearson={pear_gh:.3f}  spearman={spear_gh:.3f}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pairwise_distance.png"), dpi=200)
plt.savefig(os.path.join(OUT_DIR, "pairwise_distance.pdf"))
plt.close()

# ---------------------------------------------------------------------------
# 8: parameter-correlation heatmap (design params too)
# ---------------------------------------------------------------------------
(_, _, sim_index, time_vals, _, design_all, _) = ma.load_raw_data(9)
PARAM_NAMES = ["tol", "theta_crit", "mu", "kk1", "kk2", "kappa", "k1_fiber"]
design_by_sim = {}
for sid in np.unique(sids):
    idx = np.where(sim_index == sid)[0][0]
    design_by_sim[int(sid)] = design_all[idx]
design_val = np.array([design_by_sim[int(s)] for s in sids])

n_pcs = 6
pcs6_val = ((h_val - h_mean) / h_std) @ Vt[:n_pcs].T
quantities = {
    "Ag": val["Ag"], "mean_lambda_g1": val["mean_lambda_g1"], "mean_lambda_g2": val["mean_lambda_g2"],
    "growth_anisotropy": val["growth_anisotropy"],
    "z1": val["modelA_inputs"][:, 0], "volume": val["volume"], "e": val["modelA_inputs"][:, 9], "I": val["modelA_inputs"][:, 10],
}
for i, pname in enumerate(PARAM_NAMES):
    quantities[f"design_{pname}"] = design_val[:, i]

labels = list(quantities.keys())
spear_mat = np.zeros((n_pcs, len(labels)))
for j, name in enumerate(labels):
    for i in range(n_pcs):
        spear_mat[i, j] = spearmanr(pcs6_val[:, i], quantities[name])[0]

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(spear_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticks(range(n_pcs)); ax.set_yticklabels([f"PC{i+1}" for i in range(n_pcs)])
ax.set_title("h_g PCs vs. growth/mechanical/design quantities (Spearman)")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "parameter_correlation_heatmap.png"), dpi=200)
plt.savefig(os.path.join(OUT_DIR, "parameter_correlation_heatmap.pdf"))
plt.close()

results = {
    "6_7_pairwise_distance": {"pearson_dz_dh": float(pear_zh), "spearman_dz_dh": float(spear_zh),
                               "pearson_dg_dh": float(pear_gh), "spearman_dg_dh": float(spear_gh)},
    "8_param_correlation": {"labels": labels, "spearman_matrix": spear_mat.tolist(),
                             "explained_var_ratio_first6": (S[:n_pcs] ** 2 / np.sum(S ** 2)).tolist()},
    "representative_sims": [int(s) for s in rep_sims],
}
with open(os.path.join(OUT_DIR, "representation_structure_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("d_z vs d_h: pearson=%.3f spearman=%.3f" % (pear_zh, spear_zh))
print("d_g vs d_h: pearson=%.3f spearman=%.3f" % (pear_gh, spear_gh))
print("saved representation_structure_results.json")
print("DONE")
