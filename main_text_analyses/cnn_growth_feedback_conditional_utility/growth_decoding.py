"""Does Model D's learned CNN growth feature h_g (g_CNN) actually encode the
physical growth state, or could it in principle be an arbitrary/uninformative
vector? Probes h_g (from train/val_features.npz, produced by
extract_features.py) against known physical growth quantities in three ways:

2A: h_g -> 16D growth-PCA coefficients (linear + MLP probe).
2B: h_g -> scalar quantities Ag (net area gain), mean_lambda_g1,
    mean_lambda_g2 (linear + MLP probe).
2C: PCA of h_g itself (train-fit only), correlating the first several
    principal components against the physical growth quantities above
    (Pearson + Spearman), visualized as a heatmap.

High R^2/correlation here says h_g is a faithful (if compressed) encoding of
the growth field -- necessary, but not sufficient, evidence that the NODE
actually uses it (that question is answered by intervention_sweep.py).
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
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

from probe_utils import linear_probe, mlp_probe  # noqa: E402

OUT_DIR = SCRIPT_DIR

train = np.load("train_features.npz")
val = np.load("val_features.npz")

h_train = train["cnn_final_features"]
h_val = val["cnn_final_features"]

results = {}

# ---------------------------------------------------------------------------
# 2A: h_g -> growth_pca (16D)
# ---------------------------------------------------------------------------
print("=== 2A: h_g -> growth_pca ===")
gpca_train = train["growth_pca"]
gpca_val = val["growth_pca"]

lin = linear_probe(h_train, gpca_train, h_val, gpca_val)
mlp = mlp_probe(h_train, gpca_train, h_val, gpca_val)

print(f"  linear: R2_global={lin['r2_global']:.4f}  rmse_norm={lin['rmse_normalized']:.4f}")
print(f"  MLP:    R2_global={mlp['r2_global']:.4f}  rmse_norm={mlp['rmse_normalized']:.4f}")
print(f"  MLP per-mode R2: {[f'{r:.3f}' for r in mlp['r2_per_dim']]}")

results["2A_growth_pca_decoding"] = {
    "linear": {"r2_global": lin["r2_global"], "r2_per_dim": lin["r2_per_dim"], "rmse_normalized": lin["rmse_normalized"]},
    "mlp": {"r2_global": mlp["r2_global"], "r2_per_dim": mlp["r2_per_dim"], "rmse_normalized": mlp["rmse_normalized"]},
}

# ---------------------------------------------------------------------------
# 2B: h_g -> Ag, mean_lambda_g1, mean_lambda_g2
# ---------------------------------------------------------------------------
print("=== 2B: h_g -> Ag / mean_lambda_g1 / mean_lambda_g2 ===")
targets_2b = {
    "Ag": (train["Ag"].reshape(-1, 1), val["Ag"].reshape(-1, 1)),
    "mean_lambda_g1": (train["mean_lambda_g1"].reshape(-1, 1), val["mean_lambda_g1"].reshape(-1, 1)),
    "mean_lambda_g2": (train["mean_lambda_g2"].reshape(-1, 1), val["mean_lambda_g2"].reshape(-1, 1)),
}
results["2B_scalar_decoding"] = {}
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, (yt, yv)) in zip(axes, targets_2b.items()):
    lin_s = linear_probe(h_train, yt, h_val, yv)
    mlp_s = mlp_probe(h_train, yt, h_val, yv, hidden=64)
    pred = mlp_s["pred_val_denorm"].reshape(-1)
    true = yv.reshape(-1)
    rho, _ = spearmanr(pred, true)
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    print(f"  {name}: linear R2={lin_s['r2_global']:.4f}  MLP R2={mlp_s['r2_global']:.4f}  "
          f"Spearman={rho:.4f}  RMSE(physical units)={rmse:.4f}")
    results["2B_scalar_decoding"][name] = {
        "linear_r2": lin_s["r2_global"], "mlp_r2": mlp_s["r2_global"],
        "spearman": float(rho), "rmse_physical": rmse,
    }
    ax.scatter(true, pred, s=8, alpha=0.4)
    lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel(f"true {name}")
    ax.set_ylabel(f"predicted {name} (from h_g, MLP)")
    ax.set_title(f"{name}: R2={mlp_s['r2_global']:.3f}, rho={rho:.3f}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "scalar_decoding_scatter.png"), dpi=200)
plt.savefig(os.path.join(OUT_DIR, "scalar_decoding_scatter.pdf"))
plt.close()

# ---------------------------------------------------------------------------
# 2C: PCA of h_g, correlate PCs against physical quantities
# ---------------------------------------------------------------------------
print("=== 2C: PCA of h_g, correlate PCs against physical quantities ===")
h_mean = h_train.mean(axis=0, keepdims=True)
h_std = h_train.std(axis=0, keepdims=True) + 1e-8
Hn_train = (h_train - h_mean) / h_std
Hn_val = (h_val - h_mean) / h_std

U, S, Vt = np.linalg.svd(Hn_train, full_matrices=False)
n_pcs = min(6, Vt.shape[0])
pcs_val = Hn_val @ Vt[:n_pcs].T  # (n_val, n_pcs)

phys_quantities = {
    "Ag": val["Ag"],
    "mean_lambda_g1": val["mean_lambda_g1"],
    "mean_lambda_g2": val["mean_lambda_g2"],
    "growth_anisotropy": val["growth_anisotropy"],
    "growth_pca_mode0": val["growth_pca"][:, 0],
    "time": val["time"],
    "volume": val["volume"],
}

pearson_mat = np.zeros((n_pcs, len(phys_quantities)))
spearman_mat = np.zeros((n_pcs, len(phys_quantities)))
labels = list(phys_quantities.keys())
for j, name in enumerate(labels):
    q = phys_quantities[name]
    for i in range(n_pcs):
        pearson_mat[i, j] = pearsonr(pcs_val[:, i], q)[0]
        spearman_mat[i, j] = spearmanr(pcs_val[:, i], q)[0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, mat, title in zip(axes, [pearson_mat, spearman_mat], ["Pearson", "Spearman"]):
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n_pcs))
    ax.set_yticklabels([f"PC{i+1}" for i in range(n_pcs)])
    ax.set_title(f"h_g PCs vs. physical quantities ({title})")
    plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pca_correlation_heatmap.png"), dpi=200)
plt.savefig(os.path.join(OUT_DIR, "pca_correlation_heatmap.pdf"))
plt.close()

results["2C_pca_correlation"] = {
    "explained_var_ratio": (S[:n_pcs] ** 2 / np.sum(S ** 2)).tolist(),
    "pearson": pearson_mat.tolist(),
    "spearman": spearman_mat.tolist(),
    "labels": labels,
}

with open(os.path.join(OUT_DIR, "growth_decoding_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("saved growth_decoding_results.json")
print("DONE")
