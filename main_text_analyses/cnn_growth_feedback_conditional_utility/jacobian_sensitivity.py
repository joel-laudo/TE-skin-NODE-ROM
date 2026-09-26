"""Local sensitivity analysis: at a given rollout state, how strongly does
a small change in h_g change the NODE's predicted velocity dz/dt? Computes
the Jacobian J_h = d(dz)/d(h_g) via autodiff on the loaded VelocityNet
submodule. This is cheap -- no rollout needed, just a single forward+backward
pass per sample -- using already-extracted (x_base_raw, h_g) pairs from the
validation feature dataset, renormalized to the checkpoint's own internal
z-score convention (matching what the network actually consumes at
inference).

Each row of J_h (one row per predicted state dimension) says how much that
dimension's predicted rate of change would shift per unit change in each
h_g component, at that specific state -- a purely LOCAL, linearized
measure. It is a useful corroborating diagnostic (e.g. large ||J_h|| near
active growth vs. small ||J_h|| when growth is gated off), but it is not
proof of global importance: a large local gradient does not guarantee the
network's behavior would change much under a large, realistic substitution,
and a small one does not rule it out. The intervention sweep in
intervention_sweep.py, which actually substitutes different h_g
values into full rollouts and measures the resulting error, is the
decisive evidence for whether the NODE uses h_g; this script is secondary.

NOTE: must be run with the working directory set to the repository root
(where the FE dataset from data/DATA_AVAILABILITY.md has been placed) --
this script does not chdir. It adds evaluation/ to sys.path relative to its
own file location.
"""
import os
import sys
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(SCRIPT_DIR, "..", "..", "evaluation")
sys.path.insert(0, EVAL_DIR)
sys.path.insert(0, SCRIPT_DIR)

import shared_eval_utils as md  # noqa: E402

D_CKPT = "checkpoints/Model_D_v2_BEST.pt"
OUT_DIR = SCRIPT_DIR

ckpt = torch.load(D_CKPT, map_location="cpu")
model = md.build_cnn_node_from_ckpt(ckpt, torch.device("cpu"))
model.eval()

z_mean = np.asarray(ckpt["z_mean"], dtype=np.float32).reshape(-1)
z_std = np.asarray(ckpt["z_std"], dtype=np.float32).reshape(-1)
sp_mean = float(np.asarray(ckpt["sp_mean"]).reshape(-1)[0])
sp_std = float(np.asarray(ckpt["sp_std"]).reshape(-1)[0])
e_mean = float(np.asarray(ckpt["e_mean"]).reshape(-1)[0])
e_std = float(np.asarray(ckpt["e_std"]).reshape(-1)[0])
I_mean = float(np.asarray(ckpt["I_mean"]).reshape(-1)[0])
I_std = float(np.asarray(ckpt["I_std"]).reshape(-1)[0])
design_mean = np.asarray(ckpt["design_mean"], dtype=np.float32).reshape(-1)
design_std = np.asarray(ckpt["design_std"], dtype=np.float32).reshape(-1)

val = np.load(os.path.join(OUT_DIR, "val_features.npz"))
modelA_inputs = val["modelA_inputs"]  # raw: [z(10), e, I, sp, design(7)]
n_sample = min(2000, modelA_inputs.shape[0])
rng = np.random.default_rng(0)
sample_idx = rng.choice(modelA_inputs.shape[0], n_sample, replace=False)

jac_norms = np.zeros(n_sample)
per_feature_sens = np.zeros((n_sample, ckpt["model_state_dict"]["growth_encoder.head.1.bias"].numel()))

for i, row_idx in enumerate(sample_idx):
    row = modelA_inputs[row_idx]
    z_raw, e_raw, I_raw, sp_raw, design_raw = row[:10], row[10], row[11], row[12], row[13:20]
    z_norm = (z_raw - z_mean) / z_std
    e_norm = (e_raw - e_mean) / e_std
    I_norm = (I_raw - I_mean) / I_std
    sp_norm = (sp_raw - sp_mean) / sp_std
    design_norm = (design_raw - design_mean) / design_std

    x_base = torch.tensor(
        np.concatenate([z_norm, [e_norm], [I_norm], [sp_norm], design_norm]).astype(np.float32)
    ).view(1, -1)
    h_true = torch.tensor(val["cnn_final_features"][row_idx], dtype=torch.float32).view(1, -1)
    h_true.requires_grad_(True)

    x_full = torch.cat([x_base, h_true], dim=-1)
    dz = model.net(x_full)  # (1, D_state)

    # J[d, k] = d(dz_d)/d(h_true_k): how much the predicted rate of change
    # of state dimension d shifts per unit change in growth-feature
    # component k, evaluated at this exact (x_base, h_true) point.
    J = torch.zeros(dz.shape[1], h_true.shape[1])
    for d in range(dz.shape[1]):
        grad = torch.autograd.grad(dz[0, d], h_true, retain_graph=True)[0]
        J[d] = grad[0]

    jac_norms[i] = float(torch.norm(J, p="fro"))
    per_feature_sens[i] = torch.norm(J, dim=0).detach().numpy()  # per-h_g-feature sensitivity

print(f"Jacobian Frobenius norm: mean={jac_norms.mean():.4f}  median={np.median(jac_norms):.4f}  "
      f"std={jac_norms.std():.4f}")

time_sample = val["time"][sample_idx]
ag_sample = val["Ag"][sample_idx]
vol_sample = val["volume"][sample_idx]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].scatter(time_sample, jac_norms, s=4, alpha=0.4)
axes[0].set_xlabel("time"); axes[0].set_ylabel("||J_h||_F")
axes[1].scatter(ag_sample, jac_norms, s=4, alpha=0.4)
axes[1].set_xlabel("Ag"); axes[1].set_ylabel("||J_h||_F")
axes[2].scatter(vol_sample, jac_norms, s=4, alpha=0.4)
axes[2].set_xlabel("volume"); axes[2].set_ylabel("||J_h||_F")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "jacobian_norm_vs_context.png"), dpi=200)
plt.savefig(os.path.join(OUT_DIR, "jacobian_norm_vs_context.pdf"))
plt.close()

mean_sens_per_feature = per_feature_sens.mean(axis=0)
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(len(mean_sens_per_feature)), mean_sens_per_feature)
ax.set_xlabel("h_g feature index"); ax.set_ylabel("mean sensitivity ||d(dz)/d(h_k)||")
ax.set_title("Per-feature NODE sensitivity to h_g")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "per_feature_sensitivity.png"), dpi=200)
plt.savefig(os.path.join(OUT_DIR, "per_feature_sensitivity.pdf"))
plt.close()

results = {
    "jacobian_fro_norm_mean": float(jac_norms.mean()),
    "jacobian_fro_norm_median": float(np.median(jac_norms)),
    "jacobian_fro_norm_std": float(jac_norms.std()),
    "per_feature_mean_sensitivity": mean_sens_per_feature.tolist(),
    "n_sampled": int(n_sample),
    "caveat": "Local sensitivity only -- not proof of global importance. The intervention-sweep results are the decisive evidence for whether the NODE uses h_g.",
}
with open(os.path.join(OUT_DIR, "jacobian_sensitivity_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("saved jacobian_sensitivity_results.json")
print("DONE")
