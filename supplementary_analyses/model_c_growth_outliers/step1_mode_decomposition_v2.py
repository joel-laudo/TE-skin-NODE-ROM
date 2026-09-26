"""Growth-POD mode decomposition: does growth-prediction error come from
the dominant growth PCA mode, or from the higher-order "residual" modes?

The growth field predicted by Model C's NODE rollout is represented as a
16-dimensional PCA coefficient vector at each timestep (2 channels, gx and
gy, x 8 PCA modes each). This script splits that vector into:
  - the dominant subspace: mode 1 of each channel (2 of the 16
    coefficients) -- the low-order, large-scale growth pattern that
    carries most of the growth field's variance
  - the residual subspace: modes 2-8 of each channel (the other 14
    coefficients) -- higher-order, smaller-scale spatial detail that
    individually carries much less variance, but can still be amplified
    by the model's closed-loop rollout

and computes, at a given rollout fraction, R_dom = ||g_pred||/||g_true|| in
the dominant subspace and R_res = ||g_pred||/||g_true|| in the residual
subspace. R_res near 1 means the model tracks the residual-mode amplitude
correctly; R_res >> 1 means the model is amplifying high-order growth-field
structure well beyond what the true FE simulation shows. R_res is the
diagnostic used to check whether growth-outlier simulations (large final
|Ag| error) are driven by this residual-mode amplification, rather than by
the dominant growth pattern being wrong.

This is entirely post-hoc analysis of already-computed rollout outputs for
Model C's v2 checkpoint (epoch 353): the predicted elemental growth field
per rollout step (lamdag_pred_elem) and the true PCA coefficient
trajectory (growth_pca_trainonly_H60_W60_k8_val0.20_seed123.npz's
X_pca_all, aligned to each rollout via sim_index/time_vals) are both read
from existing artifacts. No new rollout is run.

Reads:
  - growth_pca_trainonly_H60_W60_k8_val0.20_seed123.npz (the growth-POD
    basis: mean_x/mean_y, components_x/components_y, and the true
    coefficient trajectories X_pca_all)
  - Design_and_Metadata.zarr (sim_index/time_vals, used to align each
    rollout's true trajectory)
  - deliverables/C_v2_val_rollouts_r9.zarr (Model C's predicted elemental
    growth field per rollout step, lamdag_pred_elem) and
    deliverables/deliverables_results.json (the final-step growth-area
    values Ag_true/Ag_pred, used to correlate mode ratio with error).
    Both are produced by running `python evaluation/generate_deliverables.py`
    from the repository root (with the FE dataset in place); run that
    first if they don't exist yet.

Writes (consumed by step2_outlier_and_crosscheck.py):
  - step1_mode_decomposition_v2_results.csv: per-simulation R_dom/R_res at
    the final rollout step and at 25% into the rollout, plus the
    final-step |Ag| error
  - step1_mode_decomposition_v2_summary.json: summary statistics (medians,
    percent of sims with ratio > 2x/5x, correlations with |Ag| error)

This is the mode-decomposition analysis behind SI Section 3 / Figure 3
(Model C growth-prediction outliers).

Run from the repository root; growth_pca_trainonly_*.npz and
Design_and_Metadata.zarr are plain relative paths resolved against the
repository root.
"""
import os
import json
import numpy as np
import pandas as pd
import zarr

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

npz = np.load("growth_pca_trainonly_H60_W60_k8_val0.20_seed123.npz")
mean_x, components_x = npz["mean_x"], npz["components_x"]  # (3600,), (8,3600)
mean_y, components_y = npz["mean_y"], npz["components_y"]
X_pca_all = npz["X_pca_all"]  # (42237, 16) true coefficients, raw scale
val_sims_npz = set(int(s) for s in npz["val_sims"])

root_design = zarr.open_group("Design_and_Metadata.zarr", mode="r")
sim_index = np.asarray(root_design["Mapping_indexes_and_metadata"]["sim_index"], dtype=np.int64)
time_vals = np.asarray(root_design["Mapping_indexes_and_metadata"]["time_vals"], dtype=np.float64)

root_roll = zarr.open_group(os.path.join("deliverables", "C_v2_val_rollouts_r9.zarr"), mode="r")
sims_grp = root_roll["simulations"]
sim_keys = sorted(list(sims_grp.group_keys()))
assert len(sim_keys) == 185

# Load the final-step growth-area (Ag) error per simulation, already
# computed by the deliverables pipeline, to correlate against the mode
# ratios below.
deliverables = json.load(open(os.path.join("deliverables", "deliverables_results.json")))
c_entry = deliverables["C_v2"]
ag_err_by_sim = {int(sid): abs(p - t) for sid, t, p in
                  zip(c_entry["sim_id"], c_entry["Ag_true"], c_entry["Ag_pred"])}

# Dominant subspace: mode-1 coefficient of each channel (index 0 = gx
# mode 1, index 8 = gy mode 1 in the concatenated 16-dim [gx modes 1-8,
# gy modes 1-8] vector). Residual subspace: everything else (modes 2-8).
DOM_IDX = np.array([0, 8])  # gx mode1, gy mode1 (0-indexed into the 16-dim vector)
RES_IDX = np.array([i for i in range(16) if i not in DOM_IDX])


def encode_pred(lamdag_elem_t):
    """Project one timestep's predicted elemental growth field
    (lamdag_elem_t: (3600,2), one (gx,gy) pair per element) onto the
    growth-POD basis, returning the 16-dim raw PCA coefficients in the
    same [gx modes 1-8, gy modes 1-8] layout as the true trajectory
    X_pca_all, so predicted and true can be compared mode-by-mode."""
    gx, gy = lamdag_elem_t[:, 0], lamdag_elem_t[:, 1]
    cx = (gx - mean_x) @ components_x.T  # (8,)
    cy = (gy - mean_y) @ components_y.T
    return np.concatenate([cx, cy])


rows = []
for sk in sim_keys:
    g = sims_grp[sk]
    sid = int(g.attrs["sim_id"])
    lamdag_pred = np.asarray(g["lamdag_pred_elem"])  # (T, 3600, 2)
    T = lamdag_pred.shape[0]

    true_idx = np.where(sim_index == sid)[0]
    order = np.argsort(time_vals[true_idx])
    true_idx_sorted = true_idx[order]
    assert len(true_idx_sorted) == T, f"sim {sid}: T mismatch pred={T} true={len(true_idx_sorted)}"
    g_true_traj = X_pca_all[true_idx_sorted]  # (T, 16)

    g_pred_traj = np.stack([encode_pred(lamdag_pred[t]) for t in range(T)], axis=0)  # (T, 16)

    def ratio_at(frac):
        """Return (R_dom, R_res): the predicted/true norm ratio in the
        dominant- and residual-mode subspaces, evaluated at the rollout
        step closest to fraction `frac` of the trajectory (frac=1.0 is the
        final step, frac=0.25 is 25% through the rollout)."""
        t_idx = min(T - 1, int(round(frac * (T - 1))))
        dom_pred = np.linalg.norm(g_pred_traj[t_idx, DOM_IDX])
        dom_true = np.linalg.norm(g_true_traj[t_idx, DOM_IDX])
        res_pred = np.linalg.norm(g_pred_traj[t_idx, RES_IDX])
        res_true = np.linalg.norm(g_true_traj[t_idx, RES_IDX])
        dom_ratio = dom_pred / dom_true if dom_true > 1e-9 else np.nan
        res_ratio = res_pred / res_true if res_true > 1e-9 else np.nan
        return dom_ratio, res_ratio

    dom_final, res_final = ratio_at(1.0)
    dom_25pct, res_25pct = ratio_at(0.25)

    rows.append(dict(sim_id=sid, T=T, dom_ratio_final=dom_final, res_ratio_final=res_final,
                       dom_ratio_25pct=dom_25pct, res_ratio_25pct=res_25pct,
                       ag_abs_err=ag_err_by_sim.get(sid, np.nan)))

df = pd.DataFrame(rows)
out_path = os.path.join(OUT_DIR, "step1_mode_decomposition_v2_results.csv")
df.to_csv(out_path, index=False)

print(f"n={len(df)}")
print(f"\nDominant mode (mode 1 of each channel), final-step ratio:")
print(f"  median={df['dom_ratio_final'].median():.3f}  "
      f"%>2x={100*(df['dom_ratio_final']>2).mean():.1f}%  "
      f"%>5x={100*(df['dom_ratio_final']>5).mean():.1f}%")
print(f"\nResidual modes (2-8 of each channel), final-step ratio:")
print(f"  median={df['res_ratio_final'].median():.3f}  "
      f"%>2x={100*(df['res_ratio_final']>2).mean():.1f}%  "
      f"%>5x={100*(df['res_ratio_final']>5).mean():.1f}%")
print(f"\nResidual modes, ratio at 25% into rollout: median={df['res_ratio_25pct'].median():.3f}  "
      f"%>2x={100*(df['res_ratio_25pct']>2).mean():.1f}%")

from scipy import stats
rho_dom, p_dom = stats.spearmanr(df["dom_ratio_final"], df["ag_abs_err"])
rho_res, p_res = stats.spearmanr(df["res_ratio_final"], df["ag_abs_err"])
pear_dom, _ = stats.pearsonr(df["dom_ratio_final"].fillna(0), df["ag_abs_err"])
pear_res, _ = stats.pearsonr(df["res_ratio_final"].fillna(0), df["ag_abs_err"])
print(f"\nCorrelation with final |Ag error|:")
print(f"  dominant-mode ratio: Spearman rho={rho_dom:.3f} (p={p_dom:.4g}), Pearson r={pear_dom:.3f}")
print(f"  residual-mode ratio: Spearman rho={rho_res:.3f} (p={p_res:.4g}), Pearson r={pear_res:.3f}")

summary = dict(
    n=len(df),
    dom_median_final=float(df["dom_ratio_final"].median()),
    dom_pct_gt2x_final=float(100*(df["dom_ratio_final"]>2).mean()),
    res_median_final=float(df["res_ratio_final"].median()),
    res_pct_gt2x_final=float(100*(df["res_ratio_final"]>2).mean()),
    res_pct_gt5x_final=float(100*(df["res_ratio_final"]>5).mean()),
    res_median_25pct=float(df["res_ratio_25pct"].median()),
    res_pct_gt2x_25pct=float(100*(df["res_ratio_25pct"]>2).mean()),
    spearman_dom_vs_ag_err=float(rho_dom), spearman_dom_p=float(p_dom),
    spearman_res_vs_ag_err=float(rho_res), spearman_res_p=float(p_res),
    pearson_dom_vs_ag_err=float(pear_dom), pearson_res_vs_ag_err=float(pear_res),
)
with open(os.path.join(OUT_DIR, "step1_mode_decomposition_v2_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nsaved {out_path}")
print("saved step1_mode_decomposition_v2_summary.json")
print("DONE")
