"""Manuscript Figure 14 (data step): spatial extension of the fully-
decomposed 7-parameter regression (tol, theta_crit, mu, kk1, kk2, kappa,
k1_fiber) -- fit the same regression independently at every mesh node
(vectorized normal-equations solve), then rescale each parameter's
coefficient map by that parameter's own observed range, giving a fair,
per-parameter spatial effect map at both the early (t=12.6) and late
(t=68.6) checkpoints. Saves spatial_effect_maps.npz, consumed by
plot_figure14_mechanical_influence.py to produce the actual published
figure.

Requires the full FE dataset (displacements.zarr, Design_and_Metadata.zarr)
-- see data/DATA_AVAILABILITY.md. Run from the repository root, since the
`evaluation` module is imported via a path relative to this script's own
location. This script only computes and saves the coefficient maps; it does
not draw the figure itself (see plot_figure14_mechanical_influence.py).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "evaluation"))

import numpy as np
import zarr
import eval_model_a as ma

R_MODES = 9
(U_lat_all, volume_snap_all, sim_index, time_vals, volume_SP_all, design_all, n_sims) = ma.load_raw_data(R_MODES)
all_sims = np.unique(sim_index)

g_disp = zarr.open_group(ma.ZARR_DISP, mode="r")
snaps = g_disp["snapshots"]

node_xyz_bot, elem_conn_bot, node_ids_bot, _ = ma.load_bottom_surface_mesh_direct(
    "GOH_Nodes_Test_for_Visualization.csv", "GOH_Elements_Test_for_Visualization.csv")
N_nodes = node_xyz_bot.shape[0]

tol_xy = 1e-8
xn = np.round(node_xyz_bot[:, 0] / tol_xy) * tol_xy
yn = np.round(node_xyz_bot[:, 1] / tol_xy) * tol_xy
xs_n = np.unique(xn); xs_n.sort()
ys_n = np.unique(yn); ys_n.sort()
H_node, W_node = ys_n.size, xs_n.size
ix_n = np.searchsorted(xs_n, xn)
iy_n = np.searchsorted(ys_n, yn)
node_grid_idx = np.full((H_node, W_node), -1, dtype=np.int64)
node_grid_idx[iy_n, ix_n] = np.arange(N_nodes)


def to_node_grid(field_per_node):
    """Scatter a length-N_nodes 1D field onto the regular (H_node, W_node)
    pixel grid defined by node_grid_idx, so it can be shown with imshow.
    Grid cells with no corresponding mesh node (outside the tissue domain)
    are left as NaN."""
    out = np.full((H_node, W_node), np.nan)
    valid = node_grid_idx >= 0
    out[valid] = field_per_node[node_grid_idx[valid]]
    return out


# Column order of the 7 design parameters as stored in design_all / Design_and_Metadata.zarr.
PARAM_NAMES = ["tol", "theta_crit", "mu", "kk1", "kk2", "kappa", "k1_fiber"]


def get_sim_traj(sid):
    """Return (sorted snapshot indices, sorted time values) for one simulation ID."""
    idx = np.where(sim_index == sid)[0]
    t_s = time_vals[idx]
    order = np.argsort(t_s)
    return idx[order], t_s[order]


def build_records(chosen_t, min_final_sp):
    """Collect one snapshot per simulation at (approximately) time `chosen_t`,
    keeping only simulations whose final tissue expander volume (surface
    pressure proxy `volume_SP_all`) exceeds `min_final_sp` -- this excludes
    runs where the expander was never meaningfully inflated by the end of
    the simulation, which would otherwise dilute the regression at the
    early/late checkpoints. Each kept record stores its global snapshot
    index and its 7 design parameters."""
    records = []
    for sid in all_sims:
        idx_sorted, t_sorted = get_sim_traj(int(sid))
        if len(t_sorted) == 0:
            continue
        final_sp = volume_SP_all[idx_sorted[-1]]
        if final_sp <= min_final_sp:
            continue
        match = np.where(np.abs(t_sorted - chosen_t) < 0.01)[0]
        if len(match) == 0:
            continue
        global_idx = idx_sorted[match[0]]
        records.append({"sid": int(sid), "global_idx": int(global_idx),
                         "params": design_all[global_idx, :].copy()})
    return records


def spatial_regression(records, n_pairs=20000, seed=7):
    """Fit, at every mesh node independently, a linear regression of
    pairwise nodal-displacement distance on the pairwise absolute
    difference in each of the 7 design parameters. Concretely: draw
    `n_pairs` random pairs of records, regress the pairwise Euclidean
    displacement difference at each node onto the pairwise |parameter
    difference| vector (plus an intercept), via one vectorized
    normal-equations solve shared across all nodes. Returns the raw
    per-node regression coefficients (row 0 = intercept, rows 1-7 = the 7
    parameters, columns = mesh nodes) and each parameter's observed range
    across the sampled pairs -- the caller rescales |coef| by range to get
    a fair, comparable "effect size" per parameter."""
    params_mat = np.array([r["params"] for r in records])  # (n, 7)
    global_idxs = np.array([r["global_idx"] for r in records])
    X_all = np.asarray(snaps[:, global_idxs], dtype=np.float64)
    N = X_all.shape[0] // 3

    n_records = len(records)
    rng = np.random.RandomState(seed)
    pi = rng.randint(0, n_records, size=n_pairs)
    pj = rng.randint(0, n_records, size=n_pairs)
    keep = pi != pj
    pi, pj = pi[keep], pj[keep]

    dparams = np.abs(params_mat[pi] - params_mat[pj])  # (n_pairs, 7)
    X_reg = np.column_stack([np.ones(len(pi))] + [dparams[:, k] for k in range(7)])

    # Displacement snapshots are stacked as [x_1..x_N, y_1..y_N, z_1..z_N];
    # split back into per-axis blocks to form the Euclidean distance per node.
    diff = X_all[:, pi] - X_all[:, pj]
    dx = diff[0:N, :]; dy = diff[N:2*N, :]; dz = diff[2*N:3*N, :]
    Y = np.sqrt(dx**2 + dy**2 + dz**2).T  # (n_pairs, N)

    XtX_inv = np.linalg.inv(X_reg.T @ X_reg)
    coefs = XtX_inv @ (X_reg.T @ Y)  # (8, N) -- row 0 intercept, rows 1-7 = params

    ranges = dparams.max(axis=0)  # (7,)
    return coefs, ranges


print("=== early (t=12.6) ===")
records_early = build_records(12.6, 0)
coefs_early, ranges_early = spatial_regression(records_early)
print(f"n_records={len(records_early)}")

print("=== late (t=68.6) ===")
records_late = build_records(68.6, 600000)
coefs_late, ranges_late = spatial_regression(records_late)
print(f"n_records={len(records_late)}")

# rescaled per-parameter effect maps: |coef_i| * range_i, converted to node grid
effects_early = {name: to_node_grid(np.abs(coefs_early[i+1, :]) * ranges_early[i])
                  for i, name in enumerate(PARAM_NAMES)}
effects_late = {name: to_node_grid(np.abs(coefs_late[i+1, :]) * ranges_late[i])
                 for i, name in enumerate(PARAM_NAMES)}

OUT_NPZ = os.path.join(os.path.dirname(__file__), "spatial_effect_maps.npz")
np.savez(OUT_NPZ,
         **{f"early_{k}": v for k, v in effects_early.items()},
         **{f"late_{k}": v for k, v in effects_late.items()})
print(f"saved {OUT_NPZ}")

vmax_early = np.nanmax([np.nanmax(v) for v in effects_early.values()])
vmax_late = np.nanmax([np.nanmax(v) for v in effects_late.values()])
print(f"early row vmax={vmax_early:.4f} cm   late row vmax={vmax_late:.4f} cm")
print("DONE -- run plot_figure14_mechanical_influence.py to produce the published figure.")
