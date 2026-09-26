"""Fast, batched, plotting-free periodic displacement-RMSE eval.

Used by every training/*.py script for BEST-checkpoint selection during
training (invoked as a subprocess, not imported in-process -- see the
"why a subprocess" note in each training script's
run_fast_disp_rmse_eval_subprocess). Metric, precisely defined: for each
val sim, per-step RMSE trajectory -> mean over steps -> one scalar per sim;
the model-selection metric is the MEDIAN over the 185 val sims of that
per-sim mean.

No plotting anywhere. True displacement trajectories for all 185 val sims
are loaded via ONE batched zarr fancy-index read (not per-sim), then split
in-memory -- reused across every subsequent call within a process via a
module-level cache.

Lives at the repository root (not inside evaluation/ or training/) because
every training script invokes it as a bare-relative subprocess path
(`[sys.executable, "campaign_fast_eval.py", ...]`, cwd = repository root),
matching this repo's "run every script with cwd = repo root" convention.

Imports evaluation/eval_model_{a,b,c,d}.py, one per model family (A: no
growth feedback, B: scalar growth-area feedback, C: POD-compressed
feedback, D: CNN-compressed feedback), and dispatches to the right one's
rollout function via the CLI's --rollout_fn choice
("vanilla"/"scalar_ag"/"pca"/"cnn", corresponding to A/B/C/D respectively).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation"))

import numpy as np
import zarr
import eval_model_a as ma

_TRUE_DISP_CACHE = {}  # val_sims tuple -> {sim_id: (T,N,3) float32}


def _get_true_disp_trajectories(val_sims):
    """Batched load: one zarr fancy-index read across all val sims' snapshot
    columns, split in-memory. Cached per process."""
    key = tuple(sorted(int(s) for s in val_sims))
    if key in _TRUE_DISP_CACHE:
        return _TRUE_DISP_CACHE[key]

    (U_lat_all, volume_snap_all, sim_index, time_vals, volume_SP_all, design_all, n_sims) = ma.load_raw_data(9)
    g_disp = zarr.open_group(ma.ZARR_DISP, mode="r")
    snaps = g_disp["snapshots"]

    per_sim_idx = {}
    all_idx = []
    for sid in key:
        idx = np.where(sim_index == sid)[0]
        t_s = time_vals[idx]
        order = np.argsort(t_s)
        idx_sorted = idx[order]
        per_sim_idx[sid] = idx_sorted
        all_idx.append(idx_sorted)
    all_idx_concat = np.concatenate(all_idx)

    # ONE batched read across every val sim's needed columns
    flat_all = np.asarray(snaps[:, all_idx_concat], dtype=np.float32)  # (3N, sum_T)
    N = flat_all.shape[0] // 3

    out = {}
    pos = 0
    for sid in key:
        T = per_sim_idx[sid].shape[0]
        flat_sim = flat_all[:, pos:pos + T]
        pos += T
        ux = flat_sim[0:N, :].T; uy = flat_sim[N:2 * N, :].T; uz = flat_sim[2 * N:3 * N, :].T
        out[sid] = np.ascontiguousarray(np.stack([ux, uy, uz], axis=2), dtype=np.float32)  # (T,N,3)

    _TRUE_DISP_CACHE[key] = out
    return out


def _compute_surface_disp_rmse_trajectory(u_pred, u_true):
    """Per-step RMSE between predicted and true node displacements.

    u_pred, u_true: (T, N, 3) arrays (T time steps, N surface nodes).
    Returns a length-T array: at each step, the RMSE over all nodes and
    xyz components of (u_pred - u_true).
    """
    u_pred = np.asarray(u_pred, dtype=np.float64)
    u_true = np.asarray(u_true, dtype=np.float64)
    if u_pred.shape != u_true.shape:
        raise ValueError(f"Shape mismatch: u_pred {u_pred.shape} vs u_true {u_true.shape}")
    err = u_pred - u_true
    err_sq = np.sum(err ** 2, axis=2)
    return np.sqrt(np.mean(err_sq, axis=1))


def fast_disp_rmse_eval(rollout_fn, ckpt_path, r_modes, val_sims, nodes_csv, elems_csv, **rollout_kwargs):
    """rollout_fn: one of rollout_single_sim_vanilla_node / _ag_node / _pca_node / _cnn_node
    (all share the same (r_modes, sim_id, nodes_csv, elems_csv, ckpt_path_template, make_plots) signature
    and return a dict containing 'u_pred_nodes' (T,N,3)).
    Returns (median_disp_rmse, per_sim_dict) -- no plotting, no disk writes.
    """
    true_traj = _get_true_disp_trajectories(val_sims)
    per_sim_mean_rmse = {}
    for sid in val_sims:
        roll = rollout_fn(
            r_modes=r_modes, sim_id=int(sid), nodes_csv=nodes_csv, elems_csv=elems_csv,
            ckpt_path_template=ckpt_path, make_plots=False, **rollout_kwargs,
        )
        u_pred = roll["u_pred_nodes"]
        u_true = true_traj[int(sid)][: u_pred.shape[0]]
        rmse_t = _compute_surface_disp_rmse_trajectory(u_pred, u_true)
        per_sim_mean_rmse[int(sid)] = float(np.mean(rmse_t))
    median_rmse = float(np.median(list(per_sim_mean_rmse.values())))
    return median_rmse, per_sim_mean_rmse


def _lazy_pca_rollout_fn():
    import eval_model_c
    return eval_model_c.rollout_single_sim_pca_node


def _lazy_scalar_ag_rollout_fn():
    import eval_model_b
    return eval_model_b.rollout_single_sim_ag_node_matched


def _lazy_cnn_rollout_fn():
    import eval_model_d
    return eval_model_d.rollout_single_sim_cnn_node_matched


# Model B/C/D rollout functions are imported lazily (only when actually
# requested via --rollout_fn) so that running the cheap "vanilla" (Model A)
# eval doesn't pay the import cost of the other models' modules.
_ROLLOUT_FN_REGISTRY = {
    "vanilla": lambda: ma.rollout_single_sim_vanilla_node,
    "pca": _lazy_pca_rollout_fn,
    "scalar_ag": _lazy_scalar_ag_rollout_fn,
    "cnn": _lazy_cnn_rollout_fn,
}


def _register_rollout_fn(name, fn):
    """Called by model-family-specific eval modules (B/C/D) to add their own
    rollout function under a short CLI name."""
    _ROLLOUT_FN_REGISTRY[name] = lambda: fn


if __name__ == "__main__":
    # Subprocess-isolated CLI entry point. Training scripts MUST invoke this
    # eval via a fresh subprocess, not an in-process import -- calling the
    # numba-parallel rollout function from inside a live torch training
    # process was found (empirically, reproducibly) to deadlock, almost
    # certainly an OpenMP/TBB thread-pool conflict between torch's and
    # numba's runtimes in the same process (this environment already needs
    # KMP_DUPLICATE_LIB_OK=TRUE, a sign multiple OpenMP runtimes are loaded).
    # Subprocess isolation sidesteps this entirely at a small, acceptable
    # per-call overhead (cold-cache ~15s for 185 sims).
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--rollout_fn", required=True, choices=list(_ROLLOUT_FN_REGISTRY.keys()))
    parser.add_argument("--r_modes", type=int, default=9)
    parser.add_argument("--nodes_csv", default="GOH_Nodes_Test_for_Visualization.csv")
    parser.add_argument("--elems_csv", default="GOH_Elements_Test_for_Visualization.csv")
    parser.add_argument("--out", required=True, help="path to write result JSON")
    args = parser.parse_args()

    rollout_fn = _ROLLOUT_FN_REGISTRY[args.rollout_fn]()
    val_sims = list(ma.val_sims)
    try:
        median_rmse, per_sim = fast_disp_rmse_eval(
            rollout_fn, args.ckpt, args.r_modes, val_sims, args.nodes_csv, args.elems_csv,
        )
        with open(args.out, "w") as f:
            json.dump({"median_disp_rmse": median_rmse, "per_sim": per_sim, "ok": True}, f)
    except Exception as e:
        with open(args.out, "w") as f:
            json.dump({"ok": False, "error": repr(e)}, f)
        raise
    sys.exit(0)
