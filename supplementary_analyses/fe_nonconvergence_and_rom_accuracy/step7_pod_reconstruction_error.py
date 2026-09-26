"""POD reconstruction error for all 927 converged simulations.

Uses the main paper's r=9 displacement POD basis (displacements.zarr's
pod_full/{U,mean}, loaded via this repo's load_disp_decoder -- no new basis
is fit) and the true POD-projected alpha coefficients per snapshot
(load_raw_data(9)'s U_lat: the true FE displacement fields projected onto
the basis, computed once during dataset construction and reused here, not
re-derived). No trained model/NODE is involved -- this isolates the
representational (basis) error floor on its own, independent of any
downstream time-integration model.

Reconstruction: u_hat(t) = mean + U_r @ alpha_true(t). Error definition is
the same per-timestep spatial RMSE of the nodal vector-norm error used
elsewhere in the evaluation pipeline (compute_surface_disp_rmse_trajectory,
reused rather than reimplemented), averaged over the trajectory to give one
scalar per simulation.

Read-only with respect to models/basis: no retraining, no basis refit.

Reads: displacement snapshots and the r=9 POD decoder via eval_model_a.py's
load_raw_data/load_disp_decoder (evaluation/eval_model_a.py), and
compute_surface_disp_rmse_trajectory from evaluation/shared_eval_utils.py.

Writes: step7_pod_reconstruction_error.csv (per-simulation e_pod, the POD
basis-error floor referenced in SI Section 2's ROM-accuracy comparison).

Run from the repository root; data paths are plain relative paths resolved
against the current working directory.
"""
import os
import sys
import numpy as np
import pandas as pd
import zarr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SCRIPT_DIR
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "..", "evaluation"))

import eval_model_a as ma  # noqa: E402
from shared_eval_utils import compute_surface_disp_rmse_trajectory as _compute_surface_disp_rmse_trajectory  # noqa: E402

R_MODES = 9

print("Loading raw data (true POD alpha per snapshot) and the r=9 decoder...")
U_lat_all, volume_snap, sim_index, time_vals, volume_SP, design_all, n_sims = ma.load_raw_data(R_MODES)
print(f"  n_sims={n_sims} (expect 927), n_snapshots={U_lat_all.shape[0]}")
decode = ma.load_disp_decoder(R_MODES)

# Batched true raw-displacement read, following the same read pattern used
# elsewhere in the evaluation pipeline, but covering ALL 927 converged
# simulations here rather than just a validation subset.
g_disp = zarr.open_group(ma.ZARR_DISP, mode="r")
snaps = g_disp["snapshots"]

unique_sims = np.unique(sim_index)
assert len(unique_sims) == 927, f"expected 927 unique sims, got {len(unique_sims)}"

per_sim_idx = {}
all_idx = []
for sid in unique_sims:
    idx = np.where(sim_index == sid)[0]
    order = np.argsort(time_vals[idx])
    idx_sorted = idx[order]
    per_sim_idx[int(sid)] = idx_sorted
    all_idx.append(idx_sorted)
all_idx_concat = np.concatenate(all_idx)

print(f"Batched true-displacement read for all {len(unique_sims)} sims "
      f"({len(all_idx_concat)} total snapshot-timesteps)...")
flat_all = np.asarray(snaps[:, all_idx_concat], dtype=np.float32)  # (3N, sum_T)
N = flat_all.shape[0] // 3

results = []
pos = 0
for sid in unique_sims:
    sid = int(sid)
    idx_sorted = per_sim_idx[sid]
    T = idx_sorted.shape[0]
    flat_sim = flat_all[:, pos:pos + T]
    pos += T
    ux = flat_sim[0:N, :].T
    uy = flat_sim[N:2 * N, :].T
    uz = flat_sim[2 * N:3 * N, :].T
    u_true = np.stack([ux, uy, uz], axis=2)  # (T, N, 3)

    alpha_true = U_lat_all[idx_sorted]  # (T, r_modes), already true-projected & time-sorted
    u_hat = np.stack([decode(alpha_true[t]) for t in range(T)], axis=0)  # (T, N, 3)

    rmse_t = _compute_surface_disp_rmse_trajectory(u_hat, u_true)  # (T,)
    e_pod = float(np.mean(rmse_t))
    results.append({"sim_id": sid, "n_steps": T, "e_pod": e_pod,
                     "e_pod_final": float(rmse_t[-1]), "e_pod_max": float(rmse_t.max())})

assert pos == flat_all.shape[1], "did not consume all batched snapshot columns"

df_pod = pd.DataFrame(results).sort_values("sim_id").reset_index(drop=True)
out_path = os.path.join(OUT_DIR, "step7_pod_reconstruction_error.csv")
df_pod.to_csv(out_path, index=False)
print(f"\nsaved {out_path}")
print(f"e_pod summary: median={df_pod['e_pod'].median():.4f} mean={df_pod['e_pod'].mean():.4f} "
      f"min={df_pod['e_pod'].min():.4f} max={df_pod['e_pod'].max():.4f}")
print("DONE")
