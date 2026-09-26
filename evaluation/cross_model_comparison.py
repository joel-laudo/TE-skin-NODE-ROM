"""Cross-model (A/B/C/D) growth-prediction comparison.

For each of A/B/C/D: reports median/mean |Ag_pred - Ag_true| and mean signed
error on the 185 validation sims, replaying growth with the validated
integrate_growth_matched_numba throughout (independently of whatever growth
integrator each model's own rollout function used internally -- this ensures
a consistent, correct integrator is used for every model's Ag comparison,
regardless of any per-model rollout implementation detail).

Run from the repository root (`python evaluation/cross_model_comparison.py`)
with the FE dataset files (see data/DATA_AVAILABILITY.md) also placed at the
repository root -- all data paths here and in evaluation/eval_model_*.py are
plain relative paths resolved against the current working directory, and
CHECKPOINTS below is relative to the repo root too.
"""
import os
import json
import time
import numpy as np

import eval_model_a as ma

t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


R_MODES = 9
A0_CM2 = 0.25

CHECKPOINTS = {
    "A": "checkpoints/Model_A_v2_BEST.pt",
    "B": "checkpoints/Model_B_v2_BEST.pt",
    "C": "checkpoints/Model_C_v2_BEST.pt",
    "D": "checkpoints/Model_D_v2_BEST.pt",
}


def get_rollout_fn(family):
    """Look up the validation-rollout function for one model variant.

    `family` is one of "A"/"B"/"C"/"D" (open-loop NODE / scalar-Ag-feedback
    NODE / POD-feedback NODE / CNN-feedback NODE, respectively). B/C/D are
    imported lazily here to avoid pulling in every model's dependencies
    just to evaluate one of them. Returns the rollout function, which all
    models implement with the same call signature and output contract
    (see eval_one_model below).
    """
    if family == "A":
        return ma.rollout_single_sim_vanilla_node
    elif family == "B":
        import eval_model_b
        return eval_model_b.rollout_single_sim_ag_node_matched
    elif family == "C":
        import eval_model_c
        return eval_model_c.rollout_single_sim_pca_node
    elif family == "D":
        import eval_model_d
        return eval_model_d.rollout_single_sim_cnn_node_matched
    raise ValueError(family)


# ---------------------------------------------------------------------------
# Shared: mesh, F computation, matched growth integrator
# ---------------------------------------------------------------------------
node_xyz_bot, elem_conn_bot, node_ids_bot, _ = ma.load_bottom_surface_mesh_direct(
    "GOH_Nodes_Test_for_Visualization.csv", "GOH_Elements_Test_for_Visualization.csv"
)
node_xyz_bot = np.ascontiguousarray(node_xyz_bot, dtype=np.float64)
elem_conn_bot = np.ascontiguousarray(elem_conn_bot, dtype=np.int64)
dN_dxi, dN_deta = ma.shape_function_gradients(0.0, 0.0)
dN_dxi = np.ascontiguousarray(dN_dxi, dtype=np.float64)
dN_deta = np.ascontiguousarray(dN_deta, dtype=np.float64)
Ne = elem_conn_bot.shape[0]

from numba import njit, prange


def _growth_gate(time_):
    """Return 0.0 during a tissue-expander "rest" week, 1.0 otherwise.

    The FE simulations alternate 7-day growth windows with 7-day rest
    windows (days 7-14, 21-28, ... are rest), mimicking the intermittent
    inflation protocol. Growth is frozen (gate=0) during rest weeks so the
    implicit growth-integrator update below is skipped for those steps.
    """
    if (0.0 < time_ < 7.0) or (14.0 < time_ < 21.0) or (28.0 < time_ < 35.0) or \
       (42.0 < time_ < 49.0) or (56.0 < time_ < 63.0) or (70.0 < time_ < 77.0) or \
       (84.0 < time_ < 91.0) or (98.0 < time_ < 105.0):
        return 0.0
    return 1.0


_growth_gate_jit = njit(_growth_gate)


@njit(parallel=True, fastmath=True)
def integrate_growth_matched_numba(F_batch, elem_lamdag, k1, k2, theta_crit, time, dt, tol=1e-12, maxiter=20):
    """Advance the elementwise growth stretches (lambda_g1, lambda_g2) by one
    timestep, given the predicted deformation gradient F for every element.

    Solves the growth ODE dlambda_g/dt = k*(lambda/lambda_g - theta_crit)_+
    implicitly with a per-element Newton iteration (this is the "matched"
    integrator: the same one used to generate the FE ground truth, so any
    Ag error reported here reflects the NODE model's displacement error,
    not a discretization mismatch). `F_batch` is (Ne, 3, 3), `elem_lamdag`
    is (Ne, 2) growth stretches from the previous step, `k1`/`k2`/
    `theta_crit` are the per-simulation growth-law parameters, and `time`/
    `dt` select the growth gate (see _growth_gate). Returns the updated
    (Ne, 2) growth-stretch array.
    """
    Ne = F_batch.shape[0]
    out = elem_lamdag.copy()
    gate = _growth_gate_jit(time)
    for ei in prange(Ne):
        F = F_batch[ei]
        lam1 = np.sqrt(F[0, 0] ** 2 + F[1, 0] ** 2 + F[2, 0] ** 2)
        lam2 = np.sqrt(F[0, 1] ** 2 + F[1, 1] ** 2 + F[2, 1] ** 2)
        lam1g_n = out[ei, 0]; lam2g_n = out[ei, 1]
        if gate == 0.0:
            out[ei, 0] = lam1g_n; out[ei, 1] = lam2g_n
            continue
        phig1 = lam1 / lam1g_n - theta_crit
        if phig1 > 0.0:
            lam1g = lam1g_n
            for _ in range(maxiter):
                dot1 = k1 * (lam1 / lam1g - theta_crit)
                ddot1 = -k1 * lam1 / (lam1g * lam1g)
                res1 = lam1g - lam1g_n - dot1 * dt
                dres1 = 1.0 - ddot1 * dt
                lam1g = lam1g - res1 / dres1
                if abs(res1) < tol:
                    break
        else:
            lam1g = lam1g_n
        phig2 = lam2 / lam2g_n - theta_crit
        if phig2 > 0.0:
            lam2g = lam2g_n
            for _ in range(maxiter):
                dot2 = k2 * (lam2 / lam2g - theta_crit)
                ddot2 = -k2 * lam2 / (lam2g * lam2g)
                res2 = lam2g - lam2g_n - dot2 * dt
                dres2 = 1.0 - ddot2 * dt
                lam2g = lam2g - res2 / dres2
                if abs(res2) < tol:
                    break
        else:
            lam2g = lam2g_n
        out[ei, 0] = lam1g; out[ei, 1] = lam2g
    return out


@njit(parallel=True, fastmath=True)
def compute_F_batch(node_coords, elem_conn, node_u_batch, dN_dxi, dN_deta):
    """Compute the full 3x3 deformation gradient F at every element, for a
    batch of nodal-displacement snapshots on the bottom surface mesh.

    Each element is a bilinear quad; in-plane derivatives come from the
    shape-function gradients (`dN_dxi`/`dN_deta`) evaluated at the element
    center, and the through-thickness (normal) direction is added via the
    unit normal of the reference and deformed surfaces, so that F is
    invertible even though only the mid-surface displacement is known
    (a thin-shell-style reconstruction). `node_coords`/`elem_conn` are the
    reference mesh geometry and connectivity; `node_u_batch` is
    (S, N, 3) predicted nodal displacements for S snapshots. Returns
    F_batch of shape (S, Ne, 3, 3).
    """
    S = node_u_batch.shape[0]
    Ne = elem_conn.shape[0]
    F_batch = np.empty((S, Ne, 3, 3), dtype=np.float64)
    for si in prange(S):
        node_u = node_u_batch[si]
        for ei in range(Ne):
            nids = elem_conn[ei]
            X = node_coords[nids, :]
            u = node_u[nids, :]
            x = X + u
            Gxi = np.zeros(3); Geta = np.zeros(3)
            gxi = np.zeros(3); geta = np.zeros(3)
            for a in range(4):
                Gxi[0] += X[a, 0] * dN_dxi[a]; Gxi[1] += X[a, 1] * dN_dxi[a]; Gxi[2] += X[a, 2] * dN_dxi[a]
                Geta[0] += X[a, 0] * dN_deta[a]; Geta[1] += X[a, 1] * dN_deta[a]; Geta[2] += X[a, 2] * dN_deta[a]
                gxi[0] += x[a, 0] * dN_dxi[a]; gxi[1] += x[a, 1] * dN_dxi[a]; gxi[2] += x[a, 2] * dN_dxi[a]
                geta[0] += x[a, 0] * dN_deta[a]; geta[1] += x[a, 1] * dN_deta[a]; geta[2] += x[a, 2] * dN_deta[a]
            Nx = Gxi[1] * Geta[2] - Gxi[2] * Geta[1]
            Ny = Gxi[2] * Geta[0] - Gxi[0] * Geta[2]
            Nz = Gxi[0] * Geta[1] - Gxi[1] * Geta[0]
            nrm = np.sqrt(Nx * Nx + Ny * Ny + Nz * Nz) + 1e-12
            Nx /= nrm; Ny /= nrm; Nz /= nrm
            nx = gxi[1] * geta[2] - gxi[2] * geta[1]
            ny = gxi[2] * geta[0] - gxi[0] * geta[2]
            nz = gxi[0] * geta[1] - gxi[1] * geta[0]
            nrm2 = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
            nx /= nrm2; ny /= nrm2; nz /= nrm2
            G = np.empty((3, 3))
            G[0, 0] = Gxi[0]; G[1, 0] = Gxi[1]; G[2, 0] = Gxi[2]
            G[0, 1] = Geta[0]; G[1, 1] = Geta[1]; G[2, 1] = Geta[2]
            G[0, 2] = Nx; G[1, 2] = Ny; G[2, 2] = Nz
            det = (G[0, 0] * (G[1, 1] * G[2, 2] - G[1, 2] * G[2, 1])
                   - G[0, 1] * (G[1, 0] * G[2, 2] - G[1, 2] * G[2, 0])
                   + G[0, 2] * (G[1, 0] * G[2, 1] - G[1, 1] * G[2, 0]))
            Ginv = np.empty((3, 3))
            Ginv[0, 0] = (G[1, 1] * G[2, 2] - G[1, 2] * G[2, 1]) / det
            Ginv[0, 1] = -(G[0, 1] * G[2, 2] - G[0, 2] * G[2, 1]) / det
            Ginv[0, 2] = (G[0, 1] * G[1, 2] - G[0, 2] * G[1, 1]) / det
            Ginv[1, 0] = -(G[1, 0] * G[2, 2] - G[1, 2] * G[2, 0]) / det
            Ginv[1, 1] = (G[0, 0] * G[2, 2] - G[0, 2] * G[2, 0]) / det
            Ginv[1, 2] = -(G[0, 0] * G[1, 2] - G[0, 2] * G[1, 0]) / det
            Ginv[2, 0] = (G[1, 0] * G[2, 1] - G[1, 1] * G[2, 0]) / det
            Ginv[2, 1] = -(G[0, 0] * G[2, 1] - G[0, 1] * G[2, 0]) / det
            Ginv[2, 2] = (G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0]) / det
            G_dual_xi = Ginv[0, :]
            G_dual_eta = Ginv[1, :]
            F = np.empty((3, 3))
            for i in range(3):
                for j in range(3):
                    F[i, j] = gxi[i] * G_dual_xi[j] + geta[i] * G_dual_eta[j]
            F[0, 0] += nx * Nx; F[0, 1] += nx * Ny; F[0, 2] += nx * Nz
            F[1, 0] += ny * Nx; F[1, 1] += ny * Ny; F[1, 2] += ny * Nz
            F[2, 0] += nz * Nx; F[2, 1] += nz * Ny; F[2, 2] += nz * Nz
            F_batch[si, ei] = F
    return F_batch


val_sims = np.asarray(ma.val_sims, dtype=np.int64)
(U_lat_all, volume_snap_all, sim_index, time_vals, volume_SP_all, design_all, n_sims) = ma.load_raw_data(R_MODES)
H, W, _ = ma.get_bottom_grid_cache()


def eval_one_model(family, ckpt_path):
    """Evaluate one model variant's final net-area-gain (Ag) accuracy.

    For every validation simulation: run the model's NODE rollout to get
    predicted nodal displacements, reconstruct the deformation gradient F
    at each timestep (compute_F_batch), and replay the growth ODE with the
    same implicit integrator used for the FE ground truth
    (integrate_growth_matched_numba) to get the predicted final growth
    stretches -> predicted net area gain. This isolates how much of the
    Ag error comes from the NODE displacement prediction itself, since the
    growth integration step is identical across all four models.

    Returns a dict with the sample count and the median/mean/mean-signed
    |Ag_pred - Ag_true| across the 185 validation sims (or None if the
    checkpoint file is missing).
    """
    if not os.path.exists(ckpt_path):
        log(f"  [skip] {family}: checkpoint not found at {ckpt_path}")
        return None

    rollout_fn = get_rollout_fn(family)
    Ag_true_list, Ag_pred_list = [], []

    for i, sid in enumerate(val_sims):
        if i % 40 == 0:
            log(f"  {family}: sim {i}/{len(val_sims)}")
        try:
            roll = rollout_fn(
                r_modes=R_MODES, sim_id=int(sid),
                nodes_csv="GOH_Nodes_Test_for_Visualization.csv",
                elems_csv="GOH_Elements_Test_for_Visualization.csv",
                ckpt_path_template=ckpt_path, make_plots=False,
            )
        except Exception as e:
            log(f"    [warn] sim {sid} rollout failed: {repr(e)}")
            continue

        try:
            # Use u_pred_nodes directly (the one contract every eval-rollout
            # module guarantees) rather than depending on a model-specific
            # alpha/z_pred key.
            nodes_hat = roll["u_pred_nodes"].astype(np.float64)
            T = nodes_hat.shape[0]
            n_steps = T - 1
            if n_steps < 1:
                continue

            idx = np.where(sim_index == sid)[0]
            t_s = time_vals[idx]
            order = np.argsort(t_s)
            t_sim = t_s[order]

            job_id, theta_crit, k1, k2 = ma.get_growth_params_for_sim(int(sid))
            t_sim_g, Gx_sim_elem, Gy_sim_elem, _ = ma.load_elem_growth_for_sim(
                sim_id=int(sid), sim_index=sim_index, time_vals=time_vals,
                zarr_sdv_path=ma.ZARR_SDV, folder_x="snaps_SDV1", folder_y="snaps_SDV4",
            )
            # Gx_sim_elem/Gy_sim_elem stack top- and bottom-surface elements
            # together; the second half of the element axis is the bottom
            # surface, matching node_xyz_bot/elem_conn_bot used below.
            Ne_full = Gx_sim_elem.shape[1]
            half = Ne_full // 2
            lamdag = np.stack([Gx_sim_elem[0, half:], Gy_sim_elem[0, half:]], axis=1).astype(np.float64)

            F_hat_all = compute_F_batch(node_xyz_bot, elem_conn_bot, nodes_hat, dN_dxi, dN_deta)

            for k in range(n_steps):
                dt_k = float(t_sim[k + 1] - t_sim[k])
                t_k = float(t_sim[k])
                lamdag = integrate_growth_matched_numba(F_hat_all[k + 1], lamdag, k1, k2, theta_crit, t_k, dt_k)

            Ag_true = ma.compute_true_net_area_gain_final_for_sim(
                sim_id=int(sid), sim_index=sim_index, time_vals=time_vals,
                H=H, W=W, zarr_sdv_path=ma.ZARR_SDV, folder_x="snaps_SDV1", folder_y="snaps_SDV4", A0_cm2=A0_CM2,
            )
            Ag_pred = ma.compute_net_area_gain_from_lamdag_elem(lamdag, A0_cm2=A0_CM2)
        except Exception as e:
            log(f"    [warn] sim {sid} post-processing failed: {repr(e)}")
            continue

        Ag_true_list.append(Ag_true)
        Ag_pred_list.append(Ag_pred)

    Ag_true_arr = np.array(Ag_true_list)
    Ag_pred_arr = np.array(Ag_pred_list)
    err = np.abs(Ag_pred_arr - Ag_true_arr)
    return {
        "n": len(Ag_true_list),
        "median_err": float(np.median(err)),
        "mean_err": float(np.mean(err)),
        "mean_signed": float(np.mean(Ag_pred_arr - Ag_true_arr)),
    }


if __name__ == "__main__":
    RESULTS_PATH = "cross_model_comparison_results.json"
    all_results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        log(f"resuming: found existing results for {list(all_results.keys())}")

    for family, ckpt_path in CHECKPOINTS.items():
        if family in all_results and all_results[family] is not None:
            log(f"=== {family}: already done, skipping ===")
            continue
        log(f"=== {family} ({ckpt_path}) ===")
        try:
            res = eval_one_model(family, ckpt_path)
        except Exception as e:
            log(f"  [ERROR] {family} failed entirely: {repr(e)}")
            res = None
        all_results[family] = res
        # save incrementally so a later crash doesn't lose earlier models' results
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2)

    log("\n=== SUMMARY: median |Ag_pred - Ag_true| ===")
    for family, res in all_results.items():
        if res is None:
            log(f"{family:5s} SKIPPED (no ckpt)")
            continue
        log(f"{family:5s} {res['median_err']:.4f}")

    log("\nDONE. Full results in cross_model_comparison_results.json")
