"""Deliverable generator: displacement-RMSE trajectory plots + Ag
true-vs-predicted scatter plots, for the reported v2 checkpoints and the
SI-cited ablation checkpoints.

This script's CONDITIONS dict previously swept over several superseded/
pre-v2 checkpoint families; none of those are part of the final manuscript, and they are not
included in this repo. This version keeps only:
  - "v2": the reported checkpoint for each of A/B/C/D (manuscript Table 3).
  - "v2_agthroughout": Model C only, SI Table 2 (Ag-loss kept active for all
    360 epochs instead of zeroed in the uniform tail stage).
  - "v2_rolloutonly": Models A/D only, SI Figure 4 (one-step loss disabled
    after the warmup epoch, rollout-only optimization for the remainder).

For every (family, condition) pair:
  1. Build a validation-rollout zarr (save_val_rollouts_to_zarr, reused
     verbatim from eval_model_a.py -- the shared exporter for every model in
     this project).
  2. Augment it with per-timestep surface displacement RMSE
     (add_surface_disp_rmse_to_rollout_zarr).
  3. Plot the median + 10th-90th percentile displacement-RMSE trajectory
     (plot_all_val_surface_disp_rmse_trajectories) -- absolute rollout time,
     not normalized, matching every other figure in this study.
  4. Ag true-vs-predicted scatter: reuse the already-written u_pred_nodes
     from the zarr (no second rollout needed) and replay growth with the
     validated matched-implicit-Newton numba integrator -- no post-hoc
     closure correction is applied anywhere in this pipeline.

Run from the repository root (`python evaluation/generate_deliverables.py`)
with the FE dataset files (see data/DATA_AVAILABILITY.md) also placed at the
repository root -- all data paths here and in evaluation/eval_model_*.py are
plain relative paths resolved against the current working directory, and the
checkpoint templates below are relative to the repo root too.
"""
import os
import json
import time
import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")  # headless backend -- must be set before any pyplot import
import matplotlib.pyplot as plt
from numba import njit, prange

import eval_model_a as ma
import eval_model_b as mb
import eval_model_c as mc
import eval_model_d as md
from shared_eval_utils import (
    add_surface_disp_rmse_to_rollout_zarr,
    plot_all_val_surface_disp_rmse_trajectories,
)

t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


R_MODES = 9
A0_CM2 = 0.25

FAMILIES = {
    "A": dict(rollout_fn=ma.rollout_single_sim_vanilla_node, prefix="Model_A"),
    "B": dict(rollout_fn=mb.rollout_single_sim_ag_node_matched, prefix="Model_B"),
    "C": dict(rollout_fn=mc.rollout_single_sim_pca_node, prefix="Model_C"),
    "D": dict(rollout_fn=md.rollout_single_sim_cnn_node_matched, prefix="Model_D"),
}

# Checkpoint templates, relative to the repository root. "{prefix}" and
# "{r_modes}" are filled in per family/condition below.
CONDITIONS = {
    "v2": "checkpoints/{prefix}_v2_BEST.pt",
}
ABLATION_CONDITIONS = {
    "A": {"v2_rolloutonly": "checkpoints/{prefix}_v2_rolloutonly_BEST.pt"},
    "C": {"v2_agthroughout": "checkpoints/{prefix}_v2_agthroughout_BEST.pt"},
    "D": {"v2_rolloutonly": "checkpoints/{prefix}_v2_rolloutonly_BEST.pt"},
}

nodes_csv = "GOH_Nodes_Test_for_Visualization.csv"
elems_csv = "GOH_Elements_Test_for_Visualization.csv"

OUT_DIR = "deliverables"
os.makedirs(OUT_DIR, exist_ok=True)

val_sims = np.asarray(ma.val_sims, dtype=np.int64)
(U_lat_all, volume_snap_all, sim_index, time_vals, volume_SP_all, design_all, n_sims) = ma.load_raw_data(R_MODES)
H, W, _ = ma.get_bottom_grid_cache()

node_xyz_bot, elem_conn_bot, node_ids_bot, _ = ma.load_bottom_surface_mesh_direct(nodes_csv, elems_csv)
node_xyz_bot = np.ascontiguousarray(node_xyz_bot, dtype=np.float64)
elem_conn_bot = np.ascontiguousarray(elem_conn_bot, dtype=np.int64)
dN_dxi, dN_deta = ma.shape_function_gradients(0.0, 0.0)
dN_dxi = np.ascontiguousarray(dN_dxi, dtype=np.float64)
dN_deta = np.ascontiguousarray(dN_deta, dtype=np.float64)
Ne = elem_conn_bot.shape[0]


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
    implicitly with a per-element Newton iteration (the same integrator
    used to generate the FE ground truth, so Ag error here reflects the
    NODE displacement error, not a discretization mismatch). `F_batch` is
    (Ne, 3, 3), `elem_lamdag` is (Ne, 2) growth stretches from the previous
    step, `k1`/`k2`/`theta_crit` are the per-simulation growth-law
    parameters, and `time`/`dt` select the growth gate (see _growth_gate).
    Returns the updated (Ne, 2) growth-stretch array.
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


def compute_ag_true_pred_from_zarr(zarr_path):
    """Reuse the u_pred_nodes already written into the rollout zarr; replay
    growth with the matched integrator (no post-hoc closure) to get Ag_pred.

    Iterates over every simulation stored in `zarr_path` (written earlier
    by ma.save_val_rollouts_to_zarr), reconstructs F at each timestep from
    the predicted displacements, and integrates the growth ODE forward
    with integrate_growth_matched_numba to get each sim's predicted final
    net area gain (Ag_pred), alongside the true value read from the FE
    ground truth. Returns three 1-D arrays of equal length: simulation
    IDs, Ag_true, and Ag_pred (one entry per successfully processed sim).
    """
    root = zarr.open_group(zarr_path, mode="r")
    sims_grp = root["simulations"]
    sim_keys = sorted(list(sims_grp.group_keys()))

    ag_true_list, ag_pred_list, sid_list = [], [], []
    for sk in sim_keys:
        g = sims_grp[sk]
        sid = int(g.attrs["sim_id"])
        nodes_hat = np.asarray(g["u_pred_nodes"], dtype=np.float64)
        t_arr = np.asarray(g["t"], dtype=np.float64)
        T, N, _ = nodes_hat.shape
        n_steps = T - 1
        if n_steps < 1:
            continue

        job_id, theta_crit, k1, k2 = ma.get_growth_params_for_sim(sid)
        t_sim_g, Gx_sim_elem, Gy_sim_elem, _ = ma.load_elem_growth_for_sim(
            sim_id=sid, sim_index=sim_index, time_vals=time_vals,
            zarr_sdv_path=ma.ZARR_SDV, folder_x="snaps_SDV1", folder_y="snaps_SDV4")
        # Gx_sim_elem/Gy_sim_elem stack top- and bottom-surface elements
        # together; the second half of the element axis is the bottom
        # surface, matching node_xyz_bot/elem_conn_bot used below.
        Ne_full = Gx_sim_elem.shape[1]
        half = Ne_full // 2
        lamdag = np.stack([Gx_sim_elem[0, half:], Gy_sim_elem[0, half:]], axis=1).astype(np.float64)

        F_hat_all = compute_F_batch(node_xyz_bot, elem_conn_bot, nodes_hat, dN_dxi, dN_deta)
        for k in range(n_steps):
            dt_k = float(t_arr[k + 1] - t_arr[k])
            t_k = float(t_arr[k])
            lamdag = integrate_growth_matched_numba(F_hat_all[k + 1], lamdag, k1, k2, theta_crit, t_k, dt_k)

        Ag_true = ma.compute_true_net_area_gain_final_for_sim(
            sim_id=sid, sim_index=sim_index, time_vals=time_vals,
            H=H, W=W, zarr_sdv_path=ma.ZARR_SDV, folder_x="snaps_SDV1", folder_y="snaps_SDV4", A0_cm2=A0_CM2)
        Ag_pred = ma.compute_net_area_gain_from_lamdag_elem(lamdag, A0_cm2=A0_CM2)

        sid_list.append(sid)
        ag_true_list.append(float(Ag_true))
        ag_pred_list.append(float(Ag_pred))

    return np.array(sid_list), np.array(ag_true_list), np.array(ag_pred_list)


def scatter_plot(ag_true, ag_pred, title, save_path):
    """Save a true-vs-predicted scatter plot of final net area gain (Ag).

    Draws the y=x identity line plus a shaded "capture" band of half-width
    tol = max(sqrt(Ag_true), 5 cm^2) (a heuristic tolerance that grows
    with the true area gain, so it does not unfairly penalize larger
    expansions with a fixed absolute threshold) and reports the fraction
    of validation sims falling inside that band as the "capture rate".
    `ag_true`/`ag_pred` are 1-D arrays of final Ag values across
    validation sims; the figure is written to `save_path`.
    """
    tol = np.maximum(np.sqrt(ag_true), 5.0)
    within = np.abs(ag_pred - ag_true) <= tol
    capture_rate = 100 * within.mean()

    xmax = max(ag_true.max(), ag_pred.max()) * 1.05
    xs = np.linspace(0, xmax, 400)
    tol_line = np.maximum(np.sqrt(xs), 5.0)

    plt.figure(figsize=(7, 7))
    plt.plot(xs, xs, "b--", linewidth=1.5, zorder=2)
    plt.plot(xs, xs + tol_line, "r-", linewidth=1.5, zorder=2)
    plt.plot(xs, xs - tol_line, "r-", linewidth=1.5, zorder=2)
    plt.fill_between(xs, xs - tol_line, xs + tol_line, color="lightgreen", alpha=0.4, zorder=1)
    plt.scatter(ag_true, ag_pred, s=60, alpha=0.75, zorder=3)
    plt.xlim(0, xmax); plt.ylim(0, xmax)
    plt.xlabel("True final net area gain [cm$^2$]")
    plt.ylabel("Pred final net area gain [cm$^2$]")
    plt.title(f"{title}\nr=9 | n={len(ag_true)} | tolerance = max(sqrt(A_true), 5 cm$^2$) | "
              f"capture rate = {capture_rate:.1f}%")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    med = np.median(np.abs(ag_pred - ag_true))
    log(f"saved {save_path}  capture_rate={capture_rate:.1f}%  median_abs_err={med:.4f}")


if __name__ == "__main__":
    RESULTS_PATH = os.path.join(OUT_DIR, "deliverables_results.json")
    all_results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        log(f"resuming: found existing results for {list(all_results.keys())}")

    for fam, fam_spec in FAMILIES.items():
        conditions_for_fam = {**CONDITIONS, **ABLATION_CONDITIONS.get(fam, {})}
        for cond, tmpl in conditions_for_fam.items():
            key = f"{fam}_{cond}"
            ckpt_template = tmpl.format(prefix=fam_spec["prefix"])
            ckpt_path_check = ckpt_template.format(r_modes=R_MODES)
            if not os.path.exists(ckpt_path_check):
                log(f"=== {key}: checkpoint not found ({ckpt_path_check}), skipping ===")
                continue
            if key in all_results:
                log(f"=== {key}: already done, skipping ===")
                continue

            log(f"=== {key}: ckpt={ckpt_path_check} ===")
            zarr_path = os.path.join(OUT_DIR, f"{key}_val_rollouts_r9.zarr")

            rollout_kwargs = {
                "nodes_csv": nodes_csv,
                "elems_csv": elems_csv,
                "ckpt_path_template": ckpt_template,
                "make_plots": False,
            }
            # Not every model's rollout function accepts a `max_steps` cap
            # (some default to a shorter horizon otherwise); pass
            # max_steps=None explicitly, but only for the functions that
            # actually declare that parameter, so we still get the full
            # simulation horizon for every model.
            try:
                import inspect
                sig = inspect.signature(fam_spec["rollout_fn"])
                if "max_steps" in sig.parameters:
                    rollout_kwargs["max_steps"] = None
            except (TypeError, ValueError):
                pass

            ma.save_val_rollouts_to_zarr(
                zarr_path=zarr_path,
                val_sims=val_sims,
                rollout_fn=fam_spec["rollout_fn"],
                rollout_kwargs=rollout_kwargs,
                r_modes=R_MODES,
                model_name=key,
            )
            add_surface_disp_rmse_to_rollout_zarr(zarr_path)

            traj_png = os.path.join(OUT_DIR, f"{key}_disp_rmse_trajectory.png")
            plot_all_val_surface_disp_rmse_trajectories(
                rollout_zarr_path=zarr_path,
                use_relative_time=False,
                show_all_curves=False,
                alpha=0.12,
                linewidth=0.9,
                summary_linewidth=3.2,
                band_alpha=0.18,
                ylim=(0, 10),
                save_path=traj_png,
                save_dpi=600,
            )

            sids, ag_true, ag_pred = compute_ag_true_pred_from_zarr(zarr_path)
            scatter_png = os.path.join(OUT_DIR, f"{key}_ag_scatter.png")
            scatter_plot(
                ag_true, ag_pred,
                f"Validation: Final net area gain (true vs pred) | Model {fam} ({cond})",
                scatter_png,
            )

            all_results[key] = {
                "ckpt": ckpt_path_check,
                "sim_id": sids.tolist(),
                "Ag_true": ag_true.tolist(),
                "Ag_pred": ag_pred.tolist(),
                "median_abs_err": float(np.median(np.abs(ag_pred - ag_true))),
                "mean_abs_err": float(np.mean(np.abs(ag_pred - ag_true))),
                "mean_signed_err": float(np.mean(ag_pred - ag_true)),
            }
            with open(RESULTS_PATH, "w") as f:
                json.dump(all_results, f, indent=2)
            log(f"{key}: median|err|={all_results[key]['median_abs_err']:.4f}  "
                f"mean|err|={all_results[key]['mean_abs_err']:.4f}")

    log("\n=== SUMMARY ===")
    for key, r in all_results.items():
        log(f"{key:24s}  median={r['median_abs_err']:.4f}  mean={r['mean_abs_err']:.4f}  "
            f"mean_signed={r['mean_signed_err']:+.4f}")

    log("\nDONE")
