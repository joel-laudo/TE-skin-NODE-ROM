"""Full validation re-evaluation for the delayed-LR-decay 560-epoch
checkpoints (Models A and D only) -- SI Table 3.

This is a training-schedule ablation, not a growth-feedback ablation: each
model is retrained for 560 total epochs with the terminal 60-epoch cosine
learning-rate decay delayed by 200 epochs relative to the reported
360-epoch recipe, to check whether the main results are sensitive to the
exact LR schedule. Produces the disp-RMSE trajectory and net-area-gain (Ag)
scatter for each of the two checkpoints (the disp-RMSE-selected best
checkpoint for that run), then runs the same paired A-vs-D bootstrap used
elsewhere in this study at this condition.

Run from the repository root:
    python evaluation/ablation_comparisons/delayeddecay_eval_and_comparison.py
"""
import os
import sys
import json
import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")  # headless backend -- must be set before any pyplot import
import matplotlib.pyplot as plt
from numba import njit, prange

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import eval_model_a as ma
import eval_model_d as md
from shared_eval_utils import add_surface_disp_rmse_to_rollout_zarr, plot_all_val_surface_disp_rmse_trajectories

R_MODES = 9
A0_CM2 = 0.25
OUT_DIR = "deliverables"

nodes_csv = "GOH_Nodes_Test_for_Visualization.csv"
elems_csv = "GOH_Elements_Test_for_Visualization.csv"

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
    """Growth on/off switch as a function of simulation time (days).

    Returns 0.0 (growth frozen) during the first 7 days of each 14-day
    cycle, and 1.0 (growth active) otherwise, matching the periodic
    growth-active/growth-inactive windows used to generate the ground-truth
    FEM growth trajectories. Feeds directly into the implicit growth-
    stretch integration below.
    """
    if (0.0 < time_ < 7.0) or (14.0 < time_ < 21.0) or (28.0 < time_ < 35.0) or \
       (42.0 < time_ < 49.0) or (56.0 < time_ < 63.0) or (70.0 < time_ < 77.0) or \
       (84.0 < time_ < 91.0) or (98.0 < time_ < 105.0):
        return 0.0
    return 1.0


_growth_gate_jit = njit(_growth_gate)


@njit(parallel=True, fastmath=True)
def integrate_growth_matched_numba(F_batch, elem_lamdag, k1, k2, theta_crit, time, dt, tol=1e-12, maxiter=20):
    """Advance the per-element anisotropic growth stretches by one time step.

    Implements the same growth law used to generate the FEM ground truth:
    each element has two in-plane growth stretches, lambda1g and lambda2g
    (`elem_lamdag[:, 0]` and `[:, 1]`; "lamdag" = lambda_g, the growth
    stretch tensor's principal values). Growth in a given direction only
    activates once the corresponding elastic stretch (lam1 or lam2,
    computed from the deformation gradient F) exceeds theta_crit times the
    current growth stretch; when active, the growth ODE
    d(lambda_g)/dt = k * (lam/lambda_g - theta_crit) is integrated
    implicitly (backward Euler + Newton iteration) for numerical
    stability, gated on/off in time by `_growth_gate`.

    F_batch: (Ne, 3, 3) deformation gradients, one per element, at the
    current time step. elem_lamdag: (Ne, 2) growth stretches from the
    previous step. Returns the updated (Ne, 2) growth stretches.
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
    """Compute the deformation gradient F for every element, for a batch of
    displacement snapshots.

    The bottom surface is a 2D (bilinear-quad) membrane embedded in 3D, so
    F is built from three directions per element: the two in-plane
    isoparametric directions (xi, eta, via the shape-function gradients
    dN_dxi/dN_deta applied to both the reference coordinates X and the
    deformed coordinates x = X + u) plus the local surface normal
    (assumed materially unstretched through the thickness). node_coords:
    (N, 3) reference node positions; elem_conn: (Ne, 4) node indices per
    element; node_u_batch: (S, N, 3) displacement snapshots. Returns
    F_batch, shape (S, Ne, 3, 3).
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
    """For every sim in a rollout zarr, compute predicted vs. true final
    net area gain (Ag).

    "Predicted" Ag comes from the NODE's predicted node displacements: the
    deformation gradient is recomputed at every rolled-out step
    (compute_F_batch) and the matched growth law is integrated forward in
    time (integrate_growth_matched_numba) starting from the true growth
    state at t=0, giving a growth-stretch trajectory consistent with what
    the predicted displacements imply. "True" Ag comes directly from the
    FEM ground-truth growth fields. Returns (sim_ids, Ag_true, Ag_pred) as
    parallel arrays.
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
        Ne_full = Gx_sim_elem.shape[1]
        half = Ne_full // 2
        # lamdag ("lambda_g"): (Ne, 2) initial growth stretches for the
        # bottom-surface elements (second half of the full element set),
        # taken from the true FEM growth field at t=0 and then advanced
        # using the NODE's predicted deformation from here on.
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
        sid_list.append(sid); ag_true_list.append(float(Ag_true)); ag_pred_list.append(float(Ag_pred))
    return np.array(sid_list), np.array(ag_true_list), np.array(ag_pred_list)


def scatter_plot(ag_true, ag_pred, title, save_path):
    """Save a true-vs-predicted net-area-gain scatter plot with a
    tolerance band, and report the fraction of sims that fall inside it.

    The tolerance band half-width at a given true value is
    max(sqrt(ag_true), 5 cm^2) -- a looser absolute tolerance for small
    area gains, tightening (in relative terms) for larger ones.
    "Capture rate" is the percentage of sims whose prediction falls within
    that band. Returns (median_abs_error, capture_rate_pct).
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
    print(f"saved {save_path}  capture_rate={capture_rate:.1f}%  median_abs_err={med:.4f}", flush=True)
    return med, capture_rate


def outlier_stats(ag_true, ag_pred):
    """Count sims whose |predicted - true| net area gain exceeds a few
    fixed thresholds (20/50/100 cm^2), plus the single worst-case error."""
    err = np.abs(ag_pred - ag_true)
    return {
        "n_err_gt_20": int(np.sum(err > 20)), "n_err_gt_50": int(np.sum(err > 50)),
        "n_err_gt_100": int(np.sum(err > 100)), "max_err": float(np.max(err)),
    }


def load_per_sim_disp_rmse(zarr_path):
    """Load the per-sim mean displacement-RMSE scalar from a rollout zarr
    (same convention used for model selection elsewhere in this study:
    per-step pointwise RMSE, averaged over time). Returns
    {sim_id: mean_disp_rmse}.
    """
    root = zarr.open_group(zarr_path, mode="r")
    sims_grp = root["simulations"]
    sim_keys = sorted(list(sims_grp.group_keys()))
    out = {}
    for sk in sim_keys:
        g = sims_grp[sk]
        sid = int(g.attrs["sim_id"])
        rmse_t = np.asarray(g["disp_err_pointwise_rmse"], dtype=np.float64)
        out[sid] = float(np.mean(rmse_t))
    return out


CONFIGS = {
    "A_delayed": dict(
        rollout_fn=ma.rollout_single_sim_vanilla_node,
        ckpt_path="checkpoints/Model_A_v2_delayeddecay560_BEST.pt",
        zarr_path=os.path.join(OUT_DIR, "A_delayeddecay560_val_rollouts_r9.zarr"),
    ),
    "D_delayed": dict(
        rollout_fn=md.rollout_single_sim_cnn_node_matched,
        ckpt_path="checkpoints/Model_D_v2_delayeddecay560_BEST.pt",
        zarr_path=os.path.join(OUT_DIR, "D_delayeddecay560_val_rollouts_r9.zarr"),
    ),
}

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    RESULTS_PATH = os.path.join(OUT_DIR, "delayeddecay560_results.json")
    all_results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)

    for key, cfg in CONFIGS.items():
        if key in all_results:
            print(f"=== {key}: already done, skipping ===", flush=True)
            continue

        print(f"=== {key}: rolling out {cfg['ckpt_path']} ===", flush=True)
        ma.save_val_rollouts_to_zarr(
            zarr_path=cfg["zarr_path"], val_sims=val_sims, rollout_fn=cfg["rollout_fn"],
            rollout_kwargs={"nodes_csv": nodes_csv, "elems_csv": elems_csv,
                             "ckpt_path_template": cfg["ckpt_path"], "make_plots": False},
            r_modes=R_MODES, model_name=key,
        )
        add_surface_disp_rmse_to_rollout_zarr(cfg["zarr_path"])

        traj_png = os.path.join(OUT_DIR, f"{key}_disp_rmse_trajectory.png")
        plot_all_val_surface_disp_rmse_trajectories(
            rollout_zarr_path=cfg["zarr_path"], use_relative_time=False, show_all_curves=False,
            alpha=0.12, linewidth=0.9, summary_linewidth=3.2, band_alpha=0.18,
            ylim=(0, 10), save_path=traj_png, save_dpi=600,
        )

        sids, ag_true, ag_pred = compute_ag_true_pred_from_zarr(cfg["zarr_path"])
        scatter_png = os.path.join(OUT_DIR, f"{key}_ag_scatter.png")
        med, capture_rate = scatter_plot(ag_true, ag_pred, f"Validation Ag: {key}", scatter_png)

        per_sim_disp = load_per_sim_disp_rmse(cfg["zarr_path"])
        disp_vals = np.array(list(per_sim_disp.values()))

        all_results[key] = {
            "ckpt": cfg["ckpt_path"],
            "disp_rmse_median": float(np.median(disp_vals)), "disp_rmse_mean": float(np.mean(disp_vals)),
            "sim_id": sids.tolist(), "Ag_true": ag_true.tolist(), "Ag_pred": ag_pred.tolist(),
            "ag_median_abs_err": float(med), "ag_mean_abs_err": float(np.mean(np.abs(ag_pred - ag_true))),
            "ag_mean_signed_err": float(np.mean(ag_pred - ag_true)), "ag_capture_rate_pct": float(capture_rate),
            **outlier_stats(ag_true, ag_pred),
            "per_sim_disp_rmse": per_sim_disp,
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2)

    print("\n=== SUMMARY (delayed-decay checkpoints) ===")
    for key, r in all_results.items():
        print(f"{key:14s}  disp_med={r['disp_rmse_median']:.4f}  Ag_med={r['ag_median_abs_err']:.4f}  "
              f"Ag_mean_signed={r['ag_mean_signed_err']:+.4f}  capture={r['ag_capture_rate_pct']:.1f}%")

    # -----------------------------------------------------------------------
    # Paired A-vs-D bootstrap at the delayed-decay condition (same scheme as
    # paired_A_vs_D_bootstrap.py: resample sim indices once per bootstrap
    # draw and apply that same resample to both models, so the CI on
    # median(D) - median(A) reflects only the A-vs-D effect, not sim-to-sim
    # difficulty variation).
    # -----------------------------------------------------------------------
    print("\n=== Paired A-vs-D bootstrap, delayed-decay condition ===")
    rng = np.random.default_rng(12345)
    N_BOOT = 5000
    per_sim_a = all_results["A_delayed"]["per_sim_disp_rmse"]
    per_sim_d = all_results["D_delayed"]["per_sim_disp_rmse"]
    common_sims = sorted(set(int(k) for k in per_sim_a.keys()) & set(int(k) for k in per_sim_d.keys()))
    vals_a = np.array([per_sim_a[str(s)] if str(s) in per_sim_a else per_sim_a[s] for s in common_sims])
    vals_d = np.array([per_sim_d[str(s)] if str(s) in per_sim_d else per_sim_d[s] for s in common_sims])
    n = len(common_sims)
    point_delta = float(np.median(vals_d) - np.median(vals_a))
    boot_deltas = np.empty(N_BOOT)
    idx_arr = np.arange(n)
    for b in range(N_BOOT):
        resample = rng.choice(idx_arr, size=n, replace=True)
        boot_deltas[b] = np.median(vals_d[resample]) - np.median(vals_a[resample])
    ci_lo, ci_hi = np.percentile(boot_deltas, [2.5, 97.5])
    bootstrap_result = {
        "n_sims": n, "median_A": float(np.median(vals_a)), "median_D": float(np.median(vals_d)),
        "point_delta_feedback": point_delta, "ci95_lo": float(ci_lo), "ci95_hi": float(ci_hi),
        "ci_excludes_zero": bool(not (ci_lo <= 0 <= ci_hi)),
    }
    print(f"delayed-decay: n={n}  median(A)={np.median(vals_a):.6f}  median(D)={np.median(vals_d):.6f}  "
          f"Delta={point_delta:+.6f}  95% CI=[{ci_lo:+.6f}, {ci_hi:+.6f}]  "
          f"excludes_zero={not (ci_lo <= 0 <= ci_hi)}")

    with open(os.path.join(OUT_DIR, "delayeddecay560_bootstrap_results.json"), "w") as f:
        json.dump(bootstrap_result, f, indent=2)

    print("\nDONE")
