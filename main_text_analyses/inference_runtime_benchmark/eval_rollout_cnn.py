"""Standalone eval-time rollout module for Model D (the CNN growth-feedback
NODE).

Why this file exists: Model D feeds its own CNN-encoded growth field back into
the velocity net at every rollout step (unlike Model A's open-loop network,
where growth integration only feeds a terminal Ag loss and never touches the
displacement trajectory). Model D's training uses a UMAT-matched
implicit-Newton growth integrator (integrate_growth_matched_torch, mirrored
here in numba form as integrate_growth_matched_numba/integrate_growth_matched
for eval-time use). The project's existing eval infra for Model D
(model_d_stabilized_eval_lib.rollout_single_sim_cnn_node) instead calls the
older explicit-Euler, UMAT-mismatched integrate_growth_fast/integrate_growth_numba.
Left uncorrected, that would create an eval-vs-train growth-field mismatch
that open-loop Model A/B/C's eval never has to worry about (their eval-time
growth integrator choice doesn't affect u_pred_nodes at all). This module
reuses everything else from model_d_stabilized_eval_lib.py (checkpoint
loading, mesh, POD decoder, growth-param lookup) but swaps ONLY the internal
per-step growth-integration call for the matched numba implementation below
(validated against the torch reference implementation to 7.6e-13 max abs
diff).

Performs NO top-level execution / IO on import (safe to import). Everything
below `# --- CLI / standalone sanity check ---` runs only when this file is
executed directly (`python eval_rollout_cnn.py`).

Exposes: rollout_single_sim_cnn_node_matched(r_modes, sim_id, nodes_csv,
elems_csv, max_steps=None, ckpt_path_template=..., make_plots=False)
-> dict with key "u_pred_nodes" (T,N,3) float32, same convention as
eval_ablation_model_a_base_closureMLP.rollout_single_sim_vanilla_node.
"""

import numpy as np
import torch
from numba import njit, prange

from model_d_stabilized_eval_lib import (
    ZARR_DISP,
    ZARR_SDV,
    DISP_POD_GROUP,
    DISP_POD_U_NAME,
    DISP_POD_MEAN_NAME,
    A0_CM2,
    get_bottom_grid_cache,
    load_bottom_surface_mesh_direct,
    load_raw_data,
    load_disp_decoder,
    get_growth_params_for_sim,
    load_elem_growth_for_sim,
    normalize_growth_grid,
    compute_net_area_gain_from_lamdag_elem,
    build_cnn_node_from_ckpt,
    shape_function_gradients,
)


# ===============================
# UMAT-matched growth integrator: a numba re-implementation of the
# training-time torch integrator (integrate_growth_matched_torch). Since
# eval only needs a forward pass (no backprop), the already-validated numba
# version is used directly here for speed, instead of the torch/autograd
# version training uses.
# ===============================
def _growth_gate(time):
    if (0.0 < time < 7.0) or (14.0 < time < 21.0) or (28.0 < time < 35.0) or \
       (42.0 < time < 49.0) or (56.0 < time < 63.0) or (70.0 < time < 77.0) or \
       (84.0 < time < 91.0) or (98.0 < time < 105.0):
        return 0.0
    return 1.0
_growth_gate_jit = njit(_growth_gate)


@njit(parallel=True, fastmath=True)
def integrate_growth_matched_numba(F_batch, elem_lamdag, k1, k2, theta_crit, time, dt, tol=1e-12, maxiter=20):
    Ne = F_batch.shape[0]
    out = elem_lamdag.copy()
    gate = _growth_gate_jit(time)
    for ei in prange(Ne):
        F = F_batch[ei]
        lam1 = np.sqrt(F[0,0]**2 + F[1,0]**2 + F[2,0]**2)
        lam2 = np.sqrt(F[0,1]**2 + F[1,1]**2 + F[2,1]**2)
        lam1g_n = out[ei,0]; lam2g_n = out[ei,1]
        if gate == 0.0:
            out[ei,0] = lam1g_n; out[ei,1] = lam2g_n; continue
        phig1 = lam1/lam1g_n - theta_crit
        if phig1 > 0.0:
            lam1g = lam1g_n
            for _ in range(maxiter):
                dot1 = k1*(lam1/lam1g - theta_crit)
                ddot1 = -k1*lam1/(lam1g*lam1g)
                res1 = lam1g - lam1g_n - dot1*dt
                dres1 = 1.0 - ddot1*dt
                lam1g = lam1g - res1/dres1
                if abs(res1) < tol: break
        else:
            lam1g = lam1g_n
        phig2 = lam2/lam2g_n - theta_crit
        if phig2 > 0.0:
            lam2g = lam2g_n
            for _ in range(maxiter):
                dot2 = k2*(lam2/lam2g - theta_crit)
                ddot2 = -k2*lam2/(lam2g*lam2g)
                res2 = lam2g - lam2g_n - dot2*dt
                dres2 = 1.0 - ddot2*dt
                lam2g = lam2g - res2/dres2
                if abs(res2) < tol: break
        else:
            lam2g = lam2g_n
        out[ei,0] = lam1g; out[ei,1] = lam2g
    return out


# ===============================
# Deformation gradient F (Ne,3,3), computed the same way it already is inside
# rollout_single_sim_cnn_node's OLD integrate_growth_numba call (bilinear Q4
# shape-function gradients at the element center xi=eta=0). Vectorized numpy
# (not numba) -- Ne~3600 elements, called once per rollout step, negligible
# cost next to the CNN forward pass and Newton growth solve.
# ===============================
def compute_F_batch_numpy(node_coords, elem_conn, node_u, xi: float = 0.0, eta: float = 0.0, eps: float = 1e-12):
    node_coords = np.asarray(node_coords, dtype=np.float64)
    node_u = np.asarray(node_u, dtype=np.float64)
    elem_conn = np.asarray(elem_conn, dtype=np.int64)

    X_e = node_coords[elem_conn]   # (Ne,4,3)
    u_e = node_u[elem_conn]        # (Ne,4,3)
    x_e = X_e + u_e                # (Ne,4,3)

    dN_dxi, dN_deta = shape_function_gradients(xi, eta)
    dN_dxi = np.asarray(dN_dxi, dtype=np.float64).reshape(4)
    dN_deta = np.asarray(dN_deta, dtype=np.float64).reshape(4)

    Gxi  = np.einsum('ead,a->ed', X_e, dN_dxi)    # (Ne,3)
    Geta = np.einsum('ead,a->ed', X_e, dN_deta)   # (Ne,3)
    cross_ref = np.cross(Gxi, Geta)               # (Ne,3)
    N = cross_ref / (np.linalg.norm(cross_ref, axis=1, keepdims=True) + eps)

    G = np.stack([Gxi, Geta, N], axis=2)          # (Ne,3,3), columns = Gxi,Geta,N
    G_inv = np.linalg.inv(G)                      # (Ne,3,3)
    G_dual_xi  = G_inv[:, 0, :]
    G_dual_eta = G_inv[:, 1, :]

    gxi  = np.einsum('ead,a->ed', x_e, dN_dxi)
    geta = np.einsum('ead,a->ed', x_e, dN_deta)
    cross_def = np.cross(gxi, geta)
    n = cross_def / (np.linalg.norm(cross_def, axis=1, keepdims=True) + eps)

    F = (gxi[:, :, None] * G_dual_xi[:, None, :]
         + geta[:, :, None] * G_dual_eta[:, None, :]
         + n[:, :, None] * N[:, None, :])
    return np.ascontiguousarray(F, dtype=np.float64)  # (Ne,3,3)


def integrate_growth_matched(node_coords, elem_conn, node_u, elem_lamdag, k1, k2, theta_crit, time, dt):
    """Drop-in replacement for model_d_stabilized_eval_lib.integrate_growth_fast,
    but UMAT-matched (implicit Newton) instead of the old explicit-Euler
    version. Same call convention (raw numpy in, raw numpy out)."""
    F_batch = compute_F_batch_numpy(node_coords, elem_conn, node_u)
    return integrate_growth_matched_numba(
        F_batch,
        np.asarray(elem_lamdag, dtype=np.float64),
        float(k1), float(k2), float(theta_crit), float(time), float(dt),
    )


# ===============================
# Core rollout (growth-feedback closed loop), matched-integrator version.
# Adapted from model_d_stabilized_eval_lib.rollout_single_sim_cnn_node; the
# ONLY functional change is the growth-integration call inside the step
# loop (integrate_growth_fast -> integrate_growth_matched). Signature matches
# eval_ablation_model_a_base_closureMLP.rollout_single_sim_vanilla_node's
# convention, so callers can treat both models' rollout functions
# interchangeably.
# ===============================
def rollout_single_sim_cnn_node_matched(
    r_modes: int,
    sim_id: int,
    nodes_csv: str,
    elems_csv: str,
    max_steps=None,
    ckpt_path_template: str = "FINAL_models/Model_D_ablation_fullrollout_v2/Model_D_CNN_r{r}_BEST.pt",
    make_plots: bool = False,
):
    """
    Rollout with growth feedback (Euler):
      z_{k+1} = z_k + dt_k * f(x_base_k, G_k_norm)

    Growth feedback loop (UMAT-matched):
      z_k -> decode u_k -> integrate growth (elem, implicit Newton) ->
      rasterize (2,H,W) -> normalize -> CNN -> NODE

    ckpt_path_template is .format(r=r_modes)'d; a literal full path (no '{r}'
    placeholder) works fine too, as long as it contains no stray curly braces.

    make_plots is accepted for interface parity with other models' rollout
    functions (e.g. Model A's); this function does no plotting regardless.

    Returns a dict containing "u_pred_nodes" (T,N,3) float32, plus the same
    auxiliary fields as rollout_single_sim_cnn_node (z_true/z_pred, vol
    trajectories, predicted growth grids/lamdag, final net area gain, etc.).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ckpt_path_template.format(r=r_modes, r_modes=r_modes)
    ckpt = torch.load(ckpt_path, map_location=device)

    D_state = int(ckpt["latent_state_dim"])

    volume_idx = int(r_modes)
    if D_state <= volume_idx:
        raise ValueError(f"D_state={D_state} too small for volume_idx=r_modes={r_modes}")

    z_mean = np.asarray(ckpt["z_mean"], dtype=np.float32).reshape(-1)
    z_std  = np.asarray(ckpt["z_std"], dtype=np.float32).reshape(-1)
    if z_mean.shape[0] != D_state or z_std.shape[0] != D_state:
        raise ValueError(f"z_mean/z_std length mismatch with D_state={D_state}: {z_mean.shape}, {z_std.shape}")

    sp_mean = float(np.asarray(ckpt["sp_mean"], dtype=np.float32).reshape(-1)[0])
    sp_std  = float(np.asarray(ckpt["sp_std"], dtype=np.float32).reshape(-1)[0])

    design_mean = np.asarray(ckpt["design_mean"], dtype=np.float32).reshape(1, -1)
    design_std  = np.asarray(ckpt["design_std"], dtype=np.float32).reshape(1, -1)

    e_mean = float(np.asarray(ckpt["e_mean"], dtype=np.float32).reshape(-1)[0])
    e_std  = float(np.asarray(ckpt["e_std"], dtype=np.float32).reshape(-1)[0])

    I_mean = float(np.asarray(ckpt["I_mean"], dtype=np.float32).reshape(-1)[0])
    I_std  = float(np.asarray(ckpt["I_std"], dtype=np.float32).reshape(-1)[0])

    g_mean = np.asarray(ckpt["g_mean"], dtype=np.float32).reshape(-1)
    g_std  = np.asarray(ckpt["g_std"], dtype=np.float32).reshape(-1)
    if g_mean.shape[0] != 2 or g_std.shape[0] != 2:
        raise ValueError(f"Expected g_mean/g_std shape (2,), got {g_mean.shape}, {g_std.shape}")

    model = build_cnn_node_from_ckpt(ckpt, device)
    model.eval()

    (U_lat,
     volume_snap,
     sim_index, time_vals,
     volume_SP, design_all,
     n_sims) = load_raw_data(r_modes)

    Z_state_all = np.concatenate([U_lat, volume_snap.reshape(-1, 1)], axis=1).astype(np.float32)

    idx = np.where(sim_index == sim_id)[0]
    if idx.size < 2:
        raise ValueError(f"Sim {sim_id} has too few frames ({idx.size}).")

    t_s   = time_vals[idx]
    order = np.argsort(t_s)
    idx_sorted = idx[order]
    t_sim = t_s[order]

    Z_true_sim = Z_state_all[idx_sorted, :]
    SP_sim     = volume_SP[idx_sorted]
    design_sim = design_all[idx_sorted, :]

    T_true  = Z_true_sim.shape[0]
    n_steps = (T_true - 1) if max_steps is None else min(max_steps, T_true - 1)

    design_raw    = design_sim[0:1, :].astype(np.float32)
    design_norm   = (design_raw - design_mean) / design_std
    design_tensor = torch.from_numpy(design_norm.astype(np.float32)).to(device)

    node_xyz_bot, elem_conn_bot, node_ids_bot, elem_node_ids_bot = load_bottom_surface_mesh_direct(nodes_csv, elems_csv)

    H, W, _ = get_bottom_grid_cache()
    if H * W != elem_conn_bot.shape[0]:
        raise ValueError(f"Expected bottom elems = H*W={H*W}, got {elem_conn_bot.shape[0]}")

    decode_u = load_disp_decoder(
        r_modes,
        zarr_disp_path=ZARR_DISP,
        pod_group=DISP_POD_GROUP,
        pod_u_name=DISP_POD_U_NAME,
        pod_mean_name=DISP_POD_MEAN_NAME,
        verbose=False,
    )

    job_id, theta_crit, k1, k2 = get_growth_params_for_sim(sim_id)

    t_sim_g, Gx_sim_elem, Gy_sim_elem, _ = load_elem_growth_for_sim(
        sim_id=sim_id,
        sim_index=sim_index,
        time_vals=time_vals,
        zarr_sdv_path=ZARR_SDV,
        folder_x="snaps_SDV1",
        folder_y="snaps_SDV4",
    )

    Ne_full = Gx_sim_elem.shape[1]
    if Ne_full % 2 != 0:
        raise ValueError(f"Expected even Ne from growth arrays, got Ne={Ne_full}")
    half = Ne_full // 2

    gx0 = Gx_sim_elem[0, half:]
    gy0 = Gy_sim_elem[0, half:]
    if gx0.shape[0] != H * W:
        raise ValueError(f"Expected bottom growth length H*W={H*W}, got {gx0.shape[0]}")

    lamdag_elem = np.stack([gx0, gy0], axis=1).astype(np.float64)

    G_grid = np.zeros((1, 2, H, W), dtype=np.float32)
    G_grid[0, 0] = lamdag_elem[:, 0].reshape(H, W)
    G_grid[0, 1] = lamdag_elem[:, 1].reshape(H, W)
    G_grid = normalize_growth_grid(G_grid, g_mean, g_std)

    Z0_raw  = Z_true_sim[0, :].astype(np.float32)
    V0_raw  = float(Z0_raw[volume_idx])
    SP0_raw = float(SP_sim[0])

    e_raw = SP0_raw - V0_raw
    I_raw = 0.0

    z0_norm = ((Z0_raw - z_mean) / z_std).astype(np.float32)[None, :]
    z_curr  = torch.from_numpy(z0_norm.astype(np.float32)).to(device)

    t_list        = [float(t_sim[0])]
    z_true_list   = [Z0_raw.copy()]
    z_pred_list   = [Z0_raw.copy()]
    vol_true_list = [V0_raw]
    vol_pred_list = [V0_raw]
    e_hist        = [e_raw]
    I_hist        = [I_raw]

    u_pred_nodes     = []
    g_pred_grid      = []
    lamdag_pred_elem = []

    u0 = decode_u(Z0_raw[:r_modes])
    if u0.shape[0] != node_xyz_bot.shape[0]:
        raise ValueError(f"Decode nodes mismatch: u0={u0.shape}, node_xyz_bot={node_xyz_bot.shape}")
    u_pred_nodes.append(u0.astype(np.float32))
    g_pred_grid.append(G_grid[0].copy())
    lamdag_pred_elem.append(lamdag_elem.copy())

    with torch.no_grad():
        for k in range(n_steps):
            t_k  = float(t_sim[k])
            t_k1 = float(t_sim[k+1])
            dt_k = float(t_k1 - t_k)
            if (not np.isfinite(dt_k)) or (dt_k <= 0.0):
                raise ValueError(f"Bad dt at step {k}: dt_k={dt_k} (t_k={t_k}, t_k1={t_k1})")

            SP_k_raw = float(SP_sim[k])
            e_norm   = (e_raw - e_mean) / e_std
            I_norm   = (I_raw - I_mean) / I_std
            sp_norm  = (SP_k_raw - sp_mean) / sp_std

            e_tensor  = torch.tensor([[e_norm]], dtype=torch.float32, device=device)
            I_tensor  = torch.tensor([[I_norm]], dtype=torch.float32, device=device)
            sp_tensor = torch.tensor([[sp_norm]], dtype=torch.float32, device=device)

            x_base   = torch.cat([z_curr, e_tensor, I_tensor, sp_tensor, design_tensor], dim=1)
            G_tensor = torch.from_numpy(G_grid.astype(np.float32)).to(device)

            dz_norm = model(x_base, G_tensor)

            z_next    = z_curr + dt_k * dz_norm
            z_next_np = z_next.detach().cpu().numpy()[0]

            Z_next_raw = (z_next_np * z_std + z_mean).astype(np.float32)
            V_next_raw = float(Z_next_raw[volume_idx])
            V_true_raw = float(Z_true_sim[k+1, volume_idx])

            z_pred_list.append(Z_next_raw.copy())
            z_true_list.append(Z_true_sim[k+1, :].astype(np.float32).copy())
            t_list.append(t_k1)
            vol_pred_list.append(V_next_raw)
            vol_true_list.append(V_true_raw)

            u_nodes = decode_u(Z_next_raw[:r_modes])
            if u_nodes.shape[0] != node_xyz_bot.shape[0]:
                raise ValueError(f"u_nodes mismatch at step {k}: {u_nodes.shape} vs {node_xyz_bot.shape}")
            if elem_conn_bot.max() >= u_nodes.shape[0]:
                raise ValueError(f"elem_conn_bot out of range at step {k}")

            # --- ONLY functional change vs rollout_single_sim_cnn_node: use
            # the UMAT-matched implicit-Newton integrator instead of the old
            # explicit-Euler integrate_growth_fast. ---
            lamdag_elem = integrate_growth_matched(
                node_coords=node_xyz_bot,
                elem_conn=elem_conn_bot,
                node_u=u_nodes,
                elem_lamdag=lamdag_elem,
                k1=k1, k2=k2, theta_crit=theta_crit,
                time=t_k,
                dt=dt_k,
            )

            G_grid = np.zeros((1, 2, H, W), dtype=np.float32)
            G_grid[0, 0] = lamdag_elem[:, 0].reshape(H, W)
            G_grid[0, 1] = lamdag_elem[:, 1].reshape(H, W)
            G_grid = normalize_growth_grid(G_grid, g_mean, g_std)

            u_pred_nodes.append(u_nodes.astype(np.float32))
            g_pred_grid.append(G_grid[0].copy())
            lamdag_pred_elem.append(lamdag_elem.copy())

            I_raw = I_raw + dt_k * e_raw
            SP_next_raw = float(SP_sim[k+1])
            e_raw = SP_next_raw - V_next_raw
            e_hist.append(e_raw)
            I_hist.append(I_raw)

            z_curr = z_next

    z_true_raw = np.vstack(z_true_list)
    z_pred_raw = np.vstack(z_pred_list)
    t_arr      = np.array(t_list, dtype=np.float64)

    vol_true = np.array(vol_true_list, dtype=np.float64)
    vol_pred = np.array(vol_pred_list, dtype=np.float64)

    err_L2  = np.linalg.norm(z_pred_raw - z_true_raw, axis=1)
    err_vol = np.abs(vol_pred - vol_true)

    area_gain_pred_final = compute_net_area_gain_from_lamdag_elem(
        lamdag_pred_elem[-1], A0_cm2=A0_CM2
    )

    rollout = {
        "t": t_arr,
        "z_true": z_true_raw,
        "z_pred": z_pred_raw,
        "vol_true": vol_true,
        "vol_pred": vol_pred,
        "err_L2": err_L2,
        "err_vol": err_vol,
        "e_hist": np.array(e_hist, dtype=np.float64),
        "I_hist": np.array(I_hist, dtype=np.float64),

        "u_pred_nodes": np.stack(u_pred_nodes, axis=0),
        "g_pred_grid": np.stack(g_pred_grid, axis=0),
        "lamdag_pred_elem": np.stack(lamdag_pred_elem, axis=0),

        "r_modes": r_modes,
        "sim_id": sim_id,
        "volume_idx": int(volume_idx),
        "volume_dim": int(volume_idx),
        "state_names": [*(f"u_lat_{i}" for i in range(r_modes)), "volume"],
        "H": int(H),
        "W": int(W),

        "area_gain_pred_final_cm2": float(area_gain_pred_final),
        "n_steps": int(n_steps),
    }

    return rollout


# --- CLI / standalone sanity check ---
if __name__ == "__main__":
    import sys

    r_modes = 9
    ckpt_template = "FINAL_models/Model_D_ablation_fullrollout/Model_D_CNN_r{r}_BEST.pt"
    test_sims = [210, 188]
    for sid in test_sims:
        roll = rollout_single_sim_cnn_node_matched(
            r_modes=r_modes,
            sim_id=sid,
            nodes_csv="GOH_Nodes_Test_for_Visualization.csv",
            elems_csv="GOH_Elements_Test_for_Visualization.csv",
            max_steps=5,
            ckpt_path_template=ckpt_template,
            make_plots=False,
        )
        u = roll["u_pred_nodes"]
        ok = np.all(np.isfinite(u))
        print(f"[sanity] sim={sid} u_pred_nodes.shape={u.shape} finite={ok}")
        if not ok:
            sys.exit(1)
    print("[sanity] OK")
