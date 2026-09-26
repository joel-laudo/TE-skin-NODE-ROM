"""
Self-contained library for evaluating Model D (the CNN growth-feedback NODE)
checkpoints on the full 185-sim validation set: data loading, mesh/POD-decoder
setup, the growth encoder + velocity-network model definition
(GrowthEncoderCNN/VelocityNet), the explicit-Euler growth integrator used by
the original (unmatched) rollout, the autoregressive rollout itself
(rollout_single_sim_cnn_node), validation-set zarr export
(save_val_rollouts_to_zarr), and a post-hoc surface-displacement-RMSE
augmentation pass (compute_surface_disp_rmse_trajectory /
add_surface_disp_rmse_to_rollout_zarr) used to compute the accuracy numbers
reported for Model D.

This module performs NO top-level execution / IO on import (safe to import).
"""

import os
import json
import functools
import time as _time

import numpy as np
import torch
import torch.nn as nn
from numba import njit, prange
import zarr


# ===============================
# Paths & templates
# ===============================
ZARR_DISP = "displacements.zarr"
ZARR_SDV = "ip_growth_elem.zarr"          # lambda_g_x, lambda_g_y latents
ZARR_VOL = "expd_volumes.zarr"
DESIGN_ZARR_PATH = "Design_and_Metadata.zarr"

U_LATENT_TEMPLATE = "latent_displ_r{r}/coeffs"

DISP_POD_GROUP = "pod_full"
DISP_POD_U_NAME = "U"
DISP_POD_MEAN_NAME = "mean"

NODES_CSV = "GOH_Nodes_Test_for_Visualization.csv"
ELEMS_CSV = "GOH_Elements_Test_for_Visualization.csv"

A0_CM2 = 0.25

SEED = 123


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===============================
# 185-sim validation set (verified identical to
# FINAL_models/Model_D/Model_D_cnn_val_rollouts_r9.zarr's simulations group, n=185)
# ===============================
val_sims = np.array([210, 188, 249,   7, 584, 921, 294, 924,  85, 316, 291, 760, 889, 279, 303, 396, 580, 200,
 267, 794, 682, 310, 732, 643, 369, 832, 280, 147, 172, 196, 894, 227, 566, 733, 343, 585,
 344, 145, 352, 577, 458, 728, 916, 749, 161, 692, 660, 726, 880, 138,   5, 269, 373, 346,
 876,  41, 657, 184, 134, 683, 353, 735, 536, 306, 606, 712, 761, 724, 447, 372,  48, 748,
 859, 548, 388, 273, 900,  35, 177, 298, 785, 171, 590,   4, 314, 879, 508, 844, 699, 430,
  50, 170, 389, 919, 663, 209, 744, 164, 561, 775,  57, 131, 551, 167, 543, 203, 520, 578,
  55, 746, 521, 461, 453, 840, 847, 103, 804, 597, 814, 328, 793,  52, 852, 849, 338,  54,
 117, 831, 769, 374, 204,  31, 421, 553,  36, 702, 798, 348, 252, 313, 100, 911, 621, 274,
 270, 624, 600, 791,  78, 743, 345,  79, 162, 336, 235, 568, 616, 329, 351,  91, 807,  90,
  75, 656, 898, 531, 825, 392, 260, 829, 221, 813, 776, 511, 655, 802, 545, 677, 861, 261,
  13,  74, 185, 229, 437], dtype=np.int64)

assert val_sims.shape[0] == 185, f"expected 185 val sims, got {val_sims.shape[0]}"


# ===============================
# Bottom-surface grid mapping cache
# ===============================
_BOTTOM_GRID_CACHE = None


def get_bottom_grid_cache():
    global _BOTTOM_GRID_CACHE
    if _BOTTOM_GRID_CACHE is None:
        H, W, bottom_idx, *_ = build_bottom_surface_elem_grid_mapping(
            nodes_csv=NODES_CSV,
            elems_csv=ELEMS_CSV,
            tol_xy=1e-8,
        )
        _BOTTOM_GRID_CACHE = (H, W, bottom_idx)
    return _BOTTOM_GRID_CACHE


def build_bottom_surface_elem_grid_mapping(
    nodes_csv: str = NODES_CSV,
    elems_csv: str = ELEMS_CSV,
    tol_xy: float = 1e-8,
):
    nodes = np.loadtxt(nodes_csv, delimiter=",", dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1] < 4:
        raise ValueError(f"Nodes CSV must have 4 cols [id,x,y,z]. Got shape {nodes.shape}")

    node_id = nodes[:, 0].astype(np.int64)
    node_xyz = nodes[:, 1:4].astype(np.float64)

    node_id_to_local = {int(nid): i for i, nid in enumerate(node_id)}

    elems = np.loadtxt(elems_csv, delimiter=",", dtype=np.int64)
    if elems.ndim != 2 or elems.shape[1] < 5:
        raise ValueError(f"Elements CSV must have >=5 cols [eid,n1,n2,n3,n4]. Got shape {elems.shape}")

    elem_id = elems[:, 0].astype(np.int64)
    elem_conn_ids = elems[:, 1:5].astype(np.int64)
    Ne = elem_conn_ids.shape[0]

    elem_conn_local = np.empty((Ne, 4), dtype=np.int64)
    for a in range(4):
        try:
            elem_conn_local[:, a] = np.array(
                [node_id_to_local[int(nid)] for nid in elem_conn_ids[:, a]],
                dtype=np.int64
            )
        except KeyError as e:
            raise KeyError(f"Element connectivity references node id not found: {e}")

    cent = node_xyz[elem_conn_local].mean(axis=1)
    xe, ye, ze = cent[:, 0], cent[:, 1], cent[:, 2]

    xe_q = np.round(xe / tol_xy) * tol_xy
    ye_q = np.round(ye / tol_xy) * tol_xy

    xs = np.unique(xe_q); xs.sort()
    ys = np.unique(ye_q); ys.sort()
    W = xs.size
    H = ys.size
    n_cells = H * W

    ix = np.searchsorted(xs, xe_q)
    iy = np.searchsorted(ys, ye_q)
    cell = (iy * W + ix).astype(np.int64)

    bottom_elem_idx = np.full(n_cells, -1, dtype=np.int64)
    best_z = np.full(n_cells, np.inf, dtype=np.float64)

    for e in range(Ne):
        c = cell[e]
        if ze[e] < best_z[c]:
            best_z[c] = ze[e]
            bottom_elem_idx[c] = e

    occupied = bottom_elem_idx >= 0
    if not np.all(occupied):
        missing = np.where(~occupied)[0][:10]
        raise ValueError(f"Some grid cells have no assigned element. Missing: {missing}")

    cell_xy = np.stack([xe[bottom_elem_idx], ye[bottom_elem_idx]], axis=1)

    return H, W, bottom_elem_idx, cell_xy, elem_id, node_id, node_xyz, elem_conn_local, xs, ys


@functools.lru_cache(maxsize=None)
def get_growth_params_for_sim(sim_id: int):
    g_disp = zarr.open(DESIGN_ZARR_PATH, mode="r")
    job_index_per_sim = g_disp["Mapping_indexes_and_metadata"]["job_index_per_sim"][:]
    job_id = int(job_index_per_sim[sim_id])

    design_file = "PGOH_Ideal_tol_Tcrit_V_mu_kk1_kk2_kappa_k1_design.txt"
    design_table = np.loadtxt(design_file, skiprows=0)

    row_index = job_id - 1
    if row_index < 0 or row_index >= design_table.shape[0]:
        raise IndexError(f"Job ID {job_id} (row idx {row_index}) out of range ({design_table.shape[0]} rows).")

    row = design_table[row_index]
    theta_crit = float(row[2])
    k1 = float(row[5])
    k2 = float(row[6])
    return job_id, theta_crit, k1, k2


@functools.lru_cache(maxsize=None)
def load_raw_data(r_modes: int):
    from pathlib import Path

    g_disp = zarr.open_group(ZARR_DISP, mode="r")
    g_design = zarr.open_group(DESIGN_ZARR_PATH, mode="r")

    vol_root = Path(ZARR_VOL)

    u_path = U_LATENT_TEMPLATE.format(r=r_modes)
    U_lat = np.asarray(g_disp[u_path], dtype=np.float32).T  # (S, r)

    mgrp = g_design["Mapping_indexes_and_metadata"]
    sim_index = np.asarray(mgrp["sim_index"], dtype=np.int64)
    time_vals = np.asarray(mgrp["time_vals"], dtype=np.float64)

    volume_SP = np.asarray(
        zarr.open_array(str(vol_root / "volume_setpoint_per_snapshot"), mode="r"),
        dtype=np.float32
    )

    design_all = np.asarray(g_design["design_params_per_snapshot"], dtype=np.float32)

    S = U_lat.shape[0]
    assert sim_index.shape[0] == S == time_vals.shape[0]
    assert volume_SP.shape[0] == S == design_all.shape[0]

    volume_snap = np.asarray(
        zarr.open_array(str(vol_root / "expander_volume" / "cvol"), mode="r"),
        dtype=np.float32
    )
    if volume_snap.ndim != 1 or volume_snap.shape[0] != S:
        raise ValueError(f"Expected cvol as (S,) with S={S}, got shape {volume_snap.shape}")

    return (
        U_lat,
        volume_snap,
        sim_index,
        time_vals,
        volume_SP,
        design_all,
        len(np.unique(sim_index)),
    )


@functools.lru_cache(maxsize=None)
def load_bottom_surface_mesh_direct(nodes_csv: str, elems_csv: str):
    nodes = np.loadtxt(nodes_csv, delimiter=",")
    elems = np.loadtxt(elems_csv, delimiter=",")

    node_ids = nodes[:, 0].astype(np.int64)
    node_xyz = nodes[:, 1:4].astype(np.float64)

    elem_node_ids = elems[:, 1:5].astype(np.int64)
    Ne = elem_node_ids.shape[0]
    if Ne % 2 != 0:
        raise ValueError(f"Expected even number of elements (2 layers), got Ne={Ne}")

    half = Ne // 2
    elem_node_ids_bot = elem_node_ids[half:, :]

    bottom_node_ids = np.unique(elem_node_ids_bot.reshape(-1))

    id2xyz = {int(nid): node_xyz[i] for i, nid in enumerate(node_ids)}

    missing = [int(nid) for nid in bottom_node_ids if int(nid) not in id2xyz]
    if len(missing) > 0:
        raise ValueError(f"{len(missing)} node IDs missing from nodes_csv. Example: {missing[:10]}")

    node_ids_bot = np.sort(bottom_node_ids.astype(np.int64))
    node_xyz_bot = np.vstack([id2xyz[int(nid)] for nid in node_ids_bot]).astype(np.float64)

    id2bot = {int(nid): i for i, nid in enumerate(node_ids_bot)}
    get_idx = np.vectorize(lambda nid: id2bot.get(int(nid), -1), otypes=[np.int64])
    elem_conn_bot = get_idx(elem_node_ids_bot)

    bad = np.where(elem_conn_bot < 0)
    if bad[0].size > 0:
        ei = int(bad[0][0])
        raise ValueError(f"Internal mapping failure. elem row {ei}: {elem_node_ids_bot[ei]} -> {elem_conn_bot[ei]}")

    return node_xyz_bot, elem_conn_bot, node_ids_bot, elem_node_ids_bot


def normalize_growth_grid(G_grid: np.ndarray, g_mean, g_std, eps: float = 1e-8) -> np.ndarray:
    G = G_grid.astype(np.float32, copy=False)
    g_mean = np.asarray(g_mean, dtype=np.float32)
    g_std = np.asarray(g_std, dtype=np.float32)

    if G.ndim == 3:
        if g_mean.shape == (2,):
            mu = g_mean[:, None, None]
            sig = g_std[:, None, None]
        else:
            mu = float(g_mean)
            sig = float(g_std)
        return (G - mu) / (sig + eps)

    if G.ndim == 4:
        if g_mean.shape == (2,):
            mu = g_mean[None, :, None, None]
            sig = g_std[None, :, None, None]
        else:
            mu = float(g_mean)
            sig = float(g_std)
        return (G - mu) / (sig + eps)

    raise ValueError(f"Expected G_grid with shape (2,H,W) or (B,2,H,W), got {G.shape}")


_GROWTH_SNAP_CACHE = {}


def load_elem_growth_for_sim(
    sim_id: int,
    sim_index: np.ndarray,
    time_vals: np.ndarray,
    zarr_sdv_path: str = ZARR_SDV,
    folder_x: str = "snaps_SDV1",
    folder_y: str = "snaps_SDV4",
):
    g_sdv = zarr.open_group(zarr_sdv_path, mode="r")

    def _load_snap_folder(folder_name: str) -> np.ndarray:
        cache_key = (zarr_sdv_path, folder_name)
        if cache_key in _GROWTH_SNAP_CACHE:
            return _GROWTH_SNAP_CACHE[cache_key]
        candidates = [
            f"{folder_name}/{folder_name}",
            f"{folder_name}/data",
            f"{folder_name}/snaps",
            f"{folder_name}",
        ]
        last_err = None
        for key in candidates:
            try:
                arr = np.asarray(g_sdv[key], dtype=np.float32)
                _GROWTH_SNAP_CACHE[cache_key] = arr
                return arr
            except Exception as e:
                last_err = e
        raise KeyError(f"Could not load '{folder_name}'. Tried={candidates}. Last error={repr(last_err)}")

    Gx_raw = _load_snap_folder(folder_x)
    Gy_raw = _load_snap_folder(folder_y)

    idx = np.where(sim_index == sim_id)[0]
    if idx.size < 2:
        raise ValueError(f"Sim {sim_id} has too few frames ({idx.size}).")

    t_s = time_vals[idx]
    order = np.argsort(t_s)
    idx_sorted = idx[order]
    t_sim = t_s[order]

    S = sim_index.shape[0]
    if Gx_raw.shape[1] == S:
        Gx_all = Gx_raw.T
    else:
        Gx_all = Gx_raw

    if Gy_raw.shape[1] == S:
        Gy_all = Gy_raw.T
    else:
        Gy_all = Gy_raw

    Gx_sim = Gx_all[idx_sorted, :]
    Gy_sim = Gy_all[idx_sorted, :]

    return t_sim, Gx_sim, Gy_sim, idx_sorted


@functools.lru_cache(maxsize=None)
def load_disp_decoder(
    r_modes: int,
    zarr_disp_path: str = ZARR_DISP,
    pod_group: str = DISP_POD_GROUP,
    pod_u_name: str = DISP_POD_U_NAME,
    pod_mean_name: str = DISP_POD_MEAN_NAME,
    verbose: bool = True,
):
    g_disp = zarr.open_group(zarr_disp_path, mode="r")

    U_key = f"{pod_group}/{pod_u_name}"
    mu_key = f"{pod_group}/{pod_mean_name}"

    U_full = np.asarray(g_disp[U_key])
    mu = np.asarray(g_disp[mu_key]).reshape(-1)

    if U_full.ndim != 2:
        raise ValueError(f"Expected POD basis U to be 2D. Got {U_full.shape} at '{U_key}'")

    M, Rmax = U_full.shape
    if mu.shape[0] != M:
        raise ValueError(f"Mean length must match basis rows M={M}. Got mu.shape={mu.shape}")

    if r_modes > Rmax:
        raise ValueError(f"Requested r_modes={r_modes} exceeds available Rmax={Rmax}")

    U_r = U_full[:, :r_modes]

    if M % 3 != 0:
        raise ValueError(f"Expected M divisible by 3. Got M={M}")
    N = M // 3

    if verbose:
        print(f"[load_disp_decoder] Using U='{U_key}' shape={U_full.shape}")
        print(f"[load_disp_decoder] Using mean='{mu_key}' shape={mu.shape}")
        print(f"[load_disp_decoder] r_modes={r_modes}, N_nodes={N}")

    def decode(alpha: np.ndarray) -> np.ndarray:
        alpha = np.asarray(alpha).reshape(-1)
        if alpha.shape[0] != r_modes:
            raise ValueError(f"alpha must have shape ({r_modes},), got {alpha.shape}")

        u_flat = mu + U_r @ alpha
        ux = u_flat[0:N]
        uy = u_flat[N:2*N]
        uz = u_flat[2*N:3*N]
        return np.stack([ux, uy, uz], axis=1)

    return decode


# ===============================
# Growth integration (bilinear quad, numba-accelerated)
# ===============================
def shape_function_gradients(xi, eta):
    dN_dxi = np.array([
        -(1 - eta) / 4,
         (1 - eta) / 4,
         (1 + eta) / 4,
        -(1 + eta) / 4
    ])
    dN_deta = np.array([
        -(1 - xi) / 4,
        -(1 + xi) / 4,
         (1 + xi) / 4,
         (1 - xi) / 4
    ])
    return dN_dxi, dN_deta


@njit
def _growth_gate(time):
    if (0.0 < time < 7.0) or (14.0 < time < 21.0) or (28.0 < time < 35.0) or \
       (42.0 < time < 49.0) or (56.0 < time < 63.0) or (70.0 < time < 77.0) or \
       (84.0 < time < 91.0) or (98.0 < time < 105.0):
        return 0.0
    return 1.0


@njit
def _inv3x3(A):
    det = (A[0,0]*(A[1,1]*A[2,2]-A[1,2]*A[2,1])
         - A[0,1]*(A[1,0]*A[2,2]-A[1,2]*A[2,0])
         + A[0,2]*(A[1,0]*A[2,1]-A[1,1]*A[2,0]))
    invA = np.empty((3,3), dtype=A.dtype)
    invA[0,0] =  (A[1,1]*A[2,2]-A[1,2]*A[2,1]) / det
    invA[0,1] = -(A[0,1]*A[2,2]-A[0,2]*A[2,1]) / det
    invA[0,2] =  (A[0,1]*A[1,2]-A[0,2]*A[1,1]) / det
    invA[1,0] = -(A[1,0]*A[2,2]-A[1,2]*A[2,0]) / det
    invA[1,1] =  (A[0,0]*A[2,2]-A[0,2]*A[2,0]) / det
    invA[1,2] = -(A[0,0]*A[1,2]-A[0,2]*A[1,0]) / det
    invA[2,0] =  (A[1,0]*A[2,1]-A[1,1]*A[2,0]) / det
    invA[2,1] = -(A[0,0]*A[2,1]-A[0,1]*A[2,0]) / det
    invA[2,2] =  (A[0,0]*A[1,1]-A[0,1]*A[1,0]) / det
    return invA


@njit(parallel=True, fastmath=True)
def integrate_growth_numba(node_coords, elem_conn, node_u, elem_lamdag,
                           k1, k2, theta_crit, time, dt,
                           dN_dxi, dN_deta):
    Ne = elem_conn.shape[0]
    out = elem_lamdag.copy()

    gate = _growth_gate(time)

    for ei in prange(Ne):
        nids = elem_conn[ei]

        X = node_coords[nids, :]
        u = node_u[nids, :]
        x = X + u

        Gxi  = np.zeros(3, dtype=np.float64)
        Geta = np.zeros(3, dtype=np.float64)
        gxi  = np.zeros(3, dtype=np.float64)
        geta = np.zeros(3, dtype=np.float64)

        for a in range(4):
            Gxi[0]  += X[a,0] * dN_dxi[a]
            Gxi[1]  += X[a,1] * dN_dxi[a]
            Gxi[2]  += X[a,2] * dN_dxi[a]

            Geta[0] += X[a,0] * dN_deta[a]
            Geta[1] += X[a,1] * dN_deta[a]
            Geta[2] += X[a,2] * dN_deta[a]

            gxi[0]  += x[a,0] * dN_dxi[a]
            gxi[1]  += x[a,1] * dN_dxi[a]
            gxi[2]  += x[a,2] * dN_dxi[a]

            geta[0] += x[a,0] * dN_deta[a]
            geta[1] += x[a,1] * dN_deta[a]
            geta[2] += x[a,2] * dN_deta[a]

        Nx = Gxi[1]*Geta[2] - Gxi[2]*Geta[1]
        Ny = Gxi[2]*Geta[0] - Gxi[0]*Geta[2]
        Nz = Gxi[0]*Geta[1] - Gxi[1]*Geta[0]
        nrm = np.sqrt(Nx*Nx + Ny*Ny + Nz*Nz) + 1e-12
        Nx /= nrm; Ny /= nrm; Nz /= nrm

        nx = gxi[1]*geta[2] - gxi[2]*geta[1]
        ny = gxi[2]*geta[0] - gxi[0]*geta[2]
        nz = gxi[0]*geta[1] - gxi[1]*geta[0]
        nrm2 = np.sqrt(nx*nx + ny*ny + nz*nz) + 1e-12
        nx /= nrm2; ny /= nrm2; nz /= nrm2

        G = np.empty((3,3), dtype=np.float64)
        G[0,0]=Gxi[0];  G[1,0]=Gxi[1];  G[2,0]=Gxi[2]
        G[0,1]=Geta[0]; G[1,1]=Geta[1]; G[2,1]=Geta[2]
        G[0,2]=Nx;      G[1,2]=Ny;      G[2,2]=Nz

        Ginv = _inv3x3(G)
        G_dual_xi  = Ginv[0, :]
        G_dual_eta = Ginv[1, :]

        F = np.empty((3,3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                F[i,j] = gxi[i]*G_dual_xi[j] + geta[i]*G_dual_eta[j]
        F[0,0] += nx*Nx; F[0,1] += nx*Ny; F[0,2] += nx*Nz
        F[1,0] += ny*Nx; F[1,1] += ny*Ny; F[1,2] += ny*Nz
        F[2,0] += nz*Nx; F[2,1] += nz*Ny; F[2,2] += nz*Nz

        lam1g_t = out[ei,0]
        lam2g_t = out[ei,1]

        inv1 = 1.0/(lam1g_t + 1e-12)
        inv2 = 1.0/(lam2g_t + 1e-12)

        Fg_inv = np.empty((3,3), dtype=np.float64)
        Fg_inv[0,0] = inv1 + Nx*Nx;  Fg_inv[0,1] = Nx*Ny;       Fg_inv[0,2] = Nx*Nz
        Fg_inv[1,0] = Ny*Nx;         Fg_inv[1,1] = inv2 + Ny*Ny; Fg_inv[1,2] = Ny*Nz
        Fg_inv[2,0] = Nz*Nx;         Fg_inv[2,1] = Nz*Ny;        Fg_inv[2,2] = 1.0 + Nz*Nz

        Fe = F @ Fg_inv
        Ce = Fe.T @ Fe

        lam1e = np.sqrt(Ce[0,0])
        lam2e = np.sqrt(Ce[1,1])

        lam1gdot = 0.0
        lam2gdot = 0.0
        if lam1e >= theta_crit:
            lam1gdot = k1 * (lam1e - theta_crit)
        if lam2e >= theta_crit:
            lam2gdot = k2 * (lam2e - theta_crit)

        lam1gdot *= gate
        lam2gdot *= gate

        out[ei,0] = lam1g_t + dt * lam1gdot
        out[ei,1] = lam2g_t + dt * lam2gdot

    return out


def integrate_growth_fast(node_coords, elem_conn, node_u, elem_lamdag, k1, k2, theta_crit, time, dt=1.4):
    dN_dxi, dN_deta = shape_function_gradients(0.0, 0.0)
    dN_dxi  = np.asarray(dN_dxi,  dtype=np.float64).reshape(4,)
    dN_deta = np.asarray(dN_deta, dtype=np.float64).reshape(4,)
    return integrate_growth_numba(
        np.asarray(node_coords, dtype=np.float64),
        np.asarray(elem_conn, dtype=np.int64),
        np.asarray(node_u, dtype=np.float64),
        np.asarray(elem_lamdag, dtype=np.float64),
        float(k1), float(k2), float(theta_crit), float(time), float(dt),
        dN_dxi, dN_deta
    )


def compute_net_area_gain_from_lamdag_elem(lamdag_elem: np.ndarray, A0_cm2: float = A0_CM2) -> float:
    gx = lamdag_elem[:, 0]
    gy = lamdag_elem[:, 1]
    A_final = gx * gy * A0_cm2
    net_gain = A_final - A0_cm2
    return float(np.sum(net_gain))


def compute_true_net_area_gain_final_for_sim(
    sim_id: int,
    sim_index: np.ndarray,
    time_vals: np.ndarray,
    H: int,
    W: int,
    zarr_sdv_path: str,
    folder_x: str = "snaps_SDV1",
    folder_y: str = "snaps_SDV4",
    A0_cm2: float = A0_CM2,
) -> float:
    t_sim_g, Gx_sim_elem, Gy_sim_elem, _ = load_elem_growth_for_sim(
        sim_id=sim_id,
        sim_index=sim_index,
        time_vals=time_vals,
        zarr_sdv_path=zarr_sdv_path,
        folder_x=folder_x,
        folder_y=folder_y,
    )
    final_k = Gx_sim_elem.shape[0] - 1

    Ne_full = Gx_sim_elem.shape[1]
    if Ne_full % 2 != 0:
        raise ValueError(f"Expected even Ne from growth arrays, got Ne={Ne_full}")
    half = Ne_full // 2

    gx = Gx_sim_elem[final_k, half:]
    gy = Gy_sim_elem[final_k, half:]

    if gx.shape[0] != H * W:
        raise ValueError(f"Expected bottom elems H*W={H*W}, got {gx.shape[0]}")

    lamdag_elem_true = np.stack([gx, gy], axis=1).astype(np.float64)
    return compute_net_area_gain_from_lamdag_elem(lamdag_elem_true, A0_cm2=A0_cm2)


# ===============================
# Model definition (Model D: CNN growth encoder + VelocityNet)
# ===============================
class GrowthEncoderCNN(nn.Module):
    """Small convolutional encoder that maps the current growth-stretch
    field (rasterized onto a (H,W) grid) down to a compact feature vector
    h_g/g_CNN. Three stride-2 conv blocks progressively downsample the
    spatial grid while increasing channel width, followed by global average
    pooling and a linear head -- so the whole spatial growth pattern is
    summarized into a single vector that VelocityNet can condition on."""

    def __init__(self, dim_gfeat: int = 32, in_channels: int = 1, base_channels: int = 16):
        super().__init__()
        C = base_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, C, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=C),
            nn.SiLU(),

            nn.Conv2d(C, 2 * C, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=2 * C),
            nn.SiLU(),

            nn.Conv2d(2 * C, 4 * C, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=4 * C),
            nn.SiLU(),

            nn.Conv2d(4 * C, 4 * C, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=4 * C),
            nn.SiLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * C, dim_gfeat),
            nn.SiLU(),
        )

    def forward(self, G: torch.Tensor) -> torch.Tensor:
        if G.dim() == 3:
            G = G.unsqueeze(1)
        if G.dim() != 4:
            raise ValueError(f"Expected G with shape [B,C,H,W], got {tuple(G.shape)}")
        return self.head(self.cnn(G))


class VelocityNet(nn.Module):
    """Model D's dynamics network: predicts the latent-state velocity dz/dt
    from the concatenation of the Model-A-style state (mechanical latent
    state, pressure setpoint, design parameters, PI tracking error/integral)
    and the CNN-encoded growth feature h_g produced by growth_encoder. This
    is the growth-feedback pathway -- h_g is recomputed from the current
    growth field at every rollout step and feeds directly into the
    velocity prediction, unlike Model A's open-loop dynamics."""

    def __init__(self,
                 dim_state: int,
                 dim_sp: int = 1,
                 dim_design: int = 7,
                 dim_error: int = 1,
                 dim_integral: int = 1,
                 dim_gfeat: int = 32,
                 hidden: int = 128,
                 growth_in_channels: int = 2):
        super().__init__()

        self.dim_state = dim_state
        self.dim_gfeat = dim_gfeat

        self.growth_encoder = GrowthEncoderCNN(
            dim_gfeat=dim_gfeat,
            in_channels=growth_in_channels
        )

        input_dim = dim_state + dim_sp + dim_design + dim_error + dim_integral + dim_gfeat

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim_state),
        )

    def forward(self, x_base: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        h = self.growth_encoder(G)
        x = torch.cat([x_base, h], dim=-1)
        return self.net(x)


def build_cnn_node_from_ckpt(ckpt: dict, device: torch.device):
    """
    Robust rebuild for CNN-NODE VelocityNet by inferring architecture from state_dict.
    Defensive: only accesses needed keys by name; does not assert an exact key set,
    so SWA checkpoints with extra keys (swa_n_snapshots, swa_snapshot_epochs, ...)
    load fine.
    """
    sd = ckpt["model_state_dict"]

    if "net.4.bias" not in sd:
        raise KeyError("Could not find 'net.4.bias' in checkpoint. Architecture naming mismatch?")
    D_state = int(sd["net.4.bias"].numel())

    if "net.0.bias" not in sd:
        raise KeyError("Could not find 'net.0.bias' in checkpoint. Architecture naming mismatch?")
    hidden = int(sd["net.0.bias"].numel())

    if "growth_encoder.head.1.bias" not in sd:
        raise KeyError("Could not find 'growth_encoder.head.1.bias' in checkpoint. Architecture naming mismatch?")
    dim_gfeat = int(sd["growth_encoder.head.1.bias"].numel())

    if "growth_encoder.cnn.0.weight" not in sd:
        raise KeyError("Could not find 'growth_encoder.cnn.0.weight' in checkpoint. Architecture naming mismatch?")
    w0 = sd["growth_encoder.cnn.0.weight"]
    base_channels = int(w0.shape[0])
    growth_in_channels = int(w0.shape[1])

    dim_sp = int(ckpt.get("dim_sp", 1))
    dim_design = int(ckpt.get("dim_design", 7))
    dim_error = int(ckpt.get("dim_error", 1))
    dim_integral = int(ckpt.get("dim_integral", 1))

    print("[infer CNN-NODE arch] D_state=%d hidden=%d dim_gfeat=%d growth_in_channels=%d base_channels=%d"
          % (D_state, hidden, dim_gfeat, growth_in_channels, base_channels))

    model = VelocityNet(
        dim_state=D_state,
        dim_sp=dim_sp,
        dim_design=dim_design,
        dim_error=dim_error,
        dim_integral=dim_integral,
        dim_gfeat=dim_gfeat,
        hidden=hidden,
        growth_in_channels=growth_in_channels,
    ).to(device)

    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


# ===============================
# Core rollout (growth-feedback closed loop)
# ===============================
def rollout_single_sim_cnn_node(
    r_modes: int,
    sim_id: int,
    nodes_csv: str,
    elems_csv: str,
    max_steps=None,
    ckpt_path_template: str = "FINAL_models/Model_D/Model_D_CNN_r{r}_BEST.pt",
):
    """
    Rollout with growth feedback (Euler):
      z_{k+1} = z_k + dt_k * f(x_base_k, G_k_norm)

    Growth feedback loop:
      z_k -> decode u_k -> integrate growth (elem) -> rasterize (2,H,W) -> normalize -> CNN -> NODE

    ckpt_path_template is .format(r=r_modes)'d; a literal full path (no '{r}'
    placeholder) works fine too, as long as it contains no stray curly braces.

    Checkpoint dict is accessed defensively by key name only (works for both
    normal BEST/END checkpoints and SWA checkpoints with extra metadata keys).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ckpt_path_template.format(r=r_modes)
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

            lamdag_elem = integrate_growth_fast(
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


# ===============================
# Zarr export (schema-matched to Model_D_ablation_fullrollout reference)
# ===============================
def _zarr_write_array(grp, name, arr, overwrite=True, _max_retries: int = 5):
    """
    Robust helper for writing arrays across common zarr versions.

    Wrapped with a small retry loop: zarr-python 3.x's async LocalStore has a
    benign race on Windows where the parent directory for a freshly-created
    array (e.g. 'sim_00210/alpha_pred/') can raise FileNotFoundError when its
    '.partial' temp file is opened for write before directory creation has
    fully landed. This is transient and disappears on retry; it is unrelated
    to the actual rollout/metric data being written.
    """
    arr = np.asarray(arr)
    if name in grp and overwrite:
        del grp[name]

    last_err = None
    for attempt in range(_max_retries):
        try:
            try:
                grp.create_dataset(name, data=arr, shape=arr.shape, dtype=arr.dtype, overwrite=overwrite)
            except TypeError:
                try:
                    grp.create_dataset(name, data=arr, overwrite=overwrite)
                except Exception:
                    grp.array(name, arr, overwrite=overwrite)
            return
        except FileNotFoundError as e:
            last_err = e
            _time.sleep(0.25 * (attempt + 1))
            continue
    raise last_err


def _compute_Ag_hist_from_lamdag_stack(lamdag_stack: np.ndarray, A0_cm2: float = A0_CM2) -> np.ndarray:
    lamdag_stack = np.asarray(lamdag_stack)
    if lamdag_stack.ndim != 3 or lamdag_stack.shape[-1] != 2:
        raise ValueError(f"Expected lamdag_stack shape (T,Ne,2), got {lamdag_stack.shape}")
    gx = lamdag_stack[:, :, 0]
    gy = lamdag_stack[:, :, 1]
    return ((gx * gy - 1.0) * A0_cm2).sum(axis=1)


def save_val_rollouts_to_zarr(
    *,
    zarr_path: str,
    val_sims,
    rollout_fn,
    rollout_kwargs: dict,
    r_modes: int,
    model_name: str,
    A0_cm2: float = A0_CM2,
    overwrite: bool = True,
    print_every: int = 1,
):
    val_sims = [int(s) for s in val_sims]

    if os.path.exists(zarr_path) and overwrite:
        import shutil
        shutil.rmtree(zarr_path)

    root = zarr.open_group(zarr_path, mode="w")

    root.attrs["model_name"] = str(model_name)
    root.attrs["r_modes"] = int(r_modes)
    root.attrs["A0_cm2"] = float(A0_cm2)
    root.attrs["n_val_sims"] = int(len(val_sims))
    root.attrs["rollout_fn_name"] = getattr(rollout_fn, "__name__", "unknown")
    root.attrs["rollout_kwargs_json"] = json.dumps(
        {k: str(v) for k, v in rollout_kwargs.items()}
    )

    sims_written = []
    lengths = []
    H_list = []
    W_list = []
    N_nodes_list = []

    sims_grp = root.require_group("simulations")

    n_sims = len(val_sims)
    t0 = _time.time()

    for i, sim_id in enumerate(val_sims, start=1):
        if print_every > 0 and (i == 1 or i % print_every == 0 or i == n_sims):
            elapsed = _time.time() - t0
            done = max(i - 1, 1)
            rate = elapsed / done
            eta = rate * (n_sims - (i - 1))
            print(f"[{model_name}] sim {i}/{n_sims} (sim_id={sim_id}) | "
                  f"elapsed={elapsed:.1f}s | avg={rate:.3f}s/sim | ETA={eta:.1f}s")

        roll = rollout_fn(
            r_modes=r_modes,
            sim_id=int(sim_id),
            **rollout_kwargs,
        )

        sim_grp = sims_grp.require_group(f"sim_{int(sim_id):05d}")

        t = np.asarray(roll["t"], dtype=np.float64)
        z_pred = np.asarray(roll["z_pred"], dtype=np.float32)
        z_true = np.asarray(roll["z_true"], dtype=np.float32) if "z_true" in roll else None

        alpha_pred = np.asarray(z_pred[:, :r_modes], dtype=np.float32)

        u_pred_nodes = np.asarray(roll["u_pred_nodes"], dtype=np.float32)
        g_pred_grid = np.asarray(roll["g_pred_grid"], dtype=np.float32)
        lamdag_pred_elem = np.asarray(roll["lamdag_pred_elem"], dtype=np.float32)

        vol_pred = np.asarray(roll["vol_pred"], dtype=np.float32) if "vol_pred" in roll else None
        vol_true = np.asarray(roll["vol_true"], dtype=np.float32) if "vol_true" in roll else None
        err_L2 = np.asarray(roll["err_L2"], dtype=np.float32) if "err_L2" in roll else None
        err_vol = np.asarray(roll["err_vol"], dtype=np.float32) if "err_vol" in roll else None
        e_hist = np.asarray(roll["e_hist"], dtype=np.float32) if "e_hist" in roll else None
        I_hist = np.asarray(roll["I_hist"], dtype=np.float32) if "I_hist" in roll else None

        if "Ag_hist_raw_cm2" in roll:
            Ag_pred_cm2 = np.asarray(roll["Ag_hist_raw_cm2"], dtype=np.float32)
        else:
            Ag_pred_cm2 = _compute_Ag_hist_from_lamdag_stack(
                lamdag_pred_elem, A0_cm2=A0_cm2
            ).astype(np.float32)

        _zarr_write_array(sim_grp, "t", t)
        _zarr_write_array(sim_grp, "alpha_pred", alpha_pred)
        _zarr_write_array(sim_grp, "z_pred", z_pred)

        if z_true is not None:
            _zarr_write_array(sim_grp, "z_true", z_true)

        _zarr_write_array(sim_grp, "u_pred_nodes", u_pred_nodes)
        _zarr_write_array(sim_grp, "g_pred_grid", g_pred_grid)
        _zarr_write_array(sim_grp, "lamdag_pred_elem", lamdag_pred_elem)
        _zarr_write_array(sim_grp, "Ag_pred_cm2", Ag_pred_cm2)

        if vol_pred is not None:
            _zarr_write_array(sim_grp, "vol_pred", vol_pred)
        if vol_true is not None:
            _zarr_write_array(sim_grp, "vol_true", vol_true)
        if err_L2 is not None:
            _zarr_write_array(sim_grp, "err_L2", err_L2)
        if err_vol is not None:
            _zarr_write_array(sim_grp, "err_vol", err_vol)
        if e_hist is not None:
            _zarr_write_array(sim_grp, "e_hist", e_hist)
        if I_hist is not None:
            _zarr_write_array(sim_grp, "I_hist", I_hist)

        sim_grp.attrs["sim_id"] = int(sim_id)
        sim_grp.attrs["r_modes"] = int(r_modes)
        sim_grp.attrs["model_name"] = str(model_name)
        sim_grp.attrs["T"] = int(t.shape[0])
        sim_grp.attrs["H"] = int(roll["H"])
        sim_grp.attrs["W"] = int(roll["W"])
        sim_grp.attrs["N_nodes"] = int(u_pred_nodes.shape[1])
        sim_grp.attrs["D_state"] = int(z_pred.shape[1])
        sim_grp.attrs["final_err_L2"] = float(err_L2[-1]) if err_L2 is not None else np.nan
        sim_grp.attrs["final_err_vol"] = float(err_vol[-1]) if err_vol is not None else np.nan
        sim_grp.attrs["final_Ag_pred_cm2"] = float(Ag_pred_cm2[-1])
        sim_grp.attrs["area_gain_pred_final_cm2"] = float(roll.get("area_gain_pred_final_cm2", Ag_pred_cm2[-1]))

        sims_written.append(int(sim_id))
        lengths.append(int(t.shape[0]))
        H_list.append(int(roll["H"]))
        W_list.append(int(roll["W"]))
        N_nodes_list.append(int(u_pred_nodes.shape[1]))

    _zarr_write_array(root, "val_sims", np.asarray(sims_written, dtype=np.int64))
    _zarr_write_array(root, "lengths", np.asarray(lengths, dtype=np.int64))
    _zarr_write_array(root, "H_per_sim", np.asarray(H_list, dtype=np.int64))
    _zarr_write_array(root, "W_per_sim", np.asarray(W_list, dtype=np.int64))
    _zarr_write_array(root, "N_nodes_per_sim", np.asarray(N_nodes_list, dtype=np.int64))

    summary = {
        "zarr_path": zarr_path,
        "model_name": model_name,
        "n_sims": len(sims_written),
        "val_sims": sims_written,
        "lengths": lengths,
    }

    print(f"[done] wrote validation rollout zarr for model='{model_name}' -> {zarr_path} (n_sims={len(sims_written)})")

    return summary


# ===============================
# Post-hoc augmentation: adds the ground-truth node displacements
# (u_true_nodes) and the resulting per-timestep surface displacement RMSE
# (disp_err_pointwise_rmse) to an already-written rollout zarr, so accuracy
# can be computed without re-running any rollout.
# ===============================
def compute_surface_disp_rmse_trajectory(u_pred: np.ndarray, u_true: np.ndarray) -> np.ndarray:
    u_pred = np.asarray(u_pred, dtype=np.float64)
    u_true = np.asarray(u_true, dtype=np.float64)
    if u_pred.shape != u_true.shape:
        raise ValueError(f"Shape mismatch: u_pred {u_pred.shape} vs u_true {u_true.shape}")
    if u_pred.ndim != 3 or u_pred.shape[2] != 3:
        raise ValueError(f"Expected shape (T,N,3), got {u_pred.shape}")

    err = u_pred - u_true
    err_sq = np.sum(err**2, axis=2)
    rmse_t = np.sqrt(np.mean(err_sq, axis=1))
    return rmse_t


def add_surface_disp_rmse_to_rollout_zarr(
    rollout_zarr_path: str,
    zarr_disp_path: str = ZARR_DISP,
    snapshots_key: str = "snapshots",
    print_every: int = 20,
    overwrite: bool = True,
):
    root = zarr.open_group(rollout_zarr_path, mode="a")
    if "simulations" not in root:
        raise KeyError(f"No 'simulations' group found in {rollout_zarr_path}")

    sims_grp = root["simulations"]
    sim_keys = sorted(list(sims_grp.group_keys()))
    if len(sim_keys) == 0:
        raise ValueError(f"No simulation groups found in {rollout_zarr_path}/simulations")

    model_name = root.attrs.get("model_name", "model")

    g_disp = zarr.open_group(zarr_disp_path, mode="r")
    if snapshots_key not in g_disp:
        raise KeyError(f"Could not find '{snapshots_key}' in {zarr_disp_path}")
    snaps = g_disp[snapshots_key]

    g_design = zarr.open_group(DESIGN_ZARR_PATH, mode="r")
    mgrp = g_design["Mapping_indexes_and_metadata"]
    sim_index_all = np.asarray(mgrp["sim_index"], dtype=np.int64)
    time_vals_all = np.asarray(mgrp["time_vals"], dtype=np.float64)
    S = sim_index_all.shape[0]

    if snaps.ndim != 2:
        raise ValueError(f"Expected snapshots to be 2D, got shape {snaps.shape}")

    if snaps.shape[0] == S:
        snapshot_major = True
        M = snaps.shape[1]
    elif snaps.shape[1] == S:
        snapshot_major = False
        M = snaps.shape[0]
    else:
        raise ValueError(f"Snapshots shape {snaps.shape} doesn't match S={S} in either axis")

    if M % 3 != 0:
        raise ValueError(f"Expected M divisible by 3, got M={M}")
    N = M // 3

    def load_true_u_nodes_for_sim_fast(sim_id: int):
        idx = np.where(sim_index_all == sim_id)[0]
        if idx.size == 0:
            raise ValueError(f"No snapshots found for sim_id={sim_id}")

        t_s = time_vals_all[idx]
        order = np.argsort(t_s)
        idx_sorted = idx[order]
        t_true = t_s[order]

        if snapshot_major:
            U_flat = np.asarray(snaps[idx_sorted, :], dtype=np.float64)
        else:
            U_flat = np.asarray(snaps[:, idx_sorted], dtype=np.float64).T

        ux = U_flat[:, 0:N]
        uy = U_flat[:, N:2*N]
        uz = U_flat[:, 2*N:3*N]
        U_true = np.stack([ux, uy, uz], axis=2)

        return t_true, U_true.astype(np.float32)

    n_sims = len(sim_keys)
    t0 = _time.time()

    for i, sk in enumerate(sim_keys, start=1):
        sim_grp = sims_grp[sk]
        sim_id = int(sim_grp.attrs.get("sim_id", int(sk.split("_")[-1])))

        if (i == 1) or (print_every > 0 and (i % print_every == 0 or i == n_sims)):
            elapsed = _time.time() - t0
            sims_done = max(i - 1, 1) if i > 1 else 1
            rate = elapsed / sims_done
            eta = rate * (n_sims - (i - 1))
            print(
                f"[{model_name}] augment sim {i}/{n_sims} "
                f"(sim_id={sim_id}) | elapsed={elapsed/60:.2f} min | "
                f"avg={rate:.2f} s/sim | ETA={eta/60:.2f} min"
            )

        if "u_pred_nodes" not in sim_grp:
            raise KeyError(f"{sk} missing 'u_pred_nodes'")

        u_pred = np.asarray(sim_grp["u_pred_nodes"], dtype=np.float32)
        _, u_true = load_true_u_nodes_for_sim_fast(sim_id)

        if u_pred.shape != u_true.shape:
            raise ValueError(f"{sk}: u_pred shape {u_pred.shape} != u_true shape {u_true.shape}")

        rmse_t = compute_surface_disp_rmse_trajectory(u_pred, u_true)

        _zarr_write_array(sim_grp, "u_true_nodes", u_true, overwrite=overwrite)
        _zarr_write_array(sim_grp, "disp_err_pointwise_rmse", rmse_t, overwrite=overwrite)

        sim_grp.attrs["final_disp_err_pointwise_rmse"] = float(rmse_t[-1])
        sim_grp.attrs["max_disp_err_pointwise_rmse"] = float(np.max(rmse_t))

    total_elapsed = _time.time() - t0
    root.attrs["has_u_true_nodes"] = True
    root.attrs["has_disp_err_pointwise_rmse"] = True

    print(f"[{model_name}] done augmenting zarr | total elapsed = {total_elapsed/60:.2f} min")
