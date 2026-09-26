"""Shared evaluation utilities used by eval_model_a.py, eval_model_b.py,
eval_model_c.py, eval_model_d.py, and cross_model_comparison.py.

Covers the pieces of the FEM -> POD -> Neural ODE pipeline that are common
across all four model variants (A/B/C/D, which differ in how much
growth-state feedback reaches the latent dynamics): mesh/grid loading and
bottom-surface mapping, the POD displacement decoder, growth-parameter
lookup, the Model D CNN growth-encoder + VelocityNet architecture and its
checkpoint loader, zarr I/O for validation rollouts, and the RMSE/plotting
helpers used to build the paper's validation-rollout and surface-displacement
error figures across the 185 held-out validation simulations.

Relative to the internal development version of this module, an older,
superseded explicit-Euler growth integrator (and an accompanying CNN-NODE
rollout function that depended on it) has been removed: every eval_model_*.py
script instead uses its own validated, UMAT-matched implicit-Newton growth
integrator, defined independently in each of those files. The surface-
displacement RMSE augmentation and plotting helpers
(add_surface_disp_rmse_to_rollout_zarr, compute_surface_disp_rmse_trajectory,
plot_all_val_surface_disp_rmse_trajectories) were also consolidated into this
module, since they used to be duplicated in a separate helper module.

This module performs no top-level execution or I/O on import (safe to import).
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
import matplotlib
matplotlib.use("Agg")  # headless backend -- must be set before any pyplot import
import matplotlib.pyplot as plt


# ===============================
# Paths & templates
# ===============================
ZARR_DISP = "displacements.zarr"
ZARR_SDV = "ip_growth_elem.zarr"          # lambda_g_x, lambda_g_y latents
ZARR_VOL = "expd_volumes.zarr"
DESIGN_ZARR_PATH = "Design_and_Metadata.zarr"

U_LATENT_TEMPLATE = "latent_displ_r{r}/coeffs"

# Group/array names for the POD (proper orthogonal decomposition) basis used
# to decode reduced displacement coordinates back to full nodal displacements.
DISP_POD_GROUP = "pod_full"
DISP_POD_U_NAME = "U"
DISP_POD_MEAN_NAME = "mean"

NODES_CSV = "GOH_Nodes_Test_for_Visualization.csv"
ELEMS_CSV = "GOH_Elements_Test_for_Visualization.csv"

# Reference (undeformed) in-plane area of a single bottom-surface element
# face, in cm^2. Used to convert per-element growth stretches into a net
# tissue area gain.
A0_CM2 = 0.25

SEED = 123


def set_seed(seed: int = SEED):
    """Seed numpy/torch (including CUDA) and force deterministic cuDNN
    kernels, so that model rollouts are reproducible across runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===============================
# 185-sim validation set: the fixed list of held-out simulation IDs used for
# all validation-rollout figures and cross-model comparisons in the paper.
# This list is shared across all four model variants so that per-sim errors
# are directly comparable.
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
    """Return (H, W, bottom_elem_idx) for the bottom-surface element grid,
    computing it once via build_bottom_surface_elem_grid_mapping and caching
    the result in a module-level global (avoids re-reading/re-gridding the
    mesh CSVs on every call)."""
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
    """Build a regular (H, W) grid mapping over the bottom surface of the
    skin mesh, needed to arrange per-element quantities (e.g. growth
    stretches) into image-like grids that the CNN growth encoder (Model D)
    or grid-based plots consume.

    For every element, the (x, y) location of its centroid is snapped to a
    quantized grid (tolerance tol_xy) to build a unique set of (x, y)
    columns/rows. For each grid cell, the element with the smallest z
    centroid (i.e. the lowest of the two element layers through the skin
    thickness) is selected as the "bottom surface" element for that cell --
    this is a different bottom-surface selection rule than the "second half
    of the element array" convention used elsewhere in this module (which
    relies on the two-layer element ordering in the growth/SDV zarr arrays).

    Returns
    -------
    H, W : grid dimensions.
    bottom_elem_idx : (H*W,) array mapping flattened grid cell -> element
        index (into the elems_csv ordering).
    cell_xy, elem_id, node_id, node_xyz, elem_conn_local, xs, ys : supporting
        arrays (element centroid coords, IDs, node coordinates/connectivity,
        and the sorted unique grid coordinates) for callers that need them.
    """
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
    """Look up the growth-model design parameters (theta_crit, k1, k2) that
    were used to generate a given simulation, by mapping sim_id -> job_id
    (via the design zarr's job_index_per_sim) -> row of the design-of-
    experiments text file. Returns (job_id, theta_crit, k1, k2)."""
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
    """Load the per-snapshot arrays needed to train/evaluate the NODE
    models for a given POD truncation r_modes: the reduced displacement
    latent coordinates, expander volume, design parameters, and the
    sim_index/time_vals bookkeeping that maps each snapshot row to a
    (simulation, time) pair.

    Returns a tuple:
        U_lat       -- (S, r_modes) reduced displacement coefficients.
        volume_snap -- (S,) true expander volume at each snapshot.
        sim_index   -- (S,) simulation ID for each snapshot row.
        time_vals   -- (S,) simulation time for each snapshot row.
        volume_SP   -- (S,) volume setpoint (control input) at each snapshot.
        design_all  -- (S, n_design) design parameters for each snapshot.
        n_sims      -- number of unique simulations covered (len of
                       np.unique(sim_index)).
    """
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
    """Load the bottom-surface sub-mesh (nodes + element connectivity)
    directly from the full-thickness mesh CSVs, using the convention that
    the element list is ordered as two layers through the skin thickness
    with the top layer listed first and the bottom layer second -- so the
    bottom-surface elements are simply the second half of elem_node_ids.

    Returns (node_xyz_bot, elem_conn_bot, node_ids_bot, elem_node_ids_bot):
    bottom-surface node coordinates, element connectivity re-indexed into
    the bottom-only node ordering, the corresponding original node IDs
    (sorted), and the original (un-reindexed) bottom element connectivity.
    """
    nodes = np.loadtxt(nodes_csv, delimiter=",")
    elems = np.loadtxt(elems_csv, delimiter=",")

    node_ids = nodes[:, 0].astype(np.int64)
    node_xyz = nodes[:, 1:4].astype(np.float64)

    elem_node_ids = elems[:, 1:5].astype(np.int64)
    Ne = elem_node_ids.shape[0]
    if Ne % 2 != 0:
        raise ValueError(f"Expected even number of elements (2 layers), got Ne={Ne}")

    half = Ne // 2
    # Bottom surface = second half of the element array (top layer is listed
    # first in elems_csv).
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
    """Standardize a growth-stretch grid (channel-wise: gx, gy) to zero mean
    / unit variance before it is fed into the CNN growth encoder.

    Accepts either a single grid (2, H, W) or a batch (B, 2, H, W); g_mean
    and g_std may be a scalar (applied to both channels) or a length-2
    array (one value per growth-stretch channel, gx and gy).
    """
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
    """Load the time history of element-wise growth stretches (lambda_g_x,
    lambda_g_y, stored as Abaqus state-dependent variables SDV1/SDV4) for
    one simulation, sorted in time order.

    Returns (t_sim, Gx_sim, Gy_sim, idx_sorted): time points, the (T, Ne)
    growth-stretch arrays for each in-plane direction, and the snapshot
    row indices (into the full sim_index/time_vals arrays) they came from.
    Loaded snapshot arrays are cached per (zarr path, folder) since they are
    reused across many sim_ids.
    """
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
    """Build and return a `decode(alpha)` closure that maps reduced POD
    displacement coordinates back to full nodal displacements.

    Loads the truncated POD basis U_r (first r_modes columns of the full
    basis) and mean displacement vector `mu` from the zarr POD group, then
    returns a function such that
        u_flat = mu + U_r @ alpha                # (3*N,) flattened field
    which is split into the three displacement components. The flattened
    layout is node-major-within-component, i.e. the first N entries are all
    nodes' x-displacements ("ux block"), the next N are y ("uy block"), and
    the final N are z ("uz block") -- NOT interleaved per-node [x,y,z].
    """
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
        # Flattened layout is [ux_block, uy_block, uz_block], each of
        # length N (not interleaved per node).
        ux = u_flat[0:N]
        uy = u_flat[N:2*N]
        uz = u_flat[2*N:3*N]
        return np.stack([ux, uy, uz], axis=1)

    return decode


# ===============================
# Growth-related geometry helpers
#
# Note: the numba-accelerated explicit-Euler growth integrator that used to
# live in this section (which stepped the growth stretch ODE forward using
# these bilinear-quad shape functions) has been removed, since it is
# superseded by the implicit-Newton integrator defined in each
# eval_model_*.py script. shape_function_gradients is kept here as it is
# still a standalone geometry utility.
# ===============================
def shape_function_gradients(xi, eta):
    """Bilinear (4-node quad) isoparametric shape function derivatives
    d N_a / d xi and d N_a / d eta at local coordinates (xi, eta) in
    [-1, 1]^2, node ordering a = 1..4 counter-clockwise. Used for
    element-level interpolation/quadrature on the bottom-surface quad mesh.
    """
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


def compute_net_area_gain_from_lamdag_elem(lamdag_elem: np.ndarray, A0_cm2: float = A0_CM2) -> float:
    """Sum, over all bottom-surface elements, the net in-plane area gained
    due to growth. `lamdag_elem` (short for lambda_g, the growth stretch)
    is an (Ne, 2) array of [lambda_g_x, lambda_g_y] per element; each
    element's deformed area is A0_cm2 * lambda_g_x * lambda_g_y, so its net
    gain relative to the reference area A0_cm2 is A0_cm2 * (gx*gy - 1)."""
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
    """Compute the ground-truth net area gain at the final time point of a
    simulation, from the raw (FEM-derived) element growth stretches, for
    comparison against the model-predicted area gain.

    Uses the "bottom surface = second half of the element array" convention
    (Ne_full elements split into two thickness layers; the bottom layer is
    elements[half:]), consistent with the H*W bottom-surface grid.
    """
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
    half = Ne_full // 2  # bottom surface = second half of the element array

    gx = Gx_sim_elem[final_k, half:]
    gy = Gy_sim_elem[final_k, half:]

    if gx.shape[0] != H * W:
        raise ValueError(f"Expected bottom elems H*W={H*W}, got {gx.shape[0]}")

    lamdag_elem_true = np.stack([gx, gy], axis=1).astype(np.float64)
    return compute_net_area_gain_from_lamdag_elem(lamdag_elem_true, A0_cm2=A0_cm2)


# ===============================
# Model definition (Model D: CNN growth encoder + VelocityNet)
#
# Model D is the most information-rich of the four NODE variants: instead of
# a scalar or POD-projected growth summary, it feeds the full 2-channel
# (lambda_g_x, lambda_g_y) growth-stretch grid over the bottom surface
# through a small CNN to get a growth feature vector, which is concatenated
# with the reduced displacement state before predicting the state's time
# derivative (the "velocity" in VelocityNet).
# ===============================
class GrowthEncoderCNN(nn.Module):
    """Small conv-net that encodes a (B, C, H, W) growth-stretch grid (the
    bottom-surface lambda_g_x / lambda_g_y grid, optionally with extra
    channels) into a fixed-length feature vector of size dim_gfeat, via
    three stride-2 conv+GroupNorm+SiLU blocks, global average pooling, and
    a linear head."""

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
        # Allow a single-channel grid to be passed without an explicit
        # channel dim: (B,H,W) -> (B,1,H,W).
        if G.dim() == 3:
            G = G.unsqueeze(1)
        if G.dim() != 4:
            raise ValueError(f"Expected G with shape [B,C,H,W], got {tuple(G.shape)}")
        return self.head(self.cnn(G))


class VelocityNet(nn.Module):
    """The NODE right-hand side dx/dt = f(x, ...) for Model D. Concatenates
    the base state (reduced displacement coordinates + volume setpoint +
    design parameters + tracking-error/integral terms used for closed-loop
    control) with the CNN-encoded growth feature vector, and passes the
    result through a small tanh MLP to predict the state's time
    derivative."""

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
        # x_base: reduced state + control/design features. G: growth grid.
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
# Zarr export: writes closed-loop model rollouts (predicted latent state,
# decoded displacements, growth grids, volume/error trajectories, etc.) for
# a batch of validation simulations into a single zarr store, using a
# schema shared across all four model variants so that downstream
# comparison/plotting code doesn't need per-model special-casing.
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
    """Fallback net-area-gain time history, used when a rollout dict does
    not already provide "Ag_hist_raw_cm2". Given the predicted growth
    stretches lamdag_stack (T, Ne, 2) = [lambda_g_x, lambda_g_y] per
    bottom-surface element at each of T time steps, returns the (T,) net
    area gain A0_cm2 * sum_e(gx_e * gy_e - 1) at each time step."""
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
    """Run a closed-loop model rollout (via `rollout_fn`) on every sim in
    `val_sims` and write the results to a single zarr store at `zarr_path`.

    `rollout_fn` is expected to be one of the per-model rollout functions
    (defined in eval_model_a/b/c/d.py), called as
    `rollout_fn(r_modes=r_modes, sim_id=sim_id, **rollout_kwargs)` and
    returning a dict with keys such as "t", "z_pred" (predicted reduced
    state trajectory), "u_pred_nodes" (decoded nodal displacements),
    "g_pred_grid"/"lamdag_pred_elem" (predicted growth grids/stretches),
    and optional ground-truth/error fields ("z_true", "vol_pred",
    "vol_true", "err_L2", "err_vol", "e_hist", "I_hist").

    For each simulation this writes one "sim_XXXXX" group containing those
    arrays (plus a derived net-area-gain history "Ag_pred_cm2") and some
    scalar attrs (final errors, area gain, mesh dims). Top-level datasets
    "val_sims"/"lengths"/"H_per_sim"/"W_per_sim"/"N_nodes_per_sim" record
    per-simulation metadata for quick indexing without opening every group.

    Returns a small summary dict (zarr_path, model_name, n_sims, val_sims,
    lengths); the full results live in the zarr store itself.
    """
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
# Post-hoc augmentation: adds ground-truth nodal displacements
# (u_true_nodes) and the pointwise surface displacement RMSE
# (disp_err_pointwise_rmse) to an existing rollout zarr written by
# save_val_rollouts_to_zarr, for use in the paper's error-trajectory plots.
# ===============================
def compute_surface_disp_rmse_trajectory(u_pred: np.ndarray, u_true: np.ndarray) -> np.ndarray:
    """Per-time-step RMSE between predicted and true nodal displacements,
    both of shape (T, N, 3) (T time steps, N surface nodes, xyz). At each
    time step, error is pooled over nodes and spatial components:
    rmse_t = sqrt(mean_over_nodes(||u_pred - u_true||^2)). Returns a (T,)
    array."""
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
    """For every simulation group in an existing rollout zarr (written by
    save_val_rollouts_to_zarr), look up the true nodal displacements from
    the raw displacement snapshots zarr, compute the pointwise RMSE against
    the model's predicted "u_pred_nodes", and write both "u_true_nodes" and
    "disp_err_pointwise_rmse" back into that simulation's group (in place).

    Also records "final_disp_err_pointwise_rmse" / "max_disp_err_pointwise_
    rmse" attrs per simulation, and sets "has_u_true_nodes" / "has_disp_err_
    pointwise_rmse" flags at the root so callers can check whether this
    augmentation has already been run.
    """
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
        # Gather this sim's snapshot rows, sort by time, and reshape the
        # flattened [ux_block, uy_block, uz_block] displacement vector into
        # (T, N, 3) -- same flattening convention as load_disp_decoder.
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


def plot_all_val_surface_disp_rmse_trajectories(
    rollout_zarr_path: str,
    use_relative_time: bool = False,
    linewidth: float = 0.9,
    alpha: float = 0.18,
    show_all_curves: bool = True,
    summary_linewidth: float = 3.0,
    band_alpha: float = 0.22,
    n_interp: int = 300,
    use_seaborn_style: bool = True,
    ylim=(0, 10),
    highlight_sim_id=None,
    highlight_color: str = "lightblue",
    highlight_linewidth: float = 2.4,
    highlight_alpha: float = 1.0,
    highlight_label: str = "Selected sim",
    save_path=None,
    save_dpi: int = 400,
):
    """Plot surface-displacement RMSE vs. time for every validation
    simulation in a rollout zarr (must already have "disp_err_pointwise_
    rmse", via add_surface_disp_rmse_to_rollout_zarr), overlaid with the
    median trajectory and a 10th-90th percentile band -- the figure style
    used for the paper's validation-error trajectory plots.

    Individual per-sim curves are optionally thinned/faded (show_all_curves,
    alpha) and one simulation can be highlighted (highlight_sim_id). Curves
    are resampled onto a common time (or, if use_relative_time, normalized
    [0,1] progress) axis via linear interpolation so percentiles can be
    computed pointwise across simulations of differing length. If
    save_path is given, the figure is saved there; the figure is always
    closed before returning (nothing is returned).
    """
    if use_seaborn_style:
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except Exception:
            plt.style.use("default")

    root = zarr.open_group(rollout_zarr_path, mode="r")
    if "simulations" not in root:
        raise KeyError(f"No 'simulations' group found in {rollout_zarr_path}")

    sims_grp = root["simulations"]
    sim_keys = sorted(list(sims_grp.group_keys()))
    if len(sim_keys) == 0:
        raise ValueError(f"No simulation groups found in {rollout_zarr_path}/simulations")

    model_name = root.attrs.get("model_name", "model")

    fig, ax = plt.subplots(figsize=(8.4, 5.6))

    traj_data = []
    all_t_min = []
    all_t_max = []

    for sk in sim_keys:
        g = sims_grp[sk]
        if "disp_err_pointwise_rmse" not in g:
            raise KeyError(
                f"{sk} missing 'disp_err_pointwise_rmse'. "
                f"Run add_surface_disp_rmse_to_rollout_zarr(...) first."
            )
        y = np.asarray(g["disp_err_pointwise_rmse"], dtype=np.float64)
        if "t" in g:
            t = np.asarray(g["t"], dtype=np.float64)
        else:
            t = np.arange(y.shape[0], dtype=np.float64)
        if t.shape[0] != y.shape[0]:
            raise ValueError(f"{sk}: t length {t.shape[0]} != rmse length {y.shape[0]}")

        if use_relative_time:
            if len(t) == 1 or (t[-1] - t[0]) <= 0:
                x = np.zeros_like(t)
            else:
                x = (t - t[0]) / (t[-1] - t[0])
        else:
            x = t
            all_t_min.append(np.min(t))
            all_t_max.append(np.max(t))

        traj_data.append((sk, x, y))

    if use_relative_time:
        x_common = np.linspace(0.0, 1.0, n_interp)
    else:
        x_common = np.linspace(min(all_t_min), max(all_t_max), n_interp)

    interp_bank = []
    highlighted_found = False

    for sk, x, y in traj_data:
        x_unique, idx_unique = np.unique(x, return_index=True)
        y_unique = y[idx_unique]
        if x_unique.size < 2:
            continue

        is_highlight = False
        if highlight_sim_id is not None:
            try:
                sim_id_this = int(sk)
                is_highlight = (sim_id_this == highlight_sim_id)
            except ValueError:
                g = sims_grp[sk]
                sim_id_attr = g.attrs.get("sim_id", None)
                if sim_id_attr is not None:
                    is_highlight = (int(sim_id_attr) == int(highlight_sim_id))

        if show_all_curves and not is_highlight:
            ax.plot(x_unique, y_unique, linewidth=linewidth, alpha=alpha)

        if is_highlight:
            ax.plot(
                x_unique, y_unique,
                color=highlight_color, linewidth=highlight_linewidth,
                alpha=highlight_alpha, label=f"{highlight_label} ({highlight_sim_id})", zorder=5,
            )
            highlighted_found = True

        y_interp = np.interp(x_common, x_unique, y_unique, left=np.nan, right=np.nan)
        interp_bank.append(y_interp)

    if len(interp_bank) == 0:
        raise ValueError("Not enough valid trajectories to compute summary statistics.")

    Y = np.vstack(interp_bank)

    y_p10 = np.nanpercentile(Y, 10, axis=0)
    y_med = np.nanpercentile(Y, 50, axis=0)
    y_p90 = np.nanpercentile(Y, 90, axis=0)

    ax.fill_between(x_common, y_p10, y_p90, color="tab:red", alpha=band_alpha, label="10th–90th percentile")
    ax.plot(x_common, y_med, color="black", linewidth=summary_linewidth, label="Median")

    if highlight_sim_id is not None and not highlighted_found:
        print(f"[warn] highlight_sim_id={highlight_sim_id} was not found in the zarr groups.")

    ax.set_title(f"Validation surface displacement RMSE trajectories | {model_name} | n={len(sim_keys)}")

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    print(f"Final median RMSE = {y_med[-1]:.6f}")
    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)
