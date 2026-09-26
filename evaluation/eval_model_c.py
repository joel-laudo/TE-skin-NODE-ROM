"""Standalone eval-time rollout module for Model C (POD/PCA growth-feedback NODE).

Of the paper's four Neural ODE growth-feedback variants (Models A-D), all four
share the same latent-displacement update z_{k+1} = z_k + dt_k * f(x_k); they
differ only in how much information about the current growth state is folded
into f()'s input. Model A gets none, Model B gets a scalar growth summary,
Model D gets a CNN-compressed growth field, and Model C (this file) gets a
growth field compressed with a separate, pre-trained POD basis. That POD
compression is a SECOND, independent POD transform living in this file
(see `encode_growth_pca_torch` / `load_growth_pca_encoder_torch` below) on top
of the displacement-POD decoder (`load_disp_decoder`) shared with the other
model families: one maps the reduced latent displacement state back out to
physical node coordinates, the other maps the physical per-element growth
field down to the low-dimensional feature vector that gets concatenated onto
the NODE input.

This module returns the "full" rollout output (including "t"/"z_pred"/
"z_true"/"g_pred_grid" in addition to "u_pred_nodes"), so this single file can
serve both evaluation/cross_model_comparison.py (which only reads
"u_pred_nodes") and evaluation/generate_deliverables.py (which also needs the
extra fields to build the shared validation-rollout zarr / RMSE-trajectory
plots). Growth is integrated throughout with the matched, UMAT-consistent,
implicit-Newton numba integrator (`integrate_growth_matched_numba`), so
predicted growth stays numerically consistent with the finite-element UMAT
used to generate the training data.

Self-contained: no imports from other model families' eval modules.

Exposes: rollout_single_sim_pca_node(r_modes, sim_id, nodes_csv, elems_csv,
ckpt_path_template=..., make_plots=False) -> dict with "u_pred_nodes"
(T,N,3), "t", "z_pred", "z_true", "g_pred_grid" (T,2,H,W),
"lamdag_pred_elem", plus metadata fields.
"""
import os
import functools

import numpy as np
import torch
import torch.nn as nn
import zarr
from numba import njit, prange

# ===============================
# Paths & templates (shared with training/train_model_c.py / Model A)
# ===============================
ZARR_DISP  = "displacements.zarr"
ZARR_SDV   = "ip_growth_elem.zarr"
ZARR_VOL   = "expd_volumes.zarr"
DESIGN_ZARR_PATH = "Design_and_Metadata.zarr"

U_LATENT_TEMPLATE = "latent_displ_r{r}/coeffs"

DISP_POD_GROUP     = "pod_full"
DISP_POD_U_NAME    = "U"
DISP_POD_MEAN_NAME = "mean"

# Pre-fit PCA basis (mean/components) used to compress the bottom-surface
# growth field into Model C's low-dimensional feedback vector; see
# load_growth_pca_encoder_torch / encode_growth_pca_torch below.
DEFAULT_GROWTH_PCA_NPZ = "growth_pca_trainonly_H60_W60_k8_val0.20_seed123.npz"


# ===============================
# Mesh / grid utilities
# ===============================
# Process-wide cache: the bottom-surface element<->(row,col) grid mapping only
# depends on the fixed mesh, not on sim_id or r_modes, so it is computed once
# and reused across every rollout call.
_BOTTOM_GRID_CACHE = None


def get_bottom_grid_cache():
    """Return the cached (H, W, bottom_idx) bottom-surface grid mapping,
    building it on first use via build_bottom_surface_elem_grid_mapping."""
    global _BOTTOM_GRID_CACHE
    if _BOTTOM_GRID_CACHE is None:
        H, W, bottom_idx, *_ = build_bottom_surface_elem_grid_mapping(
            nodes_csv="GOH_Nodes_Test_for_Visualization.csv",
            elems_csv="GOH_Elements_Test_for_Visualization.csv",
            tol_xy=1e-8,
        )
        _BOTTOM_GRID_CACHE = (H, W, bottom_idx)
    return _BOTTOM_GRID_CACHE


def build_bottom_surface_elem_grid_mapping(
    nodes_csv: str = "GOH_Nodes_Test_for_Visualization.csv",
    elems_csv: str = "GOH_Elements_Test_for_Visualization.csv",
    tol_xy: float = 1e-8,
):
    """Build a regular (H, W) image-like grid over the mesh's bottom surface
    and, for each grid cell, pick the element whose centroid falls in that
    (x, y) column and has the smallest z (i.e. the bottom-layer element).

    This lets the per-element growth stretch field (and, elsewhere, other
    per-element quantities) be reshaped into a 2D array so it can be fed to
    the PCA/CNN growth-feedback encoders and plotted as an image. Returns
    (H, W, bottom_elem_idx, cell_xy).
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

    elem_conn_ids = elems[:, 1:5].astype(np.int64)
    Ne = elem_conn_ids.shape[0]

    elem_conn_local = np.empty((Ne, 4), dtype=np.int64)
    for a in range(4):
        elem_conn_local[:, a] = np.array(
            [node_id_to_local[int(nid)] for nid in elem_conn_ids[:, a]], dtype=np.int64
        )

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

    if not np.all(bottom_elem_idx >= 0):
        missing = np.where(bottom_elem_idx < 0)[0][:10]
        raise ValueError(f"Some grid cells have no assigned element. Example missing: {missing}")

    cell_xy = np.stack([xe[bottom_elem_idx], ye[bottom_elem_idx]], axis=1)
    return H, W, bottom_elem_idx, cell_xy


@functools.lru_cache(maxsize=None)
def load_bottom_surface_mesh_direct(nodes_csv: str, elems_csv: str):
    """Robust bottom-surface mesh loader (bottom layer = second half of element rows)."""
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
    if missing:
        raise ValueError(f"{len(missing)} node IDs missing from nodes_csv. Example: {missing[:10]}")

    node_ids_bot = np.sort(bottom_node_ids.astype(np.int64))
    node_xyz_bot = np.vstack([id2xyz[int(nid)] for nid in node_ids_bot]).astype(np.float64)

    id2bot = {int(nid): i for i, nid in enumerate(node_ids_bot)}
    get_idx = np.vectorize(lambda nid: id2bot.get(int(nid), -1), otypes=[np.int64])
    elem_conn_bot = get_idx(elem_node_ids_bot)

    bad = np.where(elem_conn_bot < 0)
    if bad[0].size > 0:
        raise ValueError("Internal mapping failure building bottom-surface mesh.")

    return node_xyz_bot, elem_conn_bot, node_ids_bot, elem_node_ids_bot


# ===============================
# Raw-data / growth-params loaders
# ===============================
@functools.lru_cache(maxsize=None)
def load_raw_data(r_modes: int):
    """Load, for every simulation and snapshot, the ground-truth reduced
    (POD) latent displacement coefficients ("U_lat", using the first
    r_modes columns of the full POD basis), expander volume, volume
    setpoint schedule, and design parameters, plus the per-snapshot
    sim_index/time_vals bookkeeping needed to slice out one simulation's
    trajectory. Cached per r_modes since it is called once per rollout."""
    from pathlib import Path

    g_disp   = zarr.open_group(ZARR_DISP, mode="r")
    g_design = zarr.open_group(DESIGN_ZARR_PATH, mode="r")

    vol_root = Path(ZARR_VOL)

    u_path = U_LATENT_TEMPLATE.format(r=r_modes)
    U_lat = np.asarray(g_disp[u_path], dtype=np.float32).T  # (S, r)

    mgrp = g_design["Mapping_indexes_and_metadata"]
    sim_index = np.asarray(mgrp["sim_index"], dtype=np.int64)
    time_vals = np.asarray(mgrp["time_vals"], dtype=np.float64)

    volume_SP = np.asarray(
        zarr.open_array(str(vol_root / "volume_setpoint_per_snapshot"), mode="r"), dtype=np.float32
    )
    design_all = np.asarray(g_design["design_params_per_snapshot"], dtype=np.float32)

    S = U_lat.shape[0]
    assert sim_index.shape[0] == S == time_vals.shape[0]
    assert volume_SP.shape[0] == S == design_all.shape[0]

    volume_snap = np.asarray(
        zarr.open_array(str(vol_root / "expander_volume" / "cvol"), mode="r"), dtype=np.float32
    )
    if volume_snap.ndim != 1 or volume_snap.shape[0] != S:
        raise ValueError(f"Expected cvol as (S,) with S={S}, got shape {volume_snap.shape}")

    return U_lat, volume_snap, sim_index, time_vals, volume_SP, design_all, len(np.unique(sim_index))


@functools.lru_cache(maxsize=None)
def get_growth_params_for_sim(sim_id: int):
    """Look up the growth-law parameters (critical stretch theta_crit and
    the two in-plane growth rate constants k1, k2) used by the finite-element
    UMAT for this simulation, via the job index that maps sim_id to a row of
    the design-of-experiments table."""
    g_disp = zarr.open("Design_and_Metadata.zarr", mode="r")
    job_index_per_sim = g_disp["Mapping_indexes_and_metadata"]["job_index_per_sim"][:]
    job_id = int(job_index_per_sim[sim_id])

    design_file = "PGOH_Ideal_tol_Tcrit_V_mu_kk1_kk2_kappa_k1_design.txt"
    design_table = np.loadtxt(design_file, skiprows=0)

    row_index = job_id - 1
    if row_index < 0 or row_index >= design_table.shape[0]:
        raise IndexError(f"Job ID {job_id} out of range for design table with {design_table.shape[0]} rows.")

    row = design_table[row_index]
    theta_crit = float(row[2])
    k1 = float(row[5])
    k2 = float(row[6])
    return job_id, theta_crit, k1, k2


_GROWTH_SNAP_CACHE = {}


def load_elem_growth_for_sim(
    sim_id: int,
    sim_index: np.ndarray,
    time_vals: np.ndarray,
    zarr_sdv_path: str = "ip_growth_elem.zarr",
    folder_x: str = "snaps_SDV1",
    folder_y: str = "snaps_SDV4",
):
    """Load the raw, ground-truth per-element growth stretches for one
    simulation, sorted in time. "SDV1"/"SDV4" are the UMAT's state-variable
    slots for the two in-plane principal growth stretches (x- and
    y-direction lambda_g); returns (t_sim, Gx_sim, Gy_sim, idx_sorted) where
    Gx_sim/Gy_sim have shape (T, Ne_full) over ALL elements (both mesh
    layers, not just the bottom surface)."""
    g_sdv = zarr.open_group(zarr_sdv_path, mode="r")

    def _load_snap_folder(folder_name: str) -> np.ndarray:
        cache_key = (zarr_sdv_path, folder_name)
        if cache_key in _GROWTH_SNAP_CACHE:
            return _GROWTH_SNAP_CACHE[cache_key]
        candidates = [f"{folder_name}/{folder_name}", f"{folder_name}/data", f"{folder_name}/snaps", f"{folder_name}"]
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
    Gx_all = Gx_raw.T if Gx_raw.shape[1] == S else Gx_raw
    Gy_all = Gy_raw.T if Gy_raw.shape[1] == S else Gy_raw

    Gx_sim = Gx_all[idx_sorted, :]
    Gy_sim = Gy_all[idx_sorted, :]
    return t_sim, Gx_sim, Gy_sim, idx_sorted


@functools.lru_cache(maxsize=None)
def load_disp_decoder(
    r_modes: int,
    zarr_disp_path: str = "displacements.zarr",
    pod_group: str = "pod_full",
    pod_u_name: str = "U",
    pod_mean_name: str = "mean",
):
    """Returns a callable decode(alpha)->u_nodes (N,3), numpy in/out."""
    g_disp = zarr.open_group(zarr_disp_path, mode="r")
    U_key = f"{pod_group}/{pod_u_name}"
    mu_key = f"{pod_group}/{pod_mean_name}"

    U_full = np.asarray(g_disp[U_key])
    mu = np.asarray(g_disp[mu_key]).reshape(-1)

    M, Rmax = U_full.shape
    if r_modes > Rmax:
        raise ValueError(f"Requested r_modes={r_modes} exceeds available Rmax={Rmax}")
    if M % 3 != 0:
        raise ValueError(f"Expected M divisible by 3. Got M={M}")
    N = M // 3
    U_r = U_full[:, :r_modes]

    def decode(alpha: np.ndarray) -> np.ndarray:
        alpha = np.asarray(alpha).reshape(-1)
        if alpha.shape[0] != r_modes:
            raise ValueError(f"alpha must have shape ({r_modes},), got {alpha.shape}")
        u_flat = mu + U_r @ alpha
        ux = u_flat[0:N]; uy = u_flat[N:2 * N]; uz = u_flat[2 * N:3 * N]
        return np.stack([ux, uy, uz], axis=1)

    return decode


def shape_function_gradients(xi, eta):
    """Derivatives (w.r.t. the two in-plane parent coordinates xi, eta) of the
    standard bilinear shape functions for a 4-node quadrilateral element.
    Used below at the element center (xi=eta=0), i.e. one-point reduced
    integration, to build the deformation gradient in compute_F_batch_numba."""
    dN_dxi = np.array([-(1 - eta) / 4, (1 - eta) / 4, (1 + eta) / 4, -(1 + eta) / 4])
    dN_deta = np.array([-(1 - xi) / 4, -(1 + xi) / 4, (1 + xi) / 4, (1 - xi) / 4])
    return dN_dxi, dN_deta


# ===============================
# PCA growth-feedback encoder (reused verbatim from training/train_model_c.py)
# ===============================
def load_growth_pca_encoder_torch(npz_path: str, device: torch.device, dtype: torch.dtype = torch.float32):
    """Load the pre-fit PCA basis for compressing the bottom-surface growth
    field (see DEFAULT_GROWTH_PCA_NPZ) and move it onto `device` as torch
    tensors, for use by encode_growth_pca_torch during rollout."""
    ck = np.load(npz_path)
    enc = {
        "H": int(ck["H"]),
        "W": int(ck["W"]),
        "mean_x": torch.tensor(np.asarray(ck["mean_x"], dtype=np.float32).reshape(1, -1), dtype=dtype, device=device),
        "components_x": torch.tensor(np.asarray(ck["components_x"], dtype=np.float32), dtype=dtype, device=device),
        "mean_y": torch.tensor(np.asarray(ck["mean_y"], dtype=np.float32).reshape(1, -1), dtype=dtype, device=device),
        "components_y": torch.tensor(np.asarray(ck["components_y"], dtype=np.float32), dtype=dtype, device=device),
        "pca_feat_mean": torch.tensor(np.asarray(ck["pca_feat_mean"], dtype=np.float32).reshape(1, -1), dtype=dtype, device=device),
        "pca_feat_std": torch.tensor(np.asarray(ck["pca_feat_std"], dtype=np.float32).reshape(1, -1), dtype=dtype, device=device),
    }
    kx = enc["components_x"].shape[0]
    ky = enc["components_y"].shape[0]
    enc["dim_g"] = int(kx + ky)

    H_map, W_map, bottom_idx = get_bottom_grid_cache()
    if H_map != enc["H"] or W_map != enc["W"]:
        raise ValueError(f"Bottom grid cache shape ({H_map},{W_map}) != PCA npz ({enc['H']},{enc['W']})")
    enc["bottom_idx"] = torch.tensor(np.asarray(bottom_idx, dtype=np.int64), dtype=torch.long, device=device)
    return enc


def encode_growth_pca_torch(lamdag_elem: torch.Tensor, pca_enc: dict) -> torch.Tensor:
    """Compress the current per-element growth stretch field into Model C's
    low-dimensional feedback vector g_pca.

    This is the growth-side counterpart of `load_disp_decoder`'s POD decode:
    here the growth field lambda_g ("lamdag", the current growth stretch in
    each in-plane principal direction) is PROJECTED ONTO a separately-fit PCA
    basis (x- and y-direction components fit independently) and then
    standardized, rather than reconstructed from one. The result is what
    gets concatenated onto the NODE state in rollout_single_sim_pca_node.
    lamdag_elem may be passed either as bottom-surface-only (N_bottom, 2) or
    as the full two-layer element array, in which case `bottom_idx` (from
    get_bottom_grid_cache) selects the bottom-layer rows first.
    """
    if lamdag_elem.ndim != 2 or lamdag_elem.shape[1] != 2:
        raise ValueError(f"Expected lamdag_elem shape (N,2), got {tuple(lamdag_elem.shape)}")

    H = pca_enc["H"]; W = pca_enc["W"]
    N_bottom = H * W

    if lamdag_elem.shape[0] == N_bottom:
        gx_bottom = lamdag_elem[:, 0]
        gy_bottom = lamdag_elem[:, 1]
    else:
        bottom_idx = pca_enc["bottom_idx"]
        gx_bottom = lamdag_elem[bottom_idx, 0]
        gy_bottom = lamdag_elem[bottom_idx, 1]

    gx_flat = gx_bottom.reshape(1, -1)
    gy_flat = gy_bottom.reshape(1, -1)

    x_scores = (gx_flat - pca_enc["mean_x"]) @ pca_enc["components_x"].T
    y_scores = (gy_flat - pca_enc["mean_y"]) @ pca_enc["components_y"].T

    feat_raw = torch.cat([x_scores, y_scores], dim=1)
    feat_norm = (feat_raw - pca_enc["pca_feat_mean"]) / (pca_enc["pca_feat_std"] + 1e-8)
    return feat_norm


# ===============================
# Matched, UMAT-consistent growth integrator (validated to 7.6e-13 max abs
# diff against the autograd-differentiable torch version used at train time)
# ===============================
def _growth_gate(time):
    """Growth on/off switch as a function of simulation time.

    The underlying FE simulations alternate ~weekly (7-time-unit) windows of
    inactive vs. active growth (matching the periodic loading/rest protocol
    used to generate the training data). Returns 0.0 (growth frozen) during
    the listed windows and 1.0 (growth active) otherwise; multiplies the
    growth rate in integrate_growth_matched_numba below.
    """
    if (0.0 < time < 7.0) or (14.0 < time < 21.0) or (28.0 < time < 35.0) or \
       (42.0 < time < 49.0) or (56.0 < time < 63.0) or (70.0 < time < 77.0) or \
       (84.0 < time < 91.0) or (98.0 < time < 105.0):
        return 0.0
    return 1.0
_growth_gate_jit = njit(_growth_gate)


@njit(parallel=True, fastmath=True)
def integrate_growth_matched_numba(F_batch, elem_lamdag, k1, k2, theta_crit, time, dt, tol=1e-12, maxiter=20):
    """Advance the per-element growth stretches ("lamdag" = lambda_g, the
    growth stretch) by one implicit (backward-Euler) time step, using a local
    Newton-Raphson iteration per element -- the same discretization the
    finite-element UMAT uses, so this integrator stays bit-for-bit consistent
    with the FE ground truth (validated to ~7.6e-13 max abs difference
    against the autograd-differentiable torch version used at train time).

    For each element and each in-plane principal direction i in {1,2}:
      elastic stretch  lam_i        = column norm of the deformation
                                       gradient F_batch[ei] (from
                                       compute_F_batch_numba)
      growth criterion phig_i       = lam_i / lamdag_i - theta_crit
      growth ODE (if phig_i > 0)    d(lamdag_i)/dt = k_i * phig_i
    i.e. growth only proceeds once the elastic stretch relative to the
    current growth stretch exceeds the critical threshold theta_crit; the
    Newton loop below solves the implicit update
      lamdag_i^{n+1} - lamdag_i^n - dt * k_i*(lam_i/lamdag_i^{n+1} - theta_crit) = 0
    for lamdag_i^{n+1}. `_growth_gate_jit` freezes growth (holds lamdag fixed)
    during the scheduled rest windows.
    """
    Ne = F_batch.shape[0]
    out = elem_lamdag.copy()
    gate = _growth_gate_jit(time)
    for ei in prange(Ne):
        F = F_batch[ei]
        # Column norms of F = principal stretches along the two in-plane
        # parent directions (xi, eta); F's columns are the pushed-forward
        # basis vectors of the reference in-plane/normal frame.
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


# --- F computation kernel (needed because integrate_growth_matched_numba
# takes F as an input; same shape-function-gradient element kinematics used
# by every other model family's eval script). ---
@njit
def _inv3x3(A):
    """Analytic 3x3 matrix inverse (cofactor expansion), used inline inside
    the parallel numba loop of compute_F_batch_numba below, where calling
    np.linalg.inv per-element would be slower / less portable across numba
    versions."""
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
def compute_F_batch_numba(node_coords, elem_conn, node_u, dN_dxi, dN_deta):
    """Compute the deformation gradient F (3x3, per element) for every
    bottom-surface shell element, from reference nodal coordinates
    (node_coords), element connectivity (elem_conn), and nodal displacements
    (node_u).

    These are 4-node shell/membrane elements with no explicit through-
    thickness node, so there is no shape-function derivative in the "normal"
    (out-of-plane) direction. Instead: Gxi/Geta (and their current-config
    counterparts gxi/geta) are the in-plane tangent vectors built from the
    bilinear shape-function derivatives (dN_dxi, dN_deta), and Nx,Ny,Nz /
    nx,ny,nz are unit normals obtained from their cross product in the
    reference and current configurations, respectively. F is then assembled
    so that it maps the reference in-plane tangents + normal onto their
    current-configuration counterparts (via the dual/reciprocal in-plane
    basis from inverting G, and a rank-1 normal-stretch term nx*Nx + ...).
    The resulting F feeds integrate_growth_matched_numba's principal-stretch
    calculation above.
    """
    Ne = elem_conn.shape[0]
    F_batch = np.empty((Ne, 3, 3), dtype=np.float64)
    for ei in prange(Ne):
        nids = elem_conn[ei]
        X = node_coords[nids, :]
        u = node_u[nids, :]
        x = X + u

        Gxi = np.zeros(3, dtype=np.float64)
        Geta = np.zeros(3, dtype=np.float64)
        gxi = np.zeros(3, dtype=np.float64)
        geta = np.zeros(3, dtype=np.float64)
        for a in range(4):
            Gxi[0]  += X[a,0]*dN_dxi[a];  Gxi[1]  += X[a,1]*dN_dxi[a];  Gxi[2]  += X[a,2]*dN_dxi[a]
            Geta[0] += X[a,0]*dN_deta[a]; Geta[1] += X[a,1]*dN_deta[a]; Geta[2] += X[a,2]*dN_deta[a]
            gxi[0]  += x[a,0]*dN_dxi[a];  gxi[1]  += x[a,1]*dN_dxi[a];  gxi[2]  += x[a,2]*dN_dxi[a]
            geta[0] += x[a,0]*dN_deta[a]; geta[1] += x[a,1]*dN_deta[a]; geta[2] += x[a,2]*dN_deta[a]

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

        F_batch[ei] = F
    return F_batch


def compute_net_area_gain_from_lamdag_elem(lamdag_elem: np.ndarray, A0_cm2: float = 0.25) -> float:
    """Sum, over bottom-surface elements, the in-plane area change implied by
    the two principal growth stretches (area scales as lamx*lamy relative to
    a common reference element area A0_cm2), giving the total grown skin
    area gain in cm^2."""
    lamx = lamdag_elem[:, 0]; lamy = lamdag_elem[:, 1]
    return float(np.sum((lamx * lamy - 1.0) * A0_cm2))


# ===============================
# VelocityNet (Model C / PCA architecture -- matches training/train_model_c.py exactly)
# ===============================
class VelocityNet(nn.Module):
    """NODE right-hand-side f(x) for Model C: a small tanh-MLP mapping the
    concatenated [latent state z, volume-setpoint error e, error integral I,
    volume setpoint SP, PCA-compressed growth feedback g_pca, design
    parameters] to the latent state time-derivative dz/dt."""

    def __init__(self,
                 dim_state: int,
                 dim_sp: int = 1,
                 dim_design: int = 7,
                 dim_error: int = 1,
                 dim_integral: int = 1,
                 dim_g: int = 16,
                 hidden: int = 128):
        super().__init__()
        self.dim_state = dim_state
        self.dim_g = dim_g
        input_dim = dim_state + dim_sp + dim_design + dim_error + dim_integral + dim_g
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim_state),
        )

    def forward(self, x):
        return self.net(x)


def build_pca_node_from_ckpt(ckpt: dict, device: torch.device):
    """Rebuild Model C's VelocityNet from a checkpoint, inferring its
    architecture from the saved state_dict rather than requiring the
    dimensions to be passed in separately (mirrors eval_model_a.py's
    checkpoint-inference pattern, extended here with dim_g for the PCA
    growth-feedback input). "net.4" is the final Linear layer of the
    Sequential (output width = D_state, the latent state dimension), "net.0"
    is the first Linear layer (bias width = hidden layer size, weight's
    second dimension = total NODE input width); dim_g is then recovered
    either from the checkpoint metadata or by subtracting the other known
    input blocks from that total input width."""
    sd = ckpt["model_state_dict"]
    if "net.4.bias" not in sd:
        raise KeyError("Could not find 'net.4.bias' in checkpoint.")
    D_state = int(sd["net.4.bias"].numel())
    if "net.0.bias" not in sd:
        raise KeyError("Could not find 'net.0.bias' in checkpoint.")
    hidden = int(sd["net.0.bias"].numel())
    input_dim = int(sd["net.0.weight"].shape[1])

    dim_sp = int(ckpt.get("dim_sp", 1))
    dim_design = int(ckpt.get("dim_design", 7))
    dim_error = int(ckpt.get("dim_error", 1))
    dim_integral = int(ckpt.get("dim_integral", 1))
    if "dim_g" in ckpt:
        dim_g = int(ckpt["dim_g"])
    else:
        dim_g = input_dim - D_state - dim_sp - dim_design - dim_error - dim_integral

    print("[infer PCA NODE arch] D_state=%d hidden=%d dim_g=%d input_dim=%d" %
          (D_state, hidden, dim_g, input_dim))

    model = VelocityNet(
        dim_state=D_state, dim_sp=dim_sp, dim_design=dim_design,
        dim_error=dim_error, dim_integral=dim_integral, dim_g=dim_g, hidden=hidden,
    ).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, D_state, dim_g


# ===============================
# Main rollout function
# ===============================
def rollout_single_sim_pca_node(
    r_modes: int,
    sim_id: int,
    nodes_csv: str,
    elems_csv: str,
    ckpt_path_template: str = "FINAL_models/Model_C/Model_C_PCA_r{r_modes}_BEST.pt",
    make_plots: bool = False,
):
    """
    Autoregressive rollout for one held-out simulation, Model C (PCA
    growth-feedback NODE). Starting from the true initial state, repeatedly:
    (1) evaluate the NODE velocity network, (2) explicit-Euler step the
    latent state z forward by dt_k, (3) decode the new latent displacement
    into physical node coordinates, (4) advance the per-element growth field
    with the matched implicit integrator using the just-decoded
    displacements, and (5) re-encode that updated growth field into the next
    step's PCA feedback vector:

      z_{k+1} = z_k + dt_k * f(x_k),   x_k = [z_k, e_k, I_k, SP_k, g_pca_k, design]

    where g_pca_k is the normalized PCA projection of the current element
    growth field, re-encoded after each growth-integration step (same
    convention as training/train_model_c.py's rollout). e_k and I_k are a
    simple proportional-integral (PI) tracking signal: e_k is the current
    error between the volume setpoint SP_k and the model's own predicted
    expander volume, and I_k is its running time-integral -- both are part
    of the NODE input so the network can react to how far the tracked
    expander volume has drifted from its prescribed schedule.

    Returns a dict with "t", "z_pred", "z_true", "u_pred_nodes" (T,N,3),
    "g_pred_grid" (T,2,H,W), "lamdag_pred_elem", plus metadata fields.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ckpt_path_template.format(r=r_modes, r_modes=r_modes)
    ckpt = torch.load(ckpt_path, map_location=device)

    model, D_state, dim_g = build_pca_node_from_ckpt(ckpt, device)

    # The reduced state vector Z is [POD displacement coeffs (r_modes of
    # them), expander volume]; the volume is always appended right after the
    # POD coefficients, hence index r_modes.
    volume_idx = int(r_modes)
    if D_state <= volume_idx:
        raise ValueError(f"D_state={D_state} too small for volume_idx=r_modes={r_modes}")

    z_mean = np.asarray(ckpt["z_mean"], dtype=np.float32).reshape(-1)
    z_std  = np.asarray(ckpt["z_std"],  dtype=np.float32).reshape(-1)
    if z_mean.shape[0] != D_state or z_std.shape[0] != D_state:
        raise ValueError(f"Checkpoint z_mean/z_std length mismatch with D_state={D_state}.")

    sp_mean = float(np.asarray(ckpt["sp_mean"], dtype=np.float32).reshape(-1)[0])
    sp_std  = float(np.asarray(ckpt["sp_std"],  dtype=np.float32).reshape(-1)[0])

    design_mean = np.asarray(ckpt["design_mean"], dtype=np.float32).reshape(1, -1)
    design_std  = np.asarray(ckpt["design_std"],  dtype=np.float32).reshape(1, -1)

    e_mean = float(np.asarray(ckpt["e_mean"], dtype=np.float32).reshape(-1)[0])
    e_std  = float(np.asarray(ckpt["e_std"],  dtype=np.float32).reshape(-1)[0])

    I_mean = float(np.asarray(ckpt["I_mean"], dtype=np.float32).reshape(-1)[0])
    I_std  = float(np.asarray(ckpt["I_std"],  dtype=np.float32).reshape(-1)[0])

    growth_pca_npz = ckpt.get("growth_pca_npz", DEFAULT_GROWTH_PCA_NPZ)
    if not os.path.exists(growth_pca_npz):
        growth_pca_npz = DEFAULT_GROWTH_PCA_NPZ
    pca_enc = load_growth_pca_encoder_torch(growth_pca_npz, device=device, dtype=torch.float32)
    if int(pca_enc["dim_g"]) != dim_g:
        raise ValueError(f"PCA encoder dim_g={pca_enc['dim_g']} != model dim_g={dim_g}")

    z_mean_t = torch.from_numpy(z_mean).to(device).view(1, -1)
    z_std_t  = torch.from_numpy(z_std).to(device).view(1, -1)

    # ----------------------------
    # Load raw sim data
    # ----------------------------
    (U_lat, volume_snap, sim_index, time_vals, volume_SP, design_all, n_sims) = load_raw_data(r_modes)

    Z_state_all = np.concatenate([U_lat, volume_snap.reshape(-1, 1)], axis=1).astype(np.float32)

    idx = np.where(sim_index == sim_id)[0]
    if idx.size < 2:
        raise ValueError(f"Sim {sim_id} has too few frames ({idx.size}).")

    t_s = time_vals[idx]
    order = np.argsort(t_s)
    idx_sorted = idx[order]
    t_sim = t_s[order]

    Z_true_sim = Z_state_all[idx_sorted, :]
    SP_sim = volume_SP[idx_sorted]
    design_sim = design_all[idx_sorted, :]

    T_true = Z_true_sim.shape[0]
    n_steps = T_true - 1

    design_raw = design_sim[0:1, :].astype(np.float32)
    design_norm = (design_raw - design_mean) / design_std
    design_tensor = torch.from_numpy(design_norm.astype(np.float32)).to(device)

    # ----------------------------
    # Mesh + decoder
    # ----------------------------
    node_xyz_bot, elem_conn_bot, node_ids_bot, elem_node_ids_bot = load_bottom_surface_mesh_direct(nodes_csv, elems_csv)
    H, W, _ = get_bottom_grid_cache()
    if H * W != elem_conn_bot.shape[0]:
        raise ValueError(f"Expected bottom elems = H*W={H*W}, got {elem_conn_bot.shape[0]}")

    node_xyz_bot_f64 = np.ascontiguousarray(node_xyz_bot, dtype=np.float64)
    elem_conn_bot_i64 = np.ascontiguousarray(elem_conn_bot, dtype=np.int64)
    dN_dxi, dN_deta = shape_function_gradients(0.0, 0.0)
    dN_dxi = np.ascontiguousarray(dN_dxi, dtype=np.float64)
    dN_deta = np.ascontiguousarray(dN_deta, dtype=np.float64)

    decode_u = load_disp_decoder(
        r_modes, zarr_disp_path=ZARR_DISP, pod_group=DISP_POD_GROUP,
        pod_u_name=DISP_POD_U_NAME, pod_mean_name=DISP_POD_MEAN_NAME,
    )

    job_id, theta_crit, k1, k2 = get_growth_params_for_sim(sim_id)

    # ----------------------------
    # Init growth from TRUE (bottom = second half)
    # ----------------------------
    t_sim_g, Gx_sim_elem, Gy_sim_elem, _ = load_elem_growth_for_sim(
        sim_id=sim_id, sim_index=sim_index, time_vals=time_vals,
        zarr_sdv_path=ZARR_SDV, folder_x="snaps_SDV1", folder_y="snaps_SDV4",
    )
    Ne_full = Gx_sim_elem.shape[1]
    # Mesh convention: elements are stored as [top-layer elements,
    # bottom-layer elements], i.e. exactly two layers through the skin
    # thickness, so the bottom surface is always the second half of the
    # element array (same convention used in load_bottom_surface_mesh_direct
    # above). The rollout only tracks/feeds back bottom-surface growth.
    half = Ne_full // 2
    gx0 = Gx_sim_elem[0, half:]
    gy0 = Gy_sim_elem[0, half:]
    if gx0.shape[0] != H * W:
        raise ValueError(f"Expected bottom growth length H*W={H*W}, got {gx0.shape[0]}")
    lamdag_elem = np.ascontiguousarray(np.stack([gx0, gy0], axis=1).astype(np.float64))  # (H*W,2)

    lamdag_pred_elem = [lamdag_elem.copy()]
    g_pred_grid = [np.stack([lamdag_elem[:, 0].reshape(H, W), lamdag_elem[:, 1].reshape(H, W)], axis=0).astype(np.float32)]

    # ----------------------------
    # Init state + PI + g_pca
    # ----------------------------
    Z0_raw = Z_true_sim[0, :].astype(np.float32)
    V0_raw = float(Z0_raw[volume_idx])
    SP0_raw = float(SP_sim[0])

    # PI tracking signal fed to the NODE (see docstring above): e_raw is the
    # setpoint-minus-actual volume error, I_raw its running time-integral.
    e_raw = SP0_raw - V0_raw
    I_raw = 0.0

    z0_norm = ((Z0_raw - z_mean) / z_std).astype(np.float32)[None, :]
    z_curr = torch.from_numpy(z0_norm).to(device)

    lamdag_elem_t0 = torch.as_tensor(lamdag_elem, dtype=torch.float32, device=device)
    g_pca = encode_growth_pca_torch(lamdag_elem_t0, pca_enc)  # (1, dim_g)

    u0 = decode_u(Z0_raw[:r_modes])
    if u0.shape[0] != node_xyz_bot.shape[0]:
        raise ValueError(f"Decode nodes mismatch: u0={u0.shape}, node_xyz_bot={node_xyz_bot.shape}")

    u_pred_nodes = [u0.astype(np.float32)]
    z_pred_hist = [Z0_raw.copy()]

    # ----------------------------
    # Rollout
    # ----------------------------
    with torch.no_grad():
        for k in range(n_steps):
            t_k = float(t_sim[k])
            t_k1 = float(t_sim[k + 1])
            dt_k = float(t_k1 - t_k)
            if (not np.isfinite(dt_k)) or (dt_k <= 0.0):
                raise ValueError(f"Bad dt at step {k}: dt_k={dt_k}")

            SP_k_raw = float(SP_sim[k])
            e_norm = (e_raw - e_mean) / e_std
            I_norm = (I_raw - I_mean) / I_std
            sp_norm = (SP_k_raw - sp_mean) / sp_std

            e_tensor = torch.tensor([[e_norm]], dtype=torch.float32, device=device)
            I_tensor = torch.tensor([[I_norm]], dtype=torch.float32, device=device)
            sp_tensor = torch.tensor([[sp_norm]], dtype=torch.float32, device=device)

            # order: [z | e | I | sp | g_pca | design]  -- verified against
            # training/train_model_c.py's rollout_terminal_alpha_loss_pca_node
            x = torch.cat([z_curr, e_tensor, I_tensor, sp_tensor, g_pca, design_tensor], dim=1)

            dz_norm = model(x)
            z_next = z_curr + dt_k * dz_norm
            z_next_np = z_next.detach().cpu().numpy()[0]

            Z_next_raw = (z_next_np * z_std + z_mean).astype(np.float32)
            V_next_raw = float(Z_next_raw[volume_idx])

            u_nodes = decode_u(Z_next_raw[:r_modes])
            if u_nodes.shape[0] != node_xyz_bot.shape[0]:
                raise ValueError(f"u_nodes/mesh mismatch at step {k}")

            u_nodes_f64 = np.ascontiguousarray(u_nodes, dtype=np.float64)
            F_batch = compute_F_batch_numba(node_xyz_bot_f64, elem_conn_bot_i64, u_nodes_f64, dN_dxi, dN_deta)
            lamdag_elem = integrate_growth_matched_numba(
                F_batch, lamdag_elem, float(k1), float(k2), float(theta_crit), t_k, dt_k
            )
            lamdag_elem = np.ascontiguousarray(lamdag_elem)

            # Re-compress the just-updated growth field into next step's
            # PCA feedback vector g_pca -- this is what makes Model C's
            # dynamics growth-aware, versus Model A's no-feedback baseline.
            lamdag_elem_t = torch.as_tensor(lamdag_elem, dtype=torch.float32, device=device)
            g_pca = encode_growth_pca_torch(lamdag_elem_t, pca_enc)

            u_pred_nodes.append(u_nodes.astype(np.float32))
            lamdag_pred_elem.append(lamdag_elem.copy())
            g_pred_grid.append(np.stack(
                [lamdag_elem[:, 0].reshape(H, W), lamdag_elem[:, 1].reshape(H, W)], axis=0
            ).astype(np.float32))
            z_pred_hist.append(Z_next_raw.copy())

            I_raw = I_raw + dt_k * e_raw
            SP_next_raw = float(SP_sim[k + 1])
            e_raw = SP_next_raw - V_next_raw

            z_curr = z_next

    u_pred_nodes_arr = np.stack(u_pred_nodes, axis=0)
    lamdag_pred_elem_arr = np.stack(lamdag_pred_elem, axis=0)
    g_pred_grid_arr = np.stack(g_pred_grid, axis=0)
    z_pred_arr = np.stack(z_pred_hist, axis=0).astype(np.float32)
    area_gain_pred_final = compute_net_area_gain_from_lamdag_elem(lamdag_pred_elem_arr[-1], A0_cm2=0.25)

    rollout = {
        "t": t_sim.astype(np.float64),
        "z_pred": z_pred_arr,
        "z_true": Z_true_sim.astype(np.float32),
        "u_pred_nodes": u_pred_nodes_arr,
        "g_pred_grid": g_pred_grid_arr,
        "lamdag_pred_elem": lamdag_pred_elem_arr,
        "r_modes": r_modes,
        "sim_id": sim_id,
        "volume_idx": int(volume_idx),
        "H": int(H),
        "W": int(W),
        "area_gain_pred_final_cm2": float(area_gain_pred_final),
    }

    print(f"[rollout PCA NODE] r={r_modes}, sim={sim_id}, steps={n_steps}, "
          f"u_pred_nodes shape={u_pred_nodes_arr.shape}, "
          f"finite={bool(np.isfinite(u_pred_nodes_arr).all())}, "
          f"area_gain_pred_final={area_gain_pred_final:.4f} cm^2")

    return rollout
