"""Model A evaluation / rollout: open-loop (vanilla) Neural ODE surrogate.

Model A is the baseline latent-dynamics model: it receives no growth-state
feedback at all (contrast with Models B/C/D, which feed a scalar, POD, or
CNN-based summary of the growth field back into the velocity network).
This module reproduces Model A's validation results in the paper -- latent
and decoded-displacement rollout error, volume tracking, and final-state
growth/area-gain comparisons against the ground-truth FE simulations -- by:

  1. Loading a trained Model A checkpoint and rebuilding its VelocityNet
     from the saved state_dict (see `build_vanilla_node_from_ckpt`).
  2. Rolling out the latent state z_k open-loop over a validation
     simulation's true time grid, then decoding the displacement latent
     modes back to full nodal displacements through the POD basis (see
     `rollout_single_sim_vanilla_node`, `load_disp_decoder`).
  3. Diagnostically integrating the element-wise growth stretches
     (lambda_g_x, lambda_g_y) at every rollout step, purely for reporting
     net skin-area gain and comparing against the true FE growth field.
  4. Saving per-simulation rollout summaries to zarr for downstream
     cross-model comparison plots/tables.

The growth integrator used here (`integrate_growth_matched` /
`integrate_growth_matched_numba`) is an implicit-Newton time-discretization
that matches the original finite-element UMAT's own growth-law integration,
superseding an earlier explicit-Euler version. It is identical to the one
used in eval_model_d.py (Model D), duplicated here rather than imported so
this file remains self-contained. Because Model A's own NODE dynamics never
consume growth feedback, `lamdag_elem` is diagnostic-only throughout this
file: it is tracked at every step so net area gain can be reported, but it
is never fed into the velocity network's input. Consequently, the choice of
growth integrator only affects the diagnostic growth/area-gain output, not
the reported displacement RMSE (which is identical either way).
"""


#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ===============================
# Core imports (required for training)
# ===============================
import os
import numpy as np
import zarr
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List


# ===============================
# Paths & templates (used by dataset + training)
# ===============================
ZARR_DISP  = "displacements.zarr"
ZARR_SDV   = "ip_growth_elem.zarr"               # lambda_g_x, lambda_g_y latents
ZARR_VOL   = "expd_volumes.zarr"
DESIGN_ZARR_PATH = "Design_and_Metadata.zarr"

U_LATENT_TEMPLATE = "latent_displ_r{r}/coeffs"


# ===============================
# Training hyperparameters
# ===============================
BATCH_SIZE   = 256
N_EPOCHS     = 20
LR           = 1e-3
SEED         = 123

CTRL_DIM   = 1   # volume setpoint
PARAM_DIM  = 7   # design params

# Nominal timestep (used only for rollouts / metadata)
FIXED_DT   = 1.4


# ===============================
# Reproducibility
# ===============================
def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# In[2]:


# --- Displacement POD + snapshot path info (adjust names if needed) ---
DISP_POD_GROUP       = "pod_full"     # inside displacements.zarr
DISP_POD_U_NAME      = "U"            # dataset with basis vectors
DISP_POD_MEAN_NAME   = "mean"         # dataset with mean
DISP_SNAP_DATASET    = "snapshots"    # full-order snapshots in displacements.zarr


# In[3]:


# Global cache for the mesh coordinate/connectivity CSVs, so the (H,W)
# bottom-surface grid mapping is only built once per process (mesh topology
# is identical across all sims/rollouts).
_BOTTOM_GRID_CACHE = None

def get_bottom_grid_cache():
    """Lazily build (once) and return the cached (H, W, bottom_elem_idx)
    bottom-surface grid mapping shared by all rollouts."""
    global _BOTTOM_GRID_CACHE
    if _BOTTOM_GRID_CACHE is None:
        H, W, bottom_idx, *_ = build_bottom_surface_elem_grid_mapping(
            nodes_csv="GOH_Nodes_Test_for_Visualization.csv",
            elems_csv="GOH_Elements_Test_for_Visualization.csv",
            tol_xy=1e-8,
        )
        _BOTTOM_GRID_CACHE = (H, W, bottom_idx)
    return _BOTTOM_GRID_CACHE


# In[4]:


import numpy as np

def build_bottom_surface_elem_grid_mapping(
    nodes_csv: str = "GOH_Nodes_Test_for_Visualization.csv",
    elems_csv: str = "GOH_Elements_Test_for_Visualization.csv",
    tol_xy: float = 1e-8,
):
    """
    Build a structured (H,W) grid mapping for a two-layer quad mesh by selecting
    ONLY the bottom layer (smaller element centroid z) at each in-plane (x,y) cell.

    Assumed CSV formats (no headers):
      Nodes: [node_id, x, y, z]   (z required here)
      Elems: [elem_id, n1, n2, n3, n4]  (quads)

    Returns:
      H, W: inferred grid shape from unique centroid x/y positions
      bottom_elem_idx: (H*W,) element indices (0..Ne-1) selecting bottom element per cell
      cell_xy: (H*W, 2) centroid (x,y) per cell (for debugging)
      elem_id: (Ne,) element ids
      node_id: (Nn,) node ids
      node_xyz: (Nn,3)
      elem_conn_local: (Ne,4) node indices (0..Nn-1)
      xs, ys: grid lines (unique x and y centroid coords)
    """
    # ---- load nodes ----
    nodes = np.loadtxt(nodes_csv, delimiter=",", dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1] < 4:
        raise ValueError(
            f"Nodes CSV must have 4 cols [id,x,y,z] (no headers). Got shape {nodes.shape}"
        )

    node_id = nodes[:, 0].astype(np.int64)
    node_xyz = nodes[:, 1:4].astype(np.float64)  # (Nn,3)

    node_id_to_local = {int(nid): i for i, nid in enumerate(node_id)}

    # ---- load elements ----
    elems = np.loadtxt(elems_csv, delimiter=",", dtype=np.int64)
    if elems.ndim != 2 or elems.shape[1] < 5:
        raise ValueError(
            f"Elements CSV must have >=5 cols [eid,n1,n2,n3,n4] (no headers). Got shape {elems.shape}"
        )

    elem_id = elems[:, 0].astype(np.int64)
    elem_conn_ids = elems[:, 1:5].astype(np.int64)  # (Ne,4)
    Ne = elem_conn_ids.shape[0]

    # Convert node IDs -> local node indices
    elem_conn_local = np.empty((Ne, 4), dtype=np.int64)
    for a in range(4):
        try:
            elem_conn_local[:, a] = np.array(
                [node_id_to_local[int(nid)] for nid in elem_conn_ids[:, a]],
                dtype=np.int64
            )
        except KeyError as e:
            raise KeyError(f"Element connectivity references node id not found: {e}")

    # ---- element centroids ----
    cent = node_xyz[elem_conn_local].mean(axis=1)  # (Ne,3)
    xe, ye, ze = cent[:, 0], cent[:, 1], cent[:, 2]

    # Quantize x,y to remove floating noise
    xe_q = np.round(xe / tol_xy) * tol_xy
    ye_q = np.round(ye / tol_xy) * tol_xy

    # Build grid lines
    xs = np.unique(xe_q); xs.sort()
    ys = np.unique(ye_q); ys.sort()
    W = xs.size
    H = ys.size
    n_cells = H * W

    ix = np.searchsorted(xs, xe_q)
    iy = np.searchsorted(ys, ye_q)
    cell = (iy * W + ix).astype(np.int64)  # (Ne,)

    # ---- select bottom element per cell (min z) ----
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
        raise ValueError(
            f"Some grid cells have no assigned element (unexpected for structured mesh). "
            f"Example missing cell indices: {missing}"
        )

    # Debug: check it really reduced 2 layers -> 1 per cell
    # Count elements per cell (should be 2 for most cells in a 2-layer mesh)
    counts = np.bincount(cell, minlength=n_cells)
    print(f"[bottom mapping] Ne={Ne}, grid H={H}, W={W}, H*W={n_cells}")
    print(f"[bottom mapping] per-cell counts: min={counts[counts>0].min()}, max={counts.max()}, "
          f"unique occupied cells={np.count_nonzero(counts)}")
    print(f"[bottom mapping] selected bottom elems: {bottom_elem_idx.shape[0]} (should equal H*W)")

    # For debugging: centroid (x,y) per cell (from selected bottom element)
    cell_xy = np.stack([xe[bottom_elem_idx], ye[bottom_elem_idx]], axis=1)

    return H, W, bottom_elem_idx, cell_xy, elem_id, node_id, node_xyz, elem_conn_local, xs, ys


def rasterize_bottom_surface_field(
    vec_elem: np.ndarray,          # (Ne,)
    H: int, W: int,
    bottom_elem_idx: np.ndarray,   # (H*W,)
):
    """
    Convert an element field vector (Ne,) into a bottom-surface grid image (H,W)
    by selecting vec_elem at bottom_elem_idx for each cell.
    """
    flat = vec_elem[bottom_elem_idx]          # (H*W,)
    return flat.reshape(H, W)


# In[5]:


import numpy as np
import zarr

import functools


@functools.lru_cache(maxsize=None)
def get_growth_params_for_sim(sim_id: int):
    """
    Map sim_id -> job_id using displacements.zarr/mapping/job_index_per_sim,
    then read the correct row from the design file:
        'PGOH_Ideal_tol_Tcrit_V_mu_kk1_kk2_kappa_k1_design.txt'

    The design file:
        - Has a header row to skip
        - Each subsequent row corresponds to job_id = line_number (1-based)
        - Columns are comma- or space-delimited
        - We extract columns 2, 5, 6 (0-based) as:
             theta_crit = col 2
             k1         = col 5
             k2         = col 6
    """

    # --- 1) sim_id -> job_id ---
    g_disp = zarr.open("Design_and_Metadata.zarr", mode="r")
    job_index_per_sim = g_disp["Mapping_indexes_and_metadata"]["job_index_per_sim"][:]  # shape (n_sims,)
    job_id = int(job_index_per_sim[sim_id])   # job index for this sim (1-based!)

    # --- 2) Load design table from text file ---
    design_file = "PGOH_Ideal_tol_Tcrit_V_mu_kk1_kk2_kappa_k1_design.txt"

    # Load skipping header, allowing space or comma separators
    design_table = np.loadtxt(design_file, skiprows=0)

    # Because job_id is 1-based, convert to 0-based row index:
    row_index = job_id - 1

    if row_index < 0 or row_index >= design_table.shape[0]:
        raise IndexError(
            f"Job ID {job_id} (row idx {row_index}) is out of range for design table "
            f"with {design_table.shape[0]} rows."
        )

    row = design_table[row_index]

    # --- 3) Extract θ_crit, k1, k2 ---
    theta_crit = float(row[2])   # column 2 (0-based index)
    k1         = float(row[5])   # column 5
    k2         = float(row[6])   # column 6

    return job_id, theta_crit, k1, k2


# In[6]:


import functools


@functools.lru_cache(maxsize=None)
def load_raw_data(r_modes: int):
    """
    Load and align the raw per-snapshot arrays needed for rollouts, for a
    given number of displacement POD modes r_modes. All arrays share the
    same snapshot ordering/length S (one row per saved FE timestep, across
    all simulations).

    Returns (in this order):
        U_lat        : (S, r)  displacement POD latent coefficients
        volume_snap  : (S,)    actual expander cavity volume per snapshot
        sim_index    : (S,)    simulation id for each snapshot row
        time_vals    : (S,)    simulation time [days] for each snapshot row
        volume_SP    : (S,)    volume setpoint (target) per snapshot
        design_all   : (S, 7)  design parameters per snapshot
        n_sims       : int     number of distinct simulations
    """
    from pathlib import Path

    g_disp   = zarr.open_group(ZARR_DISP, mode="r")
    g_design = zarr.open_group(DESIGN_ZARR_PATH, mode="r")
    g_sdv    = zarr.open_group(ZARR_SDV, mode="r")   # contains snaps_SDV1 and snaps_SDV4

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

    design_all = np.asarray(
        g_design["design_params_per_snapshot"], dtype=np.float32
    )

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


# ## Load the model

# In[7]:


import numpy as np

@functools.lru_cache(maxsize=None)
def load_bottom_surface_mesh_direct(nodes_csv: str, elems_csv: str):
    """
    Robust bottom-surface mesh loader:

    - elems_csv contains both layers; bottom layer is second half of rows
    - bottom nodes are defined as the unique node IDs referenced by bottom elements
      (this is the only assumption that cannot be broken by file ordering)

    Returns:
      node_xyz_bot : (Nbot, 3)
      elem_conn_bot: (Ne_bot, 4) 0-based indices into node_xyz_bot
      node_ids_bot : (Nbot,)     node IDs corresponding to node_xyz_bot rows
      elem_node_ids_bot : (Ne_bot,4) original node IDs for debugging
    """
    nodes = np.loadtxt(nodes_csv, delimiter=",")
    elems = np.loadtxt(elems_csv, delimiter=",")

    node_ids = nodes[:, 0].astype(np.int64)
    node_xyz = nodes[:, 1:4].astype(np.float64)

    elem_ids = elems[:, 0].astype(np.int64)
    elem_node_ids = elems[:, 1:5].astype(np.int64)
    Ne = elem_node_ids.shape[0]
    if Ne % 2 != 0:
        raise ValueError(f"Expected even number of elements (2 layers), got Ne={Ne}")

    # bottom layer = second half of element rows
    half = Ne // 2
    elem_node_ids_bot = elem_node_ids[half:, :]   # (Ne_bot, 4)

    # bottom node IDs = unique node IDs referenced by bottom elements
    bottom_node_ids = np.unique(elem_node_ids_bot.reshape(-1))

    # pull those nodes from nodes_csv
    # (use a dict for fast lookup by ID)
    id2xyz = {int(nid): node_xyz[i] for i, nid in enumerate(node_ids)}

    missing = [int(nid) for nid in bottom_node_ids if int(nid) not in id2xyz]
    if len(missing) > 0:
        raise ValueError(
            f"{len(missing)} node IDs referenced by bottom elements are missing from nodes_csv. "
            f"Example missing IDs: {missing[:10]}"
        )

    # build bottom node table, sorted by node_id
    node_ids_bot = np.sort(bottom_node_ids.astype(np.int64))
    node_xyz_bot = np.vstack([id2xyz[int(nid)] for nid in node_ids_bot]).astype(np.float64)

    # map node_id -> local index [0..Nbot-1]
    id2bot = {int(nid): i for i, nid in enumerate(node_ids_bot)}

    # remap element connectivity to local indices
    get_idx = np.vectorize(lambda nid: id2bot.get(int(nid), -1), otypes=[np.int64])
    elem_conn_bot = get_idx(elem_node_ids_bot)

    bad = np.where(elem_conn_bot < 0)
    if bad[0].size > 0:
        ei = int(bad[0][0])
        raise ValueError(
            "Internal mapping failure (shouldn't happen if missing-check passed).\n"
            f"Example bad elem row: {ei}\n"
            f"elem_node_ids_bot[ei] = {elem_node_ids_bot[ei]}\n"
            f"elem_conn_bot[ei]     = {elem_conn_bot[ei]}"
        )

    print("[bottom surface mesh — robust]")
    print("  node_xyz_bot:", node_xyz_bot.shape)
    print("  elem_conn_bot:", elem_conn_bot.shape)
    print("  node_id range:", int(node_ids_bot.min()), int(node_ids_bot.max()))
    print("  conn min/max:", int(elem_conn_bot.min()), int(elem_conn_bot.max()))

    return node_xyz_bot, elem_conn_bot, node_ids_bot, elem_node_ids_bot


# In[8]:


import numpy as np

def normalize_growth_grid(G_grid: np.ndarray, g_mean, g_std, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize growth in the SAME way as training.

    G_grid: (B, 2, H, W) or (2, H, W)
    g_mean/g_std: either scalars or array-like of shape (2,)

    Returns normalized array with same shape as G_grid.
    """
    G = G_grid.astype(np.float32, copy=False)

    g_mean = np.asarray(g_mean, dtype=np.float32)
    g_std  = np.asarray(g_std, dtype=np.float32)

    if G.ndim == 3:  # (2,H,W)
        # reshape stats to (2,1,1)
        if g_mean.shape == (2,):
            mu = g_mean[:, None, None]
            sig = g_std[:, None, None]
        else:
            mu = float(g_mean)
            sig = float(g_std)
        return (G - mu) / (sig + eps)

    if G.ndim == 4:  # (B,2,H,W)
        if g_mean.shape == (2,):
            mu = g_mean[None, :, None, None]
            sig = g_std[None, :, None, None]
        else:
            mu = float(g_mean)
            sig = float(g_std)
        return (G - mu) / (sig + eps)

    raise ValueError(f"Expected G_grid with shape (2,H,W) or (B,2,H,W), got {G.shape}")


# In[9]:


import zarr

_GROWTH_SNAP_CACHE = {}


def load_elem_growth_for_sim(
    sim_id: int,
    sim_index: np.ndarray,
    time_vals: np.ndarray,
    zarr_sdv_path: str = "ip_growth_elem.zarr",
    folder_x: str = "snaps_SDV1",
    folder_y: str = "snaps_SDV4",
):
    """
    Returns element growth for this sim for ALL frames:
      t_sim: (T,)
      Gx_sim: (T, Ne)
      Gy_sim: (T, Ne)
      idx_sorted: (T,) global snapshot indices for this sim
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

    # select frames for this sim
    idx = np.where(sim_index == sim_id)[0]
    if idx.size < 2:
        raise ValueError(f"Sim {sim_id} has too few frames ({idx.size}).")

    t_s = time_vals[idx]
    order = np.argsort(t_s)
    idx_sorted = idx[order]
    t_sim = t_s[order]

    # Make snapshot-major: (S, Ne)
    S = sim_index.shape[0]
    if Gx_raw.shape[1] == S:   # (Ne, S)
        Gx_all = Gx_raw.T
    else:                      # assume (S, Ne)
        Gx_all = Gx_raw

    if Gy_raw.shape[1] == S:
        Gy_all = Gy_raw.T
    else:
        Gy_all = Gy_raw

    # slice sim
    Gx_sim = Gx_all[idx_sorted, :]  # (T, Ne)
    Gy_sim = Gy_all[idx_sorted, :]  # (T, Ne)

    return t_sim, Gx_sim, Gy_sim, idx_sorted


# In[10]:


import numpy as np
import zarr

@functools.lru_cache(maxsize=None)
def load_disp_decoder(
    r_modes: int,
    zarr_disp_path: str = "displacements.zarr",
    pod_group: str = "pod_full",
    pod_u_name: str = "U",
    pod_mean_name: str = "mean",
    verbose: bool = True,
):
    """
    Returns a callable decode(alpha)->u_nodes (N,3).
    Assumes flatten order: [ux_block, uy_block, uz_block].
    """
    g_disp = zarr.open_group(zarr_disp_path, mode="r")

    U_key  = f"{pod_group}/{pod_u_name}"
    mu_key = f"{pod_group}/{pod_mean_name}"

    # Load POD basis + mean
    U_full = np.asarray(g_disp[U_key])               # (M, Rmax) typically
    mu     = np.asarray(g_disp[mu_key]).reshape(-1)  # (M,)

    if U_full.ndim != 2:
        raise ValueError(f"Expected POD basis U to be 2D (M,Rmax). Got {U_full.shape} at '{U_key}'")

    M, Rmax = U_full.shape
    if mu.shape[0] != M:
        raise ValueError(f"Mean length must match basis rows M={M}. Got mu.shape={mu.shape} at '{mu_key}'")

    if r_modes > Rmax:
        raise ValueError(f"Requested r_modes={r_modes} exceeds available Rmax={Rmax} in '{U_key}'")

    U_r = U_full[:, :r_modes]

    # Infer node count from block layout [ux, uy, uz]
    if M % 3 != 0:
        raise ValueError(f"Expected M divisible by 3 for [ux,uy,uz] blocks. Got M={M}")
    N = M // 3

    if verbose:
        print(f"[load_disp_decoder] Using U='{U_key}' shape={U_full.shape}")
        print(f"[load_disp_decoder] Using mean='{mu_key}' shape={mu.shape}")
        print(f"[load_disp_decoder] r_modes={r_modes}, N_nodes={N}")

    def decode(alpha: np.ndarray) -> np.ndarray:
        alpha = np.asarray(alpha).reshape(-1)
        if alpha.shape[0] != r_modes:
            raise ValueError(f"alpha must have shape ({r_modes},), got {alpha.shape}")

        u_flat = mu + U_r @ alpha  # (M,)
        ux = u_flat[0:N]
        uy = u_flat[N:2*N]
        uz = u_flat[2*N:3*N]
        return np.stack([ux, uy, uz], axis=1)  # (N,3)

    return decode


# ## Define Growth Integration Function

# In[11]:


import numpy as np

def shape_function_gradients(xi, eta):
    """Return arrays dN_dxi, dN_deta of length 4 for bilinear quad."""
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

# Example: compute Gξ and Gη for one element
# Suppose nodal coordinates X are shape (4,3), each row is a node [x,y,z]
X_elem = np.array([
    [0.0, 0.0, 0.0],   # node 1
    [1.0, 0.0, 0.0],   # node 2
    [1.0, 1.0, 0.0],   # node 3
    [0.0, 1.0, 0.0]    # node 4
])

xi, eta = 0.0, 0.0  # evaluate at element center
dN_dxi, dN_deta = shape_function_gradients(xi, eta)

# Gξ and Gη are 3D vectors
Gxi = X_elem.T @ dN_dxi   # (3,) vector
Geta = X_elem.T @ dN_deta # (3,) vector

print("Gxi =", Gxi)
print("Geta =", Geta)


# In[12]:


def evalN1(xi,eta):
  N1 = (1 - xi)*(1 - eta) / 4
  return N1
def evalN2(xi,eta):
  N2 = (1 + xi)*(1 - eta) / 4
  return N2
def evalN3(xi,eta):
  N3 = (1 + xi)*(1 + eta) / 4
  return N3
def evalN4(xi,eta):
  N4 = (1 - xi)*(1 + eta) / 4
  return N4
def dN1_dxi(xi, eta):
    return -(1 - eta) / 4
def dN1_deta(xi, eta):
    return -(1 - xi) / 4

def dN2_dxi(xi, eta):
    return (1 - eta) / 4
def dN2_deta(xi, eta):
    return -(1 + xi) / 4

def dN3_dxi(xi, eta):
    return (1 + eta) / 4
def dN3_deta(xi, eta):
    return (1 + xi) / 4

def dN4_dxi(xi, eta):
    return -(1 + eta) / 4
def dN4_deta(xi, eta):
    return (1 - xi) / 4


# In[13]:


import numpy as np
from numba import njit, prange


# ===============================
# UMAT-matched implicit-Newton growth integrator (replaces an earlier,
# explicit-Euler growth integrator that used to live here). This is the
# same integrator used in eval_model_d.py (Model D); it is duplicated here
# rather than imported so that this file stays self-contained and can be
# read/run on its own.
# ===============================
def _growth_gate(time):
    """Growth on/off switch as a function of simulation time [days].

    Returns 0.0 (growth frozen) during the listed day-windows, which are
    the rest periods between tissue-expander fills in the simulated
    protocol, and 1.0 (growth active) otherwise. Each ~14-day cycle is
    7 days of rest followed by 7 days of active growth after a fill.
    """
    if (0.0 < time < 7.0) or (14.0 < time < 21.0) or (28.0 < time < 35.0) or \
       (42.0 < time < 49.0) or (56.0 < time < 63.0) or (70.0 < time < 77.0) or \
       (84.0 < time < 91.0) or (98.0 < time < 105.0):
        return 0.0
    return 1.0
_growth_gate_jit = njit(_growth_gate)


# Implicit-Newton update of the two principal growth stretches per element,
# lambda_g_x and lambda_g_y (columns of elem_lamdag), each following the
# growth law d(lambda_g)/dt = k*(lambda/lambda_g - theta_crit) while the
# growth criterion phi = lambda/lambda_g - theta_crit is positive, and
# frozen (d(lambda_g)/dt = 0) otherwise. `lambda1`/`lambda2` are the current
# elastic stretches along the element's two in-plane directions, recovered
# from the deformation gradient F_batch (see compute_F_batch_numpy). Growth
# is additionally frozen everywhere during the rest windows selected by
# _growth_gate. Each element's update is solved independently with a short
# Newton-Raphson iteration (tol/maxiter) on the backward-Euler residual;
# `elem_lamdag`/`out` hold [lambda_g_x, lambda_g_y] per element (shape
# (Ne, 2)), and the function returns the updated array for this timestep.
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


def compute_F_batch_numpy(node_coords, elem_conn, node_u, xi: float = 0.0, eta: float = 0.0, eps: float = 1e-12):
    """
    Deformation gradient F, shape (Ne,3,3), for a batch of bilinear quad
    shell/membrane elements, evaluated at one in-plane point (xi, eta)
    (the element center by default).

    Builds the reference covariant basis (Gxi, Geta, N) from the
    undeformed nodal coordinates and the deformed covariant basis
    (gxi, geta, n) from coordinates + current nodal displacements
    (node_u), using the same bilinear shape-function derivatives as the FE
    shell formulation. F is assembled from the deformed basis vectors
    dyadic with the dual (contravariant) reference basis: F = gxi ⊗ Gxi*
    + geta ⊗ Geta* + n ⊗ N, i.e. the through-thickness/normal direction is
    carried through unstretched, consistent with a thin-shell assumption.

    Parameters
    ----------
    node_coords : (Nn,3) reference (undeformed) nodal coordinates
    elem_conn   : (Ne,4) 0-based node indices per element
    node_u      : (Nn,3) nodal displacements
    xi, eta     : parametric coordinates at which to evaluate F
    eps         : small number guarding the surface-normal normalization

    Returns
    -------
    F : (Ne,3,3) deformation gradient per element
    """
    node_coords = np.asarray(node_coords, dtype=np.float64)
    node_u = np.asarray(node_u, dtype=np.float64)
    elem_conn = np.asarray(elem_conn, dtype=np.int64)

    X_e = node_coords[elem_conn]
    u_e = node_u[elem_conn]
    x_e = X_e + u_e

    dN_dxi, dN_deta = shape_function_gradients(xi, eta)
    dN_dxi = np.asarray(dN_dxi, dtype=np.float64).reshape(4)
    dN_deta = np.asarray(dN_deta, dtype=np.float64).reshape(4)

    Gxi  = np.einsum('ead,a->ed', X_e, dN_dxi)
    Geta = np.einsum('ead,a->ed', X_e, dN_deta)
    cross_ref = np.cross(Gxi, Geta)
    N = cross_ref / (np.linalg.norm(cross_ref, axis=1, keepdims=True) + eps)

    G = np.stack([Gxi, Geta, N], axis=2)
    G_inv = np.linalg.inv(G)
    G_dual_xi  = G_inv[:, 0, :]
    G_dual_eta = G_inv[:, 1, :]

    gxi  = np.einsum('ead,a->ed', x_e, dN_dxi)
    geta = np.einsum('ead,a->ed', x_e, dN_deta)
    cross_def = np.cross(gxi, geta)
    n = cross_def / (np.linalg.norm(cross_def, axis=1, keepdims=True) + eps)

    F = (gxi[:, :, None] * G_dual_xi[:, None, :]
         + geta[:, :, None] * G_dual_eta[:, None, :]
         + n[:, :, None] * N[:, None, :])
    return np.ascontiguousarray(F, dtype=np.float64)


def integrate_growth_matched(node_coords, elem_conn, node_u, elem_lamdag, k1, k2, theta_crit, time, dt):
    """UMAT-matched (implicit-Newton) growth integrator for one timestep:
    computes the deformation gradient at each element (compute_F_batch_numpy)
    from the current nodal displacements, then advances the growth stretches
    elem_lamdag -> elem_lamdag_next via integrate_growth_matched_numba. Pure
    numpy/Newton call convention (raw numpy arrays in, raw numpy array out),
    so it can be called directly from a Python rollout loop each step."""
    F_batch = compute_F_batch_numpy(node_coords, elem_conn, node_u)
    return integrate_growth_matched_numba(
        F_batch,
        np.asarray(elem_lamdag, dtype=np.float64),
        float(k1), float(k2), float(theta_crit), float(time), float(dt),
    )


def get_true_u_nodes_for_sim_step(
    sim_id: int,
    step_k: int,                       # index within this sim's timeline (0..T_true-1)
    sim_index: np.ndarray,
    time_vals: np.ndarray,
    zarr_disp_path: str = "displacements.zarr",
    snapshots_key: str = "snapshots",
) -> np.ndarray:
    """
    Returns u_true_nodes at sim local step_k as (N,3).

    Assumes displacements.zarr stores full displacement snapshots at:
        displacements.zarr/snapshots

    Supports snapshots array stored as (S,M) or (M,S),
    with flattening order [ux_block, uy_block, uz_block].
    """
    g = zarr.open_group(zarr_disp_path, mode="r")

    if snapshots_key not in g:
        raise KeyError(f"Could not find '{snapshots_key}' in {zarr_disp_path}. Available: {list(g.array_keys())}")

    snaps = g[snapshots_key]  # zarr array

    # global snapshot indices for this sim, sorted by time
    idx = np.where(sim_index == sim_id)[0]
    if idx.size == 0:
        raise ValueError(f"No snapshots for sim_id={sim_id}")
    t_s = time_vals[idx]
    order = np.argsort(t_s)
    idx_sorted = idx[order]

    if step_k < 0 or step_k >= idx_sorted.size:
        raise IndexError(f"step_k={step_k} out of range for sim length {idx_sorted.size}")

    gidx = int(idx_sorted[step_k])  # global snapshot index

    if snaps.ndim != 2:
        raise ValueError(f"Expected snapshots to be 2D, got shape {snaps.shape}")

    S = sim_index.shape[0]

    # pull snapshot vector u_flat of length M
    if snaps.shape[0] == S:          # (S,M)
        u_flat = np.asarray(snaps[gidx, :], dtype=np.float64)
    elif snaps.shape[1] == S:        # (M,S)
        u_flat = np.asarray(snaps[:, gidx], dtype=np.float64)
    else:
        raise ValueError(f"Snapshots shape {snaps.shape} doesn't match S={S} in either axis")

    M = u_flat.shape[0]
    if M % 3 != 0:
        raise ValueError(f"Expected M divisible by 3, got M={M}")
    N = M // 3

    ux = u_flat[0:N]
    uy = u_flat[N:2*N]
    uz = u_flat[2*N:3*N]
    return np.stack([ux, uy, uz], axis=1)  # (N,3)


# In[17]:


import numpy as np

def get_true_bottom_growth_grid_for_sim_step(
    sim_id: int,
    step_k: int,                      # 0-based within this sim after sorting by time
    sim_index: np.ndarray,
    time_vals: np.ndarray,
    H: int,
    W: int,
    zarr_sdv_path: str = "ip_growth_elem.zarr",
    folder_x: str = "snaps_SDV1",
    folder_y: str = "snaps_SDV4",
):
    """
    Returns:
      G_true_grid: (2,H,W) raw (not normalized), channels: [Gx, Gy]
    """
    t_sim, Gx_sim_elem, Gy_sim_elem, _ = load_elem_growth_for_sim(
        sim_id=sim_id,
        sim_index=sim_index,
        time_vals=time_vals,
        zarr_sdv_path=zarr_sdv_path,
        folder_x=folder_x,
        folder_y=folder_y,
    )

    if step_k < 0 or step_k >= Gx_sim_elem.shape[0]:
        raise IndexError(f"step_k={step_k} out of range for sim length T={Gx_sim_elem.shape[0]}")

    Ne_full = Gx_sim_elem.shape[1]
    if Ne_full % 2 != 0:
        raise ValueError(f"Expected even Ne from growth arrays, got Ne={Ne_full}")
    half = Ne_full // 2

    gx = Gx_sim_elem[step_k, half:]   # bottom half
    gy = Gy_sim_elem[step_k, half:]

    if gx.shape[0] != H * W:
        raise ValueError(f"Expected bottom elems H*W={H*W}, got {gx.shape[0]}")

    G_true = np.zeros((2, H, W), dtype=np.float32)
    G_true[0] = gx.reshape(H, W)
    G_true[1] = gy.reshape(H, W)
    return G_true


# In[18]:


import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend -- must be set before any pyplot import
import matplotlib.pyplot as plt

def plot_disp_pred_vs_true_final(
    rollout: dict,
    u_true_nodes_final: np.ndarray,    # (N,3)
    title_prefix: str = "",
):
    """
    Plots (row 1):
      - u_pred, u_true, u_pred - u_true  (component-wise signed fields)

    Uses red/white diverging colormap for signed comparison.
    """
    u_pred = rollout["u_pred_nodes"][-1]  # (N,3)
    u_true = u_true_nodes_final           # (N,3)

    if u_pred.shape != u_true.shape:
        raise ValueError(f"Shape mismatch: u_pred {u_pred.shape} vs u_true {u_true.shape}")

    H = int(rollout["H"])
    W = int(rollout["W"])
    N_expected = (H + 1) * (W + 1)
    if u_pred.shape[0] != N_expected:
        raise ValueError(
            f"Expected N=(H+1)*(W+1)={N_expected}, got {u_pred.shape[0]}. "
            "If your node ordering isn't a perfect grid, we need a node->grid map."
        )

    # ---- NO MAGNITUDE ----
    # show Z-displacement (change index if you want x or y)
    comp = 2

    field_pred = u_pred[:, comp].reshape(H + 1, W + 1)
    field_true = u_true[:, comp].reshape(H + 1, W + 1)
    field_err  = (u_pred[:, comp] - u_true[:, comp]).reshape(H + 1, W + 1)

    # symmetric color limits for signed error
    err_max = np.max(np.abs(field_err))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    ax = axes[0]
    im = ax.imshow(field_pred, origin="lower")
    ax.set_title(f"{title_prefix}\nu_pred (final)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    im = ax.imshow(field_true, origin="lower")
    ax.set_title("u_true (final)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    im = ax.imshow(
        field_err,
        origin="lower",
        cmap="coolwarm",
        vmin=-err_max,
        vmax=err_max,
    )
    ax.set_title("u_pred - u_true (signed error)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("j")
        ax.set_ylabel("i")

    plt.show()


# In[19]:


import matplotlib.pyplot as plt
import numpy as np

def plot_growth_pred_vs_true_final(roll, G_true_final, title_prefix=""):
    """
    roll["lamdag_pred_elem"] is raw element growth (T,H*W,2)
    G_true_final is (2,H,W) raw (not normalized)
    """
    H, W = roll["H"], roll["W"]

    # predicted final (raw) -> grid
    lam_final = roll["lamdag_pred_elem"][-1]  # (H*W,2)
    G_pred = np.zeros((2, H, W), dtype=np.float32)
    G_pred[0] = lam_final[:, 0].reshape(H, W)
    G_pred[1] = lam_final[:, 1].reshape(H, W)

    # errors
    E = G_pred - G_true_final

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)

    names = [("Gx", 0), ("Gy", 1)]
    for r, (nm, c) in enumerate(names):
        im0 = axes[r, 0].imshow(G_pred[c], origin="lower")
        axes[r, 0].set_title(f"Pred {nm}")
        plt.colorbar(im0, ax=axes[r, 0], fraction=0.046, pad=0.04)

        im1 = axes[r, 1].imshow(G_true_final[c], origin="lower")
        axes[r, 1].set_title(f"True {nm}")
        plt.colorbar(im1, ax=axes[r, 1], fraction=0.046, pad=0.04)

        # ---- SIGNED ERROR PLOT ----
        err_field = E[c]
        err_max = np.max(np.abs(err_field))

        im2 = axes[r, 2].imshow(
            err_field,
            origin="lower",
            cmap="coolwarm",
            vmin=-err_max,
            vmax=err_max,
        )
        axes[r, 2].set_title(f"Error {nm} (pred-true)")
        plt.colorbar(im2, ax=axes[r, 2], fraction=0.046, pad=0.04)

    main_title = title_prefix.strip()
    if not main_title:
        main_title = f"Growth fields @ final step | sim={roll.get('sim_id')} | r={roll.get('r_modes')}"
    fig.suptitle(main_title)
    plt.tight_layout()
    plt.show()


def rollout_single_sim_vanilla_node(
    r_modes: int,
    sim_id: int,
    nodes_csv: str,
    elems_csv: str,
    max_steps: int | None = None,
    ckpt_path_template: str = "FINAL_models/Model_A/Model_A_Vanilla_r{r_modes}_BEST.pt",
    make_plots: bool = True,
):
    """
    Rollout for vanilla NODE (no CNN, no growth-feedback input):

      z_{k+1} = z_k + dt_k * f(x_k)

    where
      x_k = [z_k, e_k, I_k, SP_k, design]

    Growth is still integrated AFTER each predicted displacement step so that we can:
      - compute predicted growth fields
      - compute final net area gain
      - compare predicted growth to true growth
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
        raise ValueError(
            f"Checkpoint z_mean/z_std length mismatch with D_state={D_state}. "
            f"Got z_mean={z_mean.shape}, z_std={z_std.shape}"
        )

    sp_mean = float(np.asarray(ckpt["sp_mean"], dtype=np.float32).reshape(-1)[0])
    sp_std  = float(np.asarray(ckpt["sp_std"], dtype=np.float32).reshape(-1)[0])

    design_mean = np.asarray(ckpt["design_mean"], dtype=np.float32).reshape(1, -1)
    design_std  = np.asarray(ckpt["design_std"], dtype=np.float32).reshape(1, -1)

    e_mean = float(np.asarray(ckpt["e_mean"], dtype=np.float32).reshape(-1)[0])
    e_std  = float(np.asarray(ckpt["e_std"], dtype=np.float32).reshape(-1)[0])

    I_mean = float(np.asarray(ckpt["I_mean"], dtype=np.float32).reshape(-1)[0])
    I_std  = float(np.asarray(ckpt["I_std"], dtype=np.float32).reshape(-1)[0])

    model = build_vanilla_node_from_ckpt(ckpt, device)
    model.eval()


    # ----------------------------
    # 2) Load raw sim data
    # ----------------------------
    (U_lat,
     volume_snap,
     sim_index, time_vals,
     volume_SP, design_all,
     n_sims) = load_raw_data(r_modes)

    # Latent state z = [displacement POD coefficients (r_modes of them), cavity volume]
    # -- volume is appended as the last channel, at index `volume_idx` == r_modes.
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

    # ----------------------------
    # 3) Mesh + decoder
    # ----------------------------
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
    )

    # theta_crit (critical stretch ratio) and k1/k2 (growth-rate constants for the
    # two in-plane directions) are the growth-law material parameters for this
    # sim's design point -- used only by the diagnostic growth integrator below.
    job_id, theta_crit, k1, k2 = get_growth_params_for_sim(sim_id)
    print(f"[sim {sim_id}] job_id={job_id}  theta_crit={theta_crit:.6g}  k1={k1:.6g}  k2={k2:.6g}")

    # ----------------------------
    # 4) init growth from TRUE (bottom = second half)
    # ----------------------------
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

    # lamdag_elem ("lambda_g" per element): the two principal growth stretches
    # [lambda_g_x, lambda_g_y] per bottom-surface element, shape (Ne,2), raw
    # (unnormalized) units. Initialized here from the true FE growth state at
    # t=0 and updated every step below by the diagnostic growth integrator.
    lamdag_elem = np.stack([gx0, gy0], axis=1).astype(np.float64)  # RAW

    g_pred_grid = []
    G0_raw = np.zeros((2, H, W), dtype=np.float32)
    G0_raw[0] = lamdag_elem[:, 0].reshape(H, W)
    G0_raw[1] = lamdag_elem[:, 1].reshape(H, W)
    g_pred_grid.append(G0_raw.copy())

    # ----------------------------
    # 5) init state + PI
    # ----------------------------
    Z0_raw  = Z_true_sim[0, :].astype(np.float32)
    V0_raw  = float(Z0_raw[volume_idx])
    SP0_raw = float(SP_sim[0])

    # e_raw: volume-tracking error (setpoint - current volume); I_raw: its
    # running time-integral. These are PI-control-like feedback terms that
    # get normalized and concatenated into the NODE's input x_k each step
    # below, alongside the latent state, volume setpoint, and design params.
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
    lamdag_pred_elem = []

    u0 = decode_u(Z0_raw[:r_modes])
    if u0.shape[0] != node_xyz_bot.shape[0]:
        raise ValueError(f"Decode nodes mismatch: u0={u0.shape}, node_xyz_bot={node_xyz_bot.shape}")
    u_pred_nodes.append(u0.astype(np.float32))
    lamdag_pred_elem.append(lamdag_elem.copy())

    # ----------------------------
    # 6) rollout
    # ----------------------------
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

            x = torch.cat([z_curr, e_tensor, I_tensor, sp_tensor, design_tensor], dim=1)

            dz_norm = model(x)

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
                raise ValueError(
                    f"u_nodes and node_xyz_bot mismatch at step {k}: "
                    f"u_nodes={u_nodes.shape}, node_xyz_bot={node_xyz_bot.shape}"
                )
            if elem_conn_bot.max() >= u_nodes.shape[0]:
                raise ValueError(
                    f"elem_conn_bot out of range at step {k}: max={elem_conn_bot.max()} "
                    f"but u_nodes has {u_nodes.shape[0]} nodes."
                )

            lamdag_elem = integrate_growth_matched(
                node_coords=node_xyz_bot,
                elem_conn=elem_conn_bot,
                node_u=u_nodes,
                elem_lamdag=lamdag_elem,
                k1=k1, k2=k2, theta_crit=theta_crit,
                time=t_k,
                dt=dt_k,
            )

            G_raw = np.zeros((2, H, W), dtype=np.float32)
            G_raw[0] = lamdag_elem[:, 0].reshape(H, W)
            G_raw[1] = lamdag_elem[:, 1].reshape(H, W)

            u_pred_nodes.append(u_nodes.astype(np.float32))
            g_pred_grid.append(G_raw.copy())
            lamdag_pred_elem.append(lamdag_elem.copy())

            I_raw = I_raw + dt_k * e_raw
            SP_next_raw = float(SP_sim[k+1])
            e_raw = SP_next_raw - V_next_raw
            e_hist.append(e_raw)
            I_hist.append(I_raw)

            z_curr = z_next

    # ----------------------------
    # 7) pack outputs
    # ----------------------------
    z_true_raw = np.vstack(z_true_list)
    z_pred_raw = np.vstack(z_pred_list)
    t_arr      = np.array(t_list, dtype=np.float64)

    vol_true = np.array(vol_true_list, dtype=np.float64)
    vol_pred = np.array(vol_pred_list, dtype=np.float64)

    err_L2  = np.linalg.norm(z_pred_raw - z_true_raw, axis=1)
    err_vol = np.abs(vol_pred - vol_true)

    area_gain_pred_final = compute_net_area_gain_from_lamdag_elem(
        lamdag_pred_elem[-1], A0_cm2=0.25
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
        "g_pred_grid": np.stack(g_pred_grid, axis=0),          # raw grids
        "lamdag_pred_elem": np.stack(lamdag_pred_elem, axis=0),

        "r_modes": r_modes,
        "sim_id": sim_id,
        "volume_idx": int(volume_idx),
        "volume_dim": int(volume_idx),
        "state_names": [*(f"u_lat_{i}" for i in range(r_modes)), "volume"],
        "H": int(H),
        "W": int(W),

        "area_gain_pred_final_cm2": float(area_gain_pred_final),
    }

    print(f"[rollout vanilla NODE] r={r_modes}, sim={sim_id}, steps={n_steps}")
    print("  Final L2 error:", err_L2[-1])
    print("  Final volume error:", err_vol[-1])
    print("  Pred final net area gain [cm^2]:", rollout["area_gain_pred_final_cm2"])

    if make_plots:
        # Interactive/single-sim diagnostic plots -- skip during batch export
        # (save_val_rollouts_to_zarr passes make_plots=False): each plot call
        # costs 20-40s of matplotlib layout/text-metrics computation, which
        # dominates runtime and swamps the actual rollout cost (milliseconds)
        # across 149 validation sims.
        try:
            final_k = len(t_sim) - 1

            u_true_final = get_true_u_nodes_for_sim_step(
                sim_id=sim_id,
                step_k=final_k,
                sim_index=sim_index,
                time_vals=time_vals,
                zarr_disp_path=ZARR_DISP,
                snapshots_key="snapshots",
            )

            plot_disp_pred_vs_true_final(
                rollout,
                u_true_final,
                title_prefix=f"Final displacement comparison | sim={sim_id} | r={r_modes}"
            )

            G_true_final = get_true_bottom_growth_grid_for_sim_step(
                sim_id=sim_id,
                step_k=final_k,
                sim_index=sim_index,
                time_vals=time_vals,
                H=H, W=W,
                zarr_sdv_path=ZARR_SDV,
                folder_x="snaps_SDV1",
                folder_y="snaps_SDV4",
            )

            plot_growth_pred_vs_true_final(
                rollout,
                G_true_final,
                title_prefix=f"Final growth comparison | sim={sim_id} | r={r_modes}"
            )

        except Exception as e:
            print("[warn] Could not plot final displacement/growth fields:", repr(e))

    return rollout


# In[44]:


import torch
import torch.nn as nn


class VelocityNet(nn.Module):
    """
    The Neural ODE's velocity/dynamics network: a 2-hidden-layer tanh MLP
    mapping the augmented state x_k = [z_k, sp_k, design, e_k, I_k] (plus an
    optional closure feature) to the latent-state time-derivative dz/dt,
    used in the explicit update z_{k+1} = z_k + dt_k * f(x_k).

    dim_closure defaults to 0 (unused) for Model A, which never consumes
    growth feedback; the parameter is kept so that checkpoints saved by the
    shared training code across all model variants load with the same
    class signature.
    """
    def __init__(self,
                 dim_state: int,
                 dim_sp: int = 1,
                 dim_design: int = 7,
                 dim_error: int = 1,
                 dim_integral: int = 1,
                 dim_closure: int = 0,
                 hidden: int = 128):
        super().__init__()

        self.dim_state = dim_state

        input_dim = dim_state + dim_sp + dim_design + dim_error + dim_integral + dim_closure

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim_state),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# In[ ]:





# In[ ]:





# In[ ]:





# In[45]:


import torch
import numpy as np

def build_vanilla_node_from_ckpt(ckpt: dict, device: torch.device):
    """
    Reconstruct Model A's VelocityNet (the "vanilla" open-loop NODE, i.e.
    no growth feedback) from a saved checkpoint, inferring its architecture
    directly from the state_dict tensor shapes rather than hard-coding them:
      - "net.4.bias" is the final Linear layer's bias -> its length is the
        latent state dimension D_state (output size of the velocity net).
      - "net.0.bias" is the first Linear layer's bias -> its length is the
        hidden layer width.
    The remaining input-dimension pieces (dim_sp, dim_design, dim_error,
    dim_integral, dim_closure) are read from the checkpoint dict if present,
    else fall back to Model A's standard defaults. Returns the model in
    eval() mode with weights loaded (strict=True).
    """
    sd = ckpt["model_state_dict"]

    if "net.4.bias" not in sd:
        raise KeyError("Could not find 'net.4.bias' in checkpoint.")
    D_state = int(sd["net.4.bias"].numel())

    if "net.0.bias" not in sd:
        raise KeyError("Could not find 'net.0.bias' in checkpoint.")
    hidden = int(sd["net.0.bias"].numel())

    print("[infer vanilla NODE arch]")
    print("  D_state =", D_state)
    print("  hidden  =", hidden)

    dim_sp = int(ckpt.get("dim_sp", 1))
    dim_design = int(ckpt.get("dim_design", 7))
    dim_error = int(ckpt.get("dim_error", 1))
    dim_integral = int(ckpt.get("dim_integral", 1))
    dim_closure = int(ckpt.get("dim_closure", 0))

    model = VelocityNet(
        dim_state=D_state,
        dim_sp=dim_sp,
        dim_design=dim_design,
        dim_error=dim_error,
        dim_integral=dim_integral,
        dim_closure=dim_closure,
        hidden=hidden,
    ).to(device)

    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def _coerce_aux_arrays(aux: dict | None) -> dict:
    """Return a copy of the checkpoint's auxiliary-metadata dict `aux` with
    the "train_sims"/"val_sims" entries (if present) coerced to int64 numpy
    arrays, for consistent downstream indexing."""
    if aux is None:
        return {}

    aux2 = dict(aux)
    for key in ["train_sims", "val_sims"]:
        if key in aux2 and aux2[key] is not None:
            aux2[key] = np.asarray(aux2[key], dtype=np.int64)

    return aux2


def plot_rollout_volume_and_error(roll, title_prefix=""):
    """
    Robust volume plotter.

    Prefers:
      - roll["vol_true"], roll["vol_pred"] if present
    Otherwise:
      - derives from roll["z_true"], roll["z_pred"] using an explicit volume index:
            roll["volume_idx"] or roll["volume_dim"]
    """
    t = np.asarray(roll["t"])

    # ---------------------------
    # 1) Get volume signals
    # ---------------------------
    has_vol = ("vol_true" in roll) and ("vol_pred" in roll)
    if has_vol:
        v_true = np.asarray(roll["vol_true"])
        v_pred = np.asarray(roll["vol_pred"])
    else:
        # Fall back to indexing into z vectors
        if "z_true" not in roll or "z_pred" not in roll:
            raise KeyError("Need either (vol_true, vol_pred) or (z_true, z_pred) in rollout dict.")

        z_true = np.asarray(roll["z_true"])
        z_pred = np.asarray(roll["z_pred"])

        # robust volume index lookup
        if "volume_idx" in roll:
            volume_idx = int(roll["volume_idx"])
        elif "volume_dim" in roll:
            volume_idx = int(roll["volume_dim"])
        else:
            raise KeyError(
                "Rollout dict missing 'volume_idx'/'volume_dim'. "
                "Add it during rollout so volume indexing isn't hard-coded."
            )

        v_true = z_true[:, volume_idx]
        v_pred = z_pred[:, volume_idx]

    # ---------------------------
    # 2) Error
    # ---------------------------
    err_vol = np.asarray(roll.get("err_vol", np.abs(v_pred - v_true)))

    # ---------------------------
    # 3) Title
    # ---------------------------
    r_modes = roll.get("r_modes", None)
    sim_id  = roll.get("sim_id", None)

    pieces = []
    if title_prefix:
        pieces.append(title_prefix)
    if r_modes is not None:
        pieces.append(f"r={r_modes}")
    if sim_id is not None:
        pieces.append(f"sim={sim_id}")
    title_str = " | ".join(pieces) if pieces else "Rollout"

    # ---------------------------
    # 4) Plot
    # ---------------------------
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    ax0 = axes[0]
    ax0.plot(t, v_true, label="true volume")
    ax0.plot(t, v_pred, "--", label="pred volume")
    ax0.set_ylabel("Volume")
    ax0.set_title(title_str)
    ax0.legend()
    ax0.grid(alpha=0.3)

    ax1 = axes[1]
    ax1.plot(t, err_vol, label="|V_pred - V_true|")
    ax1.set_xlabel("Time [days]")
    ax1.set_ylabel("Abs volume error")
    ax1.grid(alpha=0.3)
    ax1.legend()

    plt.tight_layout()
    plt.show()


# In[48]:


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


# In[49]:


import numpy as np

def compute_net_area_gain_from_lamdag_elem(lamdag_elem: np.ndarray, A0_cm2: float = 0.25) -> float:
    """
    lamdag_elem: (Ne,2) with columns [Gx, Gy] = [lambda_g_x, lambda_g_y] (RAW, not normalized)
    Each element final area = Gx*Gy*A0.
    Net gain per elem = (Gx*Gy*A0 - A0).
    Total net gain = sum over elems.
    """
    gx = lamdag_elem[:, 0]
    gy = lamdag_elem[:, 1]
    A_final = gx * gy * A0_cm2
    net_gain = A_final - A0_cm2
    return float(np.sum(net_gain))


# In[50]:


def compute_true_net_area_gain_final_for_sim(
    sim_id: int,
    sim_index: np.ndarray,
    time_vals: np.ndarray,
    H: int,
    W: int,
    zarr_sdv_path: str,
    folder_x: str = "snaps_SDV1",
    folder_y: str = "snaps_SDV4",
    A0_cm2: float = 0.25,
) -> float:
    """
    Loads true elem growth at the final frame for this sim, takes bottom half (H*W),
    then computes total net area gain.
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
    half = Ne_full // 2

    gx = Gx_sim_elem[final_k, half:]
    gy = Gy_sim_elem[final_k, half:]

    if gx.shape[0] != H * W:
        raise ValueError(f"Expected bottom elems H*W={H*W}, got {gx.shape[0]}")

    lamdag_elem_true = np.stack([gx, gy], axis=1).astype(np.float64)
    return compute_net_area_gain_from_lamdag_elem(lamdag_elem_true, A0_cm2=A0_cm2)


# In[51]:


# --- choose a sim ---
r_modes = 9
sim_id  = 584   # change to any valid sim index


# In[52]:


import numpy as np
import matplotlib.pyplot as plt

def eval_val_area_gain_scatter_vanilla(
    val_sims: list[int],
    r_modes: int,
    nodes_csv: str,
    elems_csv: str,
    max_steps: int | None = None,
):
    """
    Roll out Model A over each sim in `val_sims`, compare predicted vs. true
    final net skin-area gain, and scatter-plot the result (reproduces the
    Model A validation area-gain scatter). Returns a dict with the sim ids
    and matched true/predicted area-gain arrays for further analysis.
    """
    (U_lat,
     volume_snap,
     sim_index, time_vals,
     volume_SP, design_all,
     n_sims) = load_raw_data(r_modes)

    H, W, _ = get_bottom_grid_cache()

    preds = []
    trues = []
    sims_ok = []

    for sim_id in val_sims:
        print("\n==============================")
        print("VAL sim:", sim_id)

        roll = rollout_single_sim_vanilla_node(
            r_modes=r_modes,
            sim_id=sim_id,
            nodes_csv=nodes_csv,
            elems_csv=elems_csv,
            max_steps=max_steps,
        )

        pred_gain = roll["area_gain_pred_final_cm2"]

        true_gain = compute_true_net_area_gain_final_for_sim(
            sim_id=sim_id,
            sim_index=sim_index,
            time_vals=time_vals,
            H=H, W=W,
            zarr_sdv_path=ZARR_SDV,
            folder_x="snaps_SDV1",
            folder_y="snaps_SDV4",
            A0_cm2=0.25,
        )

        preds.append(pred_gain)
        trues.append(true_gain)
        sims_ok.append(sim_id)

        print(f"  true area gain [cm^2]: {true_gain:.6g}")
        print(f"  pred area gain [cm^2]: {pred_gain:.6g}")

    preds = np.array(preds, dtype=np.float64)
    trues = np.array(trues, dtype=np.float64)

    plt.figure(figsize=(6.5, 6))
    plt.scatter(trues, preds)
    mn = float(min(trues.min(), preds.min()))
    mx = float(max(trues.max(), preds.max()))
    plt.plot([mn, mx], [mn, mx], "--")
    plt.xlabel("True final net area gain [cm^2]")
    plt.ylabel("Pred final net area gain [cm^2]")
    plt.title(f"Validation: Final net area gain (true vs pred) | r={r_modes} | n={len(sims_ok)}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        "val_sims": sims_ok,
        "true_area_gain_cm2": trues,
        "pred_area_gain_cm2": preds,
    }


# In[54]:


# In[185]:


import numpy as np
import matplotlib.pyplot as plt

def plot_rel_error_vs_true_area_gain(out: dict, ymax: float = 2.0):
    """
    out: dict returned by eval_val_area_gain_scatter
      keys:
        - "val_sims"
        - "true_area_gain_cm2"
        - "pred_area_gain_cm2"

    Plots:
      relative error vs absolute true area gain,
      with y-axis clipped to ymax (default = 200%).
    """
    trues = np.asarray(out["true_area_gain_cm2"], dtype=np.float64)
    preds = np.asarray(out["pred_area_gain_cm2"], dtype=np.float64)

    abs_true = np.abs(trues)
    rel_err = np.abs(preds - trues) / np.maximum(abs_true, 1e-12)

    # --- Plot ---
    plt.figure(figsize=(7, 5))
    plt.scatter(abs_true, rel_err, alpha=0.8)
    plt.ylim(0.0, ymax)
    plt.xlabel("Absolute true final net area gain [cm$^2$]")
    plt.ylabel("Relative error |pred − true| / |true|")
    plt.title(f"Relative error vs |true area gain|  (y ≤ {int(100*ymax)}%)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Stats (unclipped) ---
    pcts = [50, 75, 90, 95, 99]
    pct_vals = np.percentile(rel_err, pcts)

    print("Relative error statistics (unclipped):")
    print(f"  mean    : {np.mean(rel_err):.4f}")
    print(f"  median : {np.median(rel_err):.4f}")
    print(f"  max    : {np.max(rel_err):.4f}")
    for p, v in zip(pcts, pct_vals):
        print(f"  p{p:02d}     : {v:.4f}")

    # how many got clipped in the plot
    frac_clipped = np.mean(rel_err > ymax)
    print(f"\nFraction with rel_err > {ymax:.1f} (clipped in plot): {frac_clipped:.2%}")


# In[186]:


import numpy as np
import matplotlib.pyplot as plt

def plot_true_area_gain_for_large_rel_error(out: dict, rel_thresh: float = 2.0):
    """
    Plots distribution of |true area gain| for sims with relative error > rel_thresh.
    """

    trues = np.asarray(out["true_area_gain_cm2"], dtype=np.float64)
    preds = np.asarray(out["pred_area_gain_cm2"], dtype=np.float64)

    abs_true = np.abs(trues)
    rel_err = np.abs(preds - trues) / np.maximum(abs_true, 1e-12)

    mask = rel_err > rel_thresh
    abs_true_bad = abs_true[mask]

    print(f"\nNumber of sims with rel_err > {rel_thresh}: {mask.sum()} / {len(mask)}")

    if mask.sum() == 0:
        print("No sims exceed threshold.")
        return

    print("\nStats of |true area gain| for high relative error sims:")
    print(f"  mean   : {abs_true_bad.mean():.6f}")
    print(f"  median : {np.median(abs_true_bad):.6f}")
    print(f"  min    : {abs_true_bad.min():.6f}")
    print(f"  max    : {abs_true_bad.max():.6f}")

    # Histogram
    plt.figure(figsize=(7, 5))
    plt.hist(abs_true_bad, bins=30)
    plt.xlabel("|True final net area gain| [cm$^2$]")
    plt.ylabel("Count")
    plt.title(f"True area gain for sims with rel_err > {rel_thresh}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Optional: also show full distribution for comparison
    plt.figure(figsize=(7, 5))
    plt.hist(abs_true, bins=30)
    plt.xlabel("|True final net area gain| [cm$^2$]")
    plt.ylabel("Count")
    plt.title("Full distribution of true area gain (all sims)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# In[ ]:





# ## Generate Model Summary Statistics on the Validation Set

# In[55]:


import os
import json
import numpy as np
import zarr


def _zarr_write_array(grp, name, arr, overwrite=True):
    """
    Robust helper for writing arrays across common zarr versions.
    """
    arr = np.asarray(arr)
    if name in grp and overwrite:
        del grp[name]

    try:
        grp.create_dataset(name, data=arr, shape=arr.shape, dtype=arr.dtype, overwrite=overwrite)
    except TypeError:
        try:
            grp.create_dataset(name, data=arr, overwrite=overwrite)
        except Exception:
            grp.array(name, arr, overwrite=overwrite)


def _compute_Ag_hist_from_lamdag_stack(lamdag_stack: np.ndarray, A0_cm2: float = 0.25) -> np.ndarray:
    """
    Reduce a per-element growth-stretch history to a net skin-area-gain
    time series Ag(t) = sum_elems (lambda_g_x*lambda_g_y - 1) * A0.

    lamdag_stack: (T, Ne, 2)
    returns: (T,)
    """
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
    A0_cm2: float = 0.25,
    overwrite: bool = True,
    save_true_fields: bool = False,
):
    """
    Generic validation-rollout exporter for any of your 4 model variants.

    Parameters
    ----------
    zarr_path : str
        Output zarr store path for this model.
    val_sims : iterable of int
        Validation simulation IDs.
    rollout_fn : callable
        One of:
            rollout_single_sim_vanilla_node
            rollout_single_sim_ag_node
            rollout_single_sim_pca_node
            rollout_single_sim_cnn_node
    rollout_kwargs : dict
        Extra kwargs passed into rollout_fn, e.g.:
            {
                "nodes_csv": ...,
                "elems_csv": ...,
                "max_steps": None,
                ...
            }
    r_modes : int
        Number of displacement latent modes.
    model_name : str
        Name to store in zarr attrs, e.g. "vanilla", "ag", "pca", "cnn".
    A0_cm2 : float
        Element reference area used for Ag integration.
    overwrite : bool
        Whether to overwrite existing zarr.
    save_true_fields : bool
        Placeholder flag in case later you want to augment with true decoded fields too.

    Returns
    -------
    summary : dict
        Basic summary info.
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

    for sim_id in val_sims:
        print("\n========================================")
        print(f"[save zarr] model={model_name} | sim={sim_id}")

        roll = rollout_fn(
            r_modes=r_modes,
            sim_id=int(sim_id),
            **rollout_kwargs,
        )

        sim_grp = sims_grp.require_group(f"sim_{int(sim_id):05d}")

        # --- core trajectories ---
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

        # --- Ag history ---
        if "Ag_hist_raw_cm2" in roll:
            Ag_pred_cm2 = np.asarray(roll["Ag_hist_raw_cm2"], dtype=np.float32)
        else:
            Ag_pred_cm2 = _compute_Ag_hist_from_lamdag_stack(
                lamdag_pred_elem, A0_cm2=A0_cm2
            ).astype(np.float32)

        # --- save arrays ---
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

        if "Ag_hist_norm" in roll:
            _zarr_write_array(sim_grp, "Ag_hist_norm", np.asarray(roll["Ag_hist_norm"], dtype=np.float32))
        if "g_pca_hist" in roll:
            _zarr_write_array(sim_grp, "g_pca_hist", np.asarray(roll["g_pca_hist"], dtype=np.float32))

        # --- attrs ---
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

        print(f"  saved T={t.shape[0]} | N_nodes={u_pred_nodes.shape[1]} | final Ag={Ag_pred_cm2[-1]:.6g}")

    # --- root-level index arrays ---
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

    print("\n========================================")
    print(f"[done] wrote validation rollout zarr for model='{model_name}'")
    print("path:", zarr_path)
    print("n_sims:", len(sims_written))

    return summary


# In[56]:


def save_val_rollouts_vanilla(
    *,
    zarr_path: str,
    val_sims,
    r_modes: int,
    nodes_csv: str,
    elems_csv: str,
    max_steps: int | None = None,
):
    """Convenience wrapper around save_val_rollouts_to_zarr specialized to
    Model A (vanilla, open-loop NODE) rollouts."""
    return save_val_rollouts_to_zarr(
        zarr_path=zarr_path,
        val_sims=val_sims,
        rollout_fn=rollout_single_sim_vanilla_node,
        rollout_kwargs={
            "nodes_csv": nodes_csv,
            "elems_csv": elems_csv,
            "max_steps": max_steps,
        },
        r_modes=r_modes,
        model_name="vanilla",
    )
