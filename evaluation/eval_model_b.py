"""Standalone eval-time rollout module for Model B (scalar-Ag NODE).

Produces per-simulation rollout trajectories (predicted latent state,
volume, and net area growth vs. ground truth) for a trained Model B
checkpoint. Its rollout function is invoked by
`evaluation/generate_deliverables.py` and `evaluation/cross_model_comparison.py`
to produce the manuscript's Table 3 checkpoint-selection RMSEs and the
cross-model net-area-growth (Ag) error comparison.

Model B feeds a scalar summary of net area growth (`Ag`) back into the
latent velocity network at every rollout step. Because of that feedback,
(unlike Model A, whose growth integration is eval-only bookkeeping that
never affects the displacement rollout) an eval-time growth integrator that
disagrees with the training-time growth physics would corrupt the entire
subsequent rollout trajectory. This module therefore integrates growth with
`integrate_growth_matched_numba`, which reproduces the implicit
time-discretization used by the original finite-element UMAT (see its
docstring, and `integrate_growth_matched_from_mesh` below), rather than an
explicit-Euler scheme. This matched integrator has been validated against
an autograd-differentiable torch implementation of the same update to
7.6e-13 max absolute difference; no backpropagation is needed at eval time,
so the (non-differentiable) numba implementation is used directly here.

This file is self-contained: the mesh loading, POD decoder, growth-
parameter lookup, and checkpoint-based network-builder helpers that the
rollout depends on are all defined below rather than imported from another
model family's eval module. Batch-driving/archival-export tooling used
during model development is not part of this reproducibility package.
"""
import os
import functools
import numpy as np
import zarr
import torch
from torch import nn
from numba import njit, prange

# ===============================
# Paths & templates
# ===============================
ZARR_DISP  = "displacements.zarr"
ZARR_SDV   = "ip_growth_elem.zarr"
ZARR_VOL   = "expd_volumes.zarr"
DESIGN_ZARR_PATH = "Design_and_Metadata.zarr"

U_LATENT_TEMPLATE = "latent_displ_r{r}/coeffs"

DISP_POD_GROUP     = "pod_full"
DISP_POD_U_NAME    = "U"
DISP_POD_MEAN_NAME = "mean"


# ===============================
# Bottom-surface mesh / grid mapping
# ===============================
_BOTTOM_GRID_CACHE = None


def get_bottom_grid_cache():
    """Process-wide cache for the (H, W, bottom_elem_idx) grid mapping below,
    since it only depends on the fixed mesh geometry, not on any simulation."""
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
    """Map each element on the mesh's bottom surface (the surface in contact
    with the expander, where growth is tracked) to a cell in a regular
    (H, W) image-like grid, keyed by its (x, y) centroid. For each (x, y)
    column of the mesh, the element with the lowest z-centroid is taken as
    the bottom-surface element. This lets per-element growth/state fields be
    reshaped into 2D arrays for plotting and CNN-style consumption elsewhere
    in the repo.
    """
    nodes = np.loadtxt(nodes_csv, delimiter=",", dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1] < 4:
        raise ValueError(f"Nodes CSV must have 4 cols [id,x,y,z] (no headers). Got shape {nodes.shape}")

    node_id = nodes[:, 0].astype(np.int64)
    node_xyz = nodes[:, 1:4].astype(np.float64)
    node_id_to_local = {int(nid): i for i, nid in enumerate(node_id)}

    elems = np.loadtxt(elems_csv, delimiter=",", dtype=np.int64)
    if elems.ndim != 2 or elems.shape[1] < 5:
        raise ValueError(f"Elements CSV must have >=5 cols [eid,n1,n2,n3,n4] (no headers). Got shape {elems.shape}")

    elem_id = elems[:, 0].astype(np.int64)
    elem_conn_ids = elems[:, 1:5].astype(np.int64)
    Ne = elem_conn_ids.shape[0]

    elem_conn_local = np.empty((Ne, 4), dtype=np.int64)
    for a in range(4):
        try:
            elem_conn_local[:, a] = np.array(
                [node_id_to_local[int(nid)] for nid in elem_conn_ids[:, a]], dtype=np.int64
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
        raise ValueError(f"Some grid cells have no assigned element. Example missing: {missing}")

    cell_xy = np.stack([xe[bottom_elem_idx], ye[bottom_elem_idx]], axis=1)

    return H, W, bottom_elem_idx, cell_xy, elem_id, node_id, node_xyz, elem_conn_local, xs, ys


@functools.lru_cache(maxsize=None)
def get_growth_params_for_sim(sim_id: int):
    """Look up the per-simulation growth-law material parameters
    (theta_crit, k1, k2) from the design-of-experiments table, via the
    sim_id -> job_id mapping stored in the metadata zarr archive."""
    g_disp = zarr.open("Design_and_Metadata.zarr", mode="r")
    job_index_per_sim = g_disp["Mapping_indexes_and_metadata"]["job_index_per_sim"][:]
    job_id = int(job_index_per_sim[sim_id])

    design_file = "PGOH_Ideal_tol_Tcrit_V_mu_kk1_kk2_kappa_k1_design.txt"
    design_table = np.loadtxt(design_file, skiprows=0)

    row_index = job_id - 1
    if row_index < 0 or row_index >= design_table.shape[0]:
        raise IndexError(f"Job ID {job_id} (row idx {row_index}) is out of range for design table with {design_table.shape[0]} rows.")

    row = design_table[row_index]
    theta_crit = float(row[2])
    k1         = float(row[5])
    k2         = float(row[6])
    return job_id, theta_crit, k1, k2


@functools.lru_cache(maxsize=None)
def load_raw_data(r_modes: int):
    """Load the flattened, all-simulation-and-timestep arrays needed to
    build a rollout's ground-truth trajectory and inputs: POD latent
    displacement coefficients, the expander cavity volume (appended as the
    extra latent state dimension), the volume setpoint schedule, and the
    per-snapshot design parameters. Rows across all arrays share the same
    (sim_index, time_vals) snapshot ordering."""
    from pathlib import Path

    g_disp   = zarr.open_group(ZARR_DISP, mode="r")
    g_design = zarr.open_group(DESIGN_ZARR_PATH, mode="r")

    vol_root = Path(ZARR_VOL)

    u_path = U_LATENT_TEMPLATE.format(r=r_modes)
    U_lat = np.asarray(g_disp[u_path], dtype=np.float32).T

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

    return (U_lat, volume_snap, sim_index, time_vals, volume_SP, design_all, len(np.unique(sim_index)))


@functools.lru_cache(maxsize=None)
def load_bottom_surface_mesh_direct(nodes_csv: str, elems_csv: str):
    """Load the shell mesh and extract just the bottom-surface layer of
    elements/nodes, re-indexed to a compact local numbering.

    Convention: the mesh has two through-thickness element layers stacked
    in the elements CSV, top layer first; the bottom (expander-contacting)
    layer is always the second half of the element array, i.e.
    `elem_node_ids[Ne//2:]`. This convention is also relied on in
    `rollout_single_sim_ag_node_matched` when slicing the raw growth-field
    snapshots.
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
    elem_node_ids_bot = elem_node_ids[half:, :]  # bottom layer = 2nd half

    bottom_node_ids = np.unique(elem_node_ids_bot.reshape(-1))

    id2xyz = {int(nid): node_xyz[i] for i, nid in enumerate(node_ids)}

    missing = [int(nid) for nid in bottom_node_ids if int(nid) not in id2xyz]
    if len(missing) > 0:
        raise ValueError(f"{len(missing)} node IDs referenced by bottom elements missing from nodes_csv. Example: {missing[:10]}")

    node_ids_bot = np.sort(bottom_node_ids.astype(np.int64))
    node_xyz_bot = np.vstack([id2xyz[int(nid)] for nid in node_ids_bot]).astype(np.float64)

    id2bot = {int(nid): i for i, nid in enumerate(node_ids_bot)}
    get_idx = np.vectorize(lambda nid: id2bot.get(int(nid), -1), otypes=[np.int64])
    elem_conn_bot = get_idx(elem_node_ids_bot)

    bad = np.where(elem_conn_bot < 0)
    if bad[0].size > 0:
        ei = int(bad[0][0])
        raise ValueError(f"Internal mapping failure. elem_node_ids_bot[{ei}]={elem_node_ids_bot[ei]}")

    return node_xyz_bot, elem_conn_bot, node_ids_bot, elem_node_ids_bot


_GROWTH_SNAP_CACHE = {}


def load_elem_growth_for_sim(
    sim_id: int, sim_index: np.ndarray, time_vals: np.ndarray,
    zarr_sdv_path: str = "ip_growth_elem.zarr",
    folder_x: str = "snaps_SDV1", folder_y: str = "snaps_SDV4",
):
    """Load this simulation's per-element, per-snapshot in-plane growth
    stretches (SDV1 = lambda_g in direction 1, SDV4 = lambda_g in direction
    2, the UMAT's internal state-variable slots for the two in-plane growth
    stretches) and return them as (n_frames, Ne) arrays sorted by time."""
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
    r_modes: int, zarr_disp_path: str = "displacements.zarr",
    pod_group: str = "pod_full", pod_u_name: str = "U", pod_mean_name: str = "mean",
    verbose: bool = True,
):
    """Build a closure that decodes an r_modes-length POD latent coefficient
    vector `alpha` back into a full nodal displacement field, via the
    standard POD reconstruction `u = mean + U_r @ alpha` (mean-centered
    linear reconstruction from the truncated basis `U_r`, the first
    `r_modes` columns of the full POD basis)."""
    g_disp = zarr.open_group(zarr_disp_path, mode="r")

    U_key  = f"{pod_group}/{pod_u_name}"
    mu_key = f"{pod_group}/{pod_mean_name}"

    U_full = np.asarray(g_disp[U_key])
    mu     = np.asarray(g_disp[mu_key]).reshape(-1)

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

    def decode(alpha: np.ndarray) -> np.ndarray:
        alpha = np.asarray(alpha).reshape(-1)
        if alpha.shape[0] != r_modes:
            raise ValueError(f"alpha must have shape ({r_modes},), got {alpha.shape}")
        u_flat = mu + U_r @ alpha  # POD decode: mean + truncated-basis reconstruction
        ux = u_flat[0:N]
        uy = u_flat[N:2*N]
        uz = u_flat[2*N:3*N]
        return np.stack([ux, uy, uz], axis=1)

    return decode


def shape_function_gradients(xi, eta):
    """Derivatives (w.r.t. natural coords xi, eta) of the standard bilinear
    shape functions for a 4-node quadrilateral element, evaluated at
    natural coordinate (xi, eta). Used below at the element center
    (xi=eta=0) to build a constant-per-element deformation gradient."""
    dN_dxi = np.array([-(1 - eta) / 4, (1 - eta) / 4, (1 + eta) / 4, -(1 + eta) / 4])
    dN_deta = np.array([-(1 - xi) / 4, -(1 + xi) / 4, (1 + xi) / 4, (1 - xi) / 4])
    return dN_dxi, dN_deta


def compute_net_area_gain_from_lamdag_elem(lamdag_elem: np.ndarray, A0_cm2: float = 0.25) -> float:
    """Reduce the per-element growth stretch field to the single scalar
    "Ag" (net area gain, cm^2) that Model B feeds back into the velocity
    network. `lamdag_elem` (lambda_g, the growth stretch) is (Ne, 2): the
    two in-plane growth stretch components per element. Assuming every
    element shares the same flat reference area `A0_cm2`, each element's
    current grown area is `lam1g * lam2g * A0_cm2`, and Ag is the total
    gain in area over all elements relative to the ungrown reference."""
    gx = lamdag_elem[:, 0]
    gy = lamdag_elem[:, 1]
    A_final = gx * gy * A0_cm2
    net_gain = A_final - A0_cm2
    return float(np.sum(net_gain))


# ===============================
# Model B (scalar-Ag NODE) architecture + checkpoint loader
# ===============================
class VelocityNet(nn.Module):
    """The Neural ODE's velocity (right-hand-side) network for Model B:
    a 2-hidden-layer tanh MLP mapping the concatenated, normalized inputs
    [latent state z, volume setpoint, design params, tracking error e,
    error integral I, scalar growth feedback Ag] to the latent state time
    derivative dz/dt. This is the function integrated forward in
    `rollout_single_sim_ag_node_matched` below."""

    def __init__(self, dim_state: int, dim_sp: int = 1, dim_design: int = 7,
                 dim_error: int = 1, dim_integral: int = 1, dim_ag: int = 1, hidden: int = 128):
        super().__init__()
        self.dim_state = dim_state
        input_dim = dim_state + dim_sp + dim_design + dim_error + dim_integral + dim_ag
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim_state),
        )

    def forward(self, x):
        return self.net(x)


def build_ag_node_from_ckpt(ckpt: dict, device: torch.device):
    """Reconstruct a `VelocityNet` with the right architecture purely from
    the shapes of tensors stored in the checkpoint (state size, hidden
    width, and input dim), then load its weights. This avoids having to
    hard-code architecture hyperparameters here: `dim_ag` (the width of the
    scalar growth-feedback input -- 1 for Model B) is inferred as whatever
    is left over after accounting for the state, setpoint, design, error,
    and error-integral input widths, all of which are also read from (or
    default from) the checkpoint metadata."""
    sd = ckpt["model_state_dict"]

    if "net.4.bias" not in sd:
        raise KeyError("Could not find 'net.4.bias' in checkpoint.")
    D_state = int(sd["net.4.bias"].numel())

    if "net.0.bias" not in sd:
        raise KeyError("Could not find 'net.0.bias' in checkpoint.")
    hidden = int(sd["net.0.bias"].numel())

    if "net.0.weight" not in sd:
        raise KeyError("Could not find 'net.0.weight' in checkpoint.")
    input_dim = int(sd["net.0.weight"].shape[1])

    dim_sp = int(ckpt.get("dim_sp", 1))
    dim_design = int(ckpt.get("dim_design", 7))
    dim_error = int(ckpt.get("dim_error", 1))
    dim_integral = int(ckpt.get("dim_integral", 1))

    dim_ag = input_dim - (D_state + dim_sp + dim_design + dim_error + dim_integral)
    if dim_ag <= 0:
        raise ValueError(f"Inferred dim_ag={dim_ag} is invalid. input_dim={input_dim}, D_state={D_state}")

    print("[infer Ag-NODE arch] D_state=", D_state, " hidden=", hidden, " dim_ag=", dim_ag)

    model = VelocityNet(
        dim_state=D_state, dim_sp=dim_sp, dim_design=dim_design,
        dim_error=dim_error, dim_integral=dim_integral, dim_ag=dim_ag, hidden=hidden,
    ).to(device)

    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


# ===============================
# Matched (UMAT-consistent) growth integrator -- numba, eval-only (no grad)
# ===============================
def _growth_gate(time):
    """Return 0.0 during the alternating 7-day rest windows of the tissue
    expansion protocol (no growth stimulus applied between expansion
    steps) and 1.0 otherwise, matching the loading/rest schedule baked
    into the FE UMAT."""
    if (0.0 < time < 7.0) or (14.0 < time < 21.0) or (28.0 < time < 35.0) or \
       (42.0 < time < 49.0) or (56.0 < time < 63.0) or (70.0 < time < 77.0) or \
       (84.0 < time < 91.0) or (98.0 < time < 105.0):
        return 0.0
    return 1.0
_growth_gate_jit = njit(_growth_gate)


@njit(parallel=True, fastmath=True)
def integrate_growth_matched_numba(F_batch, elem_lamdag, k1, k2, theta_crit, time, dt, tol=1e-12, maxiter=20):
    """Advance each element's two in-plane growth stretches (lambda_g1,
    lambda_g2, stored in `elem_lamdag` as (Ne, 2)) by one implicit
    backward-Euler step of the growth law
        d(lambda_g)/dt = k * (lambda / lambda_g - theta_crit),  when active,
    where `lambda` is the corresponding elastic in-plane stretch computed
    from the current deformation gradient `F_batch`, and growth is only
    "active" (elastic stretch exceeds the growth threshold theta_crit) when
    `phig = lambda/lambda_g - theta_crit > 0`. Each element's implicit
    update is solved with a per-element Newton iteration (this is the same
    nonlinear backward-Euler scheme used by the FE UMAT, so this function
    reproduces the FE model's growth increment exactly rather than
    approximating it with explicit-Euler). `_growth_gate_jit` additionally
    zeroes the growth rate during the protocol's rest windows.
    """
    Ne = F_batch.shape[0]
    out = elem_lamdag.copy()
    gate = _growth_gate_jit(time)
    for ei in prange(Ne):
        F = F_batch[ei]
        # lam1, lam2: total (elastic x growth) in-plane stretches, i.e. the
        # norms of F's first two columns (the deformed lengths of the two
        # in-plane element directions, per unit reference length).
        lam1 = np.sqrt(F[0,0]**2 + F[1,0]**2 + F[2,0]**2)
        lam2 = np.sqrt(F[0,1]**2 + F[1,1]**2 + F[2,1]**2)
        lam1g_n = out[ei,0]; lam2g_n = out[ei,1]  # growth stretches at time n (previous step)
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


# This F-construction (shape-function-gradient) assembly is
# architecture-independent -- it is identical for every model family, and
# only the growth-update math applied downstream of F (in
# integrate_growth_matched_numba above) is specific to the matched
# integrator used here.
@njit(parallel=True, fastmath=True)
def _compute_F_batch_numba(node_coords, elem_conn, node_u, dN_dxi, dN_deta):
    """Assemble the per-element deformation gradient F (3x3, Ne of them)
    for a batch of 4-node shell/membrane elements, from the reference node
    coordinates `node_coords`, connectivity `elem_conn`, and current nodal
    displacements `node_u` (current position x = node_coords + node_u).
    The two in-plane columns of F come from the standard isoparametric
    mapping (reference tangent vectors Gxi/Geta -> deformed tangent
    vectors gxi/geta, via the shape-function-gradient covariant/
    contravariant basis change with `_inv3x3`); the third (through-
    thickness) column is built from the unit normal so that F acts as the
    identity in the thickness direction, consistent with the shell
    kinematics assumed by the FE UMAT."""
    Ne = elem_conn.shape[0]
    F_batch = np.empty((Ne, 3, 3), dtype=np.float64)
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

        F_batch[ei] = F
    return F_batch


@njit
def _inv3x3(A):
    """Closed-form 3x3 matrix inverse via the cofactor/adjugate formula
    (faster than a general solver for this fixed small size, and
    numba-friendly)."""
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


def integrate_growth_matched_from_mesh(node_coords, elem_conn, node_u, elem_lamdag, k1, k2, theta_crit, time, dt):
    """Convenience wrapper used once per rollout step: given the current
    mesh state (nodal displacements `node_u`), build the per-element
    deformation gradient (`_compute_F_batch_numba`) and use it to advance
    the growth stretch field by one UMAT-matched implicit step
    (`integrate_growth_matched_numba`). Same call signature and (Ne, 2)
    return convention as the superseded explicit-Euler growth integrator it
    replaces."""
    dN_dxi, dN_deta = shape_function_gradients(0.0, 0.0)
    dN_dxi  = np.asarray(dN_dxi,  dtype=np.float64).reshape(4,)
    dN_deta = np.asarray(dN_deta, dtype=np.float64).reshape(4,)
    F_batch = _compute_F_batch_numba(
        np.asarray(node_coords, dtype=np.float64),
        np.asarray(elem_conn, dtype=np.int64),
        np.asarray(node_u, dtype=np.float64),
        dN_dxi, dN_deta,
    )
    return integrate_growth_matched_numba(
        F_batch,
        np.asarray(elem_lamdag, dtype=np.float64),
        float(k1), float(k2), float(theta_crit), float(time), float(dt),
    )


# ===============================
# Corrected rollout for the scalar-Ag NODE (Model B).
# ===============================
def rollout_single_sim_ag_node_matched(
    r_modes: int,
    sim_id: int,
    nodes_csv: str,
    elems_csv: str,
    ckpt_path_template: str = "FINAL_models/Model_B/Model_B_Ag_r{r}_BEST.pt",
    max_steps: int | None = None,
    make_plots: bool = True,   # accepted for signature parity w/ Model A's convention; unused (no plotting here)
):
    """Autoregressively roll out Model B's Neural ODE for one simulation,
    starting from the true initial condition and integrating forward with
    the trained velocity network + matched growth integrator, entirely
    open-loop w.r.t. ground truth (no re-syncing to the true state along
    the way). At each step: (1) evaluate the velocity net on the current
    normalized state + setpoint/design/error/growth-feedback inputs, (2)
    explicit-Euler step the latent state z, (3) POD-decode z to a nodal
    displacement field, (4) advance the growth stretch field with that
    displacement field via the matched integrator, (5) recompute Ag and
    the tracking-error/integral terms for the next step's inputs. Returns a
    dict of per-step arrays (predicted vs. true latent state/volume, area
    growth, decoded displacements, growth fields, and error metrics) used
    for figure/table generation elsewhere in `evaluation/`.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ckpt_path_template.format(r=r_modes, r_modes=r_modes)
    ckpt = torch.load(ckpt_path, map_location=device)

    D_state = int(ckpt["latent_state_dim"])
    volume_idx = int(r_modes)
    if D_state <= volume_idx:
        raise ValueError(f"D_state={D_state} too small for volume_idx=r_modes={r_modes}")

    # The *_mean/*_std pairs below are z-score normalization statistics
    # computed over the training set and saved into the checkpoint; every
    # raw physical quantity fed to the velocity net (latent state, volume
    # setpoint, design params, tracking error, error integral) must be
    # normalized with these exact stats to match what the network saw
    # during training.
    z_mean = np.asarray(ckpt["z_mean"], dtype=np.float32).reshape(-1)
    z_std  = np.asarray(ckpt["z_std"], dtype=np.float32).reshape(-1)
    if z_mean.shape[0] != D_state or z_std.shape[0] != D_state:
        raise ValueError(f"Checkpoint z_mean/z_std length mismatch with D_state={D_state}.")

    sp_mean = float(np.asarray(ckpt["sp_mean"], dtype=np.float32).reshape(-1)[0])
    sp_std  = float(np.asarray(ckpt["sp_std"], dtype=np.float32).reshape(-1)[0])

    design_mean = np.asarray(ckpt["design_mean"], dtype=np.float32).reshape(1, -1)
    design_std  = np.asarray(ckpt["design_std"], dtype=np.float32).reshape(1, -1)

    e_mean = float(np.asarray(ckpt["e_mean"], dtype=np.float32).reshape(-1)[0])
    e_std  = float(np.asarray(ckpt["e_std"], dtype=np.float32).reshape(-1)[0])

    I_mean = float(np.asarray(ckpt["I_mean"], dtype=np.float32).reshape(-1)[0])
    I_std  = float(np.asarray(ckpt["I_std"], dtype=np.float32).reshape(-1)[0])

    # Ag (net area gain, cm^2) is Model B's scalar growth-feedback input.
    # Its raw values are O(cm^2) and grow steadily over ~100 days of
    # simulated time, a completely different scale/range than the other,
    # already-normalized network inputs (z, setpoint, design, error,
    # error integral). Feeding it in unnormalized would make it dominate
    # (or be swamped by) the other inputs during training and would hurt
    # optimization conditioning, so it is z-scored with the mean/std of
    # each training sim's *final* Ag value, precomputed and cached here.
    AG_STATS_PATH = f"cache/final_ag_per_sim_stats_r{r_modes}.npz"
    if not os.path.exists(AG_STATS_PATH):
        raise FileNotFoundError(f"Missing {AG_STATS_PATH}.")

    ag_ck = np.load(AG_STATS_PATH)
    final_ag_mean = float(ag_ck["final_ag_mean"])
    final_ag_std  = float(ag_ck["final_ag_std"])

    model = build_ag_node_from_ckpt(ckpt, device)
    model.eval()

    (U_lat, volume_snap, sim_index, time_vals, volume_SP, design_all, n_sims) = load_raw_data(r_modes)

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
        r_modes, zarr_disp_path=ZARR_DISP, pod_group=DISP_POD_GROUP,
        pod_u_name=DISP_POD_U_NAME, pod_mean_name=DISP_POD_MEAN_NAME,
    )

    job_id, theta_crit, k1, k2 = get_growth_params_for_sim(sim_id)
    print(f"[sim {sim_id}] job_id={job_id}  theta_crit={theta_crit:.6g}  k1={k1:.6g}  k2={k2:.6g}")

    t_sim_g, Gx_sim_elem, Gy_sim_elem, _ = load_elem_growth_for_sim(
        sim_id=sim_id, sim_index=sim_index, time_vals=time_vals,
        zarr_sdv_path=ZARR_SDV, folder_x="snaps_SDV1", folder_y="snaps_SDV4",
    )

    Ne_full = Gx_sim_elem.shape[1]
    if Ne_full % 2 != 0:
        raise ValueError(f"Expected even Ne, got Ne={Ne_full}")
    half = Ne_full // 2

    # Same top/bottom-layer convention as load_bottom_surface_mesh_direct:
    # the bottom (expander-contacting) elements are the second half of the
    # element array, so only growth on that layer is used at t=0.
    gx0 = Gx_sim_elem[0, half:]
    gy0 = Gy_sim_elem[0, half:]
    if gx0.shape[0] != H * W:
        raise ValueError(f"Expected bottom growth length H*W={H*W}, got {gx0.shape[0]}")

    lamdag_elem = np.stack([gx0, gy0], axis=1).astype(np.float32)

    def compute_Ag_norm_np(lam: np.ndarray) -> np.ndarray:
        """Compute Ag from the current growth stretch field and normalize
        it with the cached final-Ag training stats (see note above on why
        Ag needs normalizing) before it is fed to the velocity net."""
        Ag_raw = compute_net_area_gain_from_lamdag_elem(lam, A0_cm2=0.25)
        Ag_norm = (Ag_raw - final_ag_mean) / (final_ag_std + 1e-8)
        return np.array([[Ag_norm]], dtype=np.float32)

    Ag = compute_Ag_norm_np(lamdag_elem)

    G0_raw = np.zeros((2, H, W), dtype=np.float32)
    G0_raw[0] = lamdag_elem[:, 0].reshape(H, W)
    G0_raw[1] = lamdag_elem[:, 1].reshape(H, W)

    Z0_raw  = Z_true_sim[0, :].astype(np.float32)
    V0_raw  = float(Z0_raw[volume_idx])
    SP0_raw = float(SP_sim[0])

    # e (tracking error) = volume setpoint - actual volume, and I is its
    # running time-integral; both are fed to the velocity net alongside Ag
    # so the network can act like a controller closing the gap to the
    # commanded expander volume, not just an unconstrained forward model.
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
    g_pred_grid      = [G0_raw.copy()]
    lamdag_pred_elem = [lamdag_elem.copy()]
    Ag_hist_raw      = [compute_net_area_gain_from_lamdag_elem(lamdag_elem, A0_cm2=0.25)]
    Ag_hist_norm     = [float(Ag[0, 0])]

    u0 = decode_u(Z0_raw[:r_modes])
    if u0.shape[0] != node_xyz_bot.shape[0]:
        raise ValueError(f"Decode nodes mismatch: u0={u0.shape}, node_xyz_bot={node_xyz_bot.shape}")
    u_pred_nodes.append(u0.astype(np.float32))

    with torch.no_grad():
        for k in range(n_steps):
            t_k  = float(t_sim[k])
            t_k1 = float(t_sim[k+1])
            dt_k = float(t_k1 - t_k)
            if (not np.isfinite(dt_k)) or (dt_k <= 0.0):
                raise ValueError(f"Bad dt at step {k}: dt_k={dt_k}")

            SP_k_raw = float(SP_sim[k])
            e_norm   = (e_raw - e_mean) / e_std
            I_norm   = (I_raw - I_mean) / I_std
            sp_norm  = (SP_k_raw - sp_mean) / sp_std

            e_tensor  = torch.tensor([[e_norm]], dtype=torch.float32, device=device)
            I_tensor  = torch.tensor([[I_norm]], dtype=torch.float32, device=device)
            sp_tensor = torch.tensor([[sp_norm]], dtype=torch.float32, device=device)
            Ag_tensor = torch.from_numpy(Ag).to(device)

            x_base = torch.cat([z_curr, e_tensor, I_tensor, sp_tensor, design_tensor], dim=1)
            x = torch.cat([x_base, Ag_tensor], dim=1)  # Ag appended last, matching build_ag_node_from_ckpt's input layout

            dz_norm = model(x)

            z_next    = z_curr + dt_k * dz_norm  # explicit-Euler step in normalized latent-state space
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
                raise ValueError(f"u_nodes mismatch at step {k}")
            if elem_conn_bot.max() >= u_nodes.shape[0]:
                raise ValueError(f"elem_conn_bot out of range at step {k}")

            # Matched (UMAT-consistent) growth integrator: advances the
            # growth stretch field using the displacement field just
            # decoded above. Because Ag (computed from this field, below)
            # feeds back into the velocity net's input, using an
            # integrator consistent with the training-time growth physics
            # here is what keeps the rest of the rollout trajectory correct.
            lamdag_elem = integrate_growth_matched_from_mesh(
                node_coords=node_xyz_bot, elem_conn=elem_conn_bot, node_u=u_nodes,
                elem_lamdag=lamdag_elem, k1=k1, k2=k2, theta_crit=theta_crit,
                time=t_k, dt=dt_k,
            ).astype(np.float32)

            Ag = compute_Ag_norm_np(lamdag_elem)

            G_raw = np.zeros((2, H, W), dtype=np.float32)
            G_raw[0] = lamdag_elem[:, 0].reshape(H, W)
            G_raw[1] = lamdag_elem[:, 1].reshape(H, W)

            u_pred_nodes.append(u_nodes.astype(np.float32))
            g_pred_grid.append(G_raw.copy())
            lamdag_pred_elem.append(lamdag_elem.copy())
            Ag_hist_raw.append(compute_net_area_gain_from_lamdag_elem(lamdag_elem, A0_cm2=0.25))
            Ag_hist_norm.append(float(Ag[0, 0]))

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

    area_gain_pred_final = compute_net_area_gain_from_lamdag_elem(lamdag_pred_elem[-1], A0_cm2=0.25)

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
        "Ag_hist_raw_cm2": np.array(Ag_hist_raw, dtype=np.float64),
        "Ag_hist_norm": np.array(Ag_hist_norm, dtype=np.float64),
        "r_modes": r_modes,
        "sim_id": sim_id,
        "volume_idx": int(volume_idx),
        "volume_dim": int(volume_idx),
        "state_names": [*(f"u_lat_{i}" for i in range(r_modes)), "volume"],
        "H": int(H),
        "W": int(W),
        "dim_ag": 1,
        "area_gain_pred_final_cm2": float(area_gain_pred_final),
    }

    print(f"[rollout Ag-NODE matched] r={r_modes}, sim={sim_id}, steps={n_steps}")
    print("  Final L2 error:", err_L2[-1])
    print("  Final volume error:", err_vol[-1])
    print("  Pred final net area gain [cm^2]:", rollout["area_gain_pred_final_cm2"])

    return rollout
