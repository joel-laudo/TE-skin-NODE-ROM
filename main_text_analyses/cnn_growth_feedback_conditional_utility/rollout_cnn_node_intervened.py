"""Intervention-capable duplicate of Model D's authoritative rollout
(eval_model_d.rollout_single_sim_cnn_node_matched), used by
intervention_sweep.py and functional_substitution.py to test
whether -- and how -- the NODE's dynamics depend on the CNN-encoded growth
feature h_g.

The ONLY functional change from the original: instead of calling
`model(x_base, G_tensor)` as a black box, this replicates VelocityNet's
exact internal sequence manually --

    h_true = model.growth_encoder(G_tensor)
    h_used = intervention_fn(sim_id, step_k, t_k, t_final, vol_k, x_base, h_true)
    x_full = torch.cat([x_base, h_used], dim=-1)
    dz_norm = model.net(x_full)

-- so `intervention_fn` can substitute anything for h_g while everything
else (growth integration, decoding, PI bookkeeping) proceeds byte-for-byte
identically to the authoritative rollout. Two invariants are enforced by
construction, so that a measured accuracy drop can be attributed purely to
the substituted h_g rather than to some other side effect of the
intervention: (1) growth integration is never touched -- it always
consumes the model's own decoded z_next (produced from whatever dz_norm
resulted from h_used); (2) any online-predicted substitution must use the
model's own live current-step state, not a cached/ground-truth value --
x_base/t_k/vol_k are always this exact step's freshly-computed values.

NOTE: must be run with the working directory set to the repository root
(where the FE dataset from data/DATA_AVAILABILITY.md has been placed) --
this module does not chdir. It adds evaluation/ to sys.path relative to its
own file location so it can import the shared_eval_utils/eval_model_d
modules regardless of cwd.

Unified intervention_fn signature (all interventions below share it, even
when they ignore most arguments):
    intervention_fn(sim_id: int, step_k: int, t_k: float, t_final: float,
                     vol_k: float, x_base: Tensor[1,D], x_base_raw: ndarray[1,D],
                     h_true: Tensor[1,dim_gfeat]) -> Tensor[1,dim_gfeat] (or ndarray)

x_base is Model D's own INTERNALLY NORMALIZED tensor (checkpoint z-score
stats) -- exactly what VelocityNet.net actually consumes. x_base_raw is the
same [z,e,I,sp,design] state in RAW PHYSICAL units (same convention as
extract_features.py's modelA_inputs) -- this is what any probe trained on
train_features.npz/val_features.npz (e.g. predictability.py's q_A)
must be called with, since those were trained on raw, un-normalized
quantities. Conflating the two is an easy trap: feeding normalized,
O(1)-scale values into a probe whose own internal standardization was fit
assuming raw, O(1e4)-scale inputs silently produces catastrophic,
worse-than-zero errors that look like a real feedback-divergence finding
but are actually just a units mismatch -- worth keeping in mind for anyone
extending this module with a new intervention_fn.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "evaluation"))

from shared_eval_utils import (
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
)
from eval_model_d import integrate_growth_matched


# ---------------------------------------------------------------------------
# Intervention factories: each defines what value is fed into the NODE in
# place of the true CNN growth feature h_true, letting the rollout above
# stay unchanged while only this one substitution varies. Used by
# intervention_sweep.py (identity/zero/mean/shuffle) and
# functional_substitution.py (predicted).
# ---------------------------------------------------------------------------

def identity_intervention(sim_id, step_k, t_k, t_final, vol_k, x_base, x_base_raw, h_true):
    """Sanity-check control: pass h_true through unchanged. A rollout under
    this intervention must reproduce Model D's normal (non-intervened)
    accuracy exactly, since nothing is actually being substituted."""
    return h_true


def zero_intervention(sim_id, step_k, t_k, t_final, vol_k, x_base, x_base_raw, h_true):
    """Replace h_g with the zero vector at every step, regardless of the
    true growth state -- the harshest ablation of the growth-feedback path."""
    return torch.zeros_like(h_true)


def make_mean_intervention(h_mean: np.ndarray):
    """Replace h_g with a single fixed vector (the training-set mean h_g)
    at every step -- removes all per-simulation/per-step variation while
    keeping the feature at a "typical" scale, unlike the zero intervention."""
    h_mean_t = torch.from_numpy(np.asarray(h_mean, dtype=np.float32)).view(1, -1)

    def _fn(sim_id, step_k, t_k, t_final, vol_k, x_base, x_base_raw, h_true):
        return h_mean_t
    return _fn


def make_predicted_intervention(predict_fn):
    """Replace h_g with a value predicted online from the model's own
    current state, instead of the true CNN output. predict_fn(x_base_raw_numpy:
    (1,D), RAW physical units, matching extract_features.py's modelA_inputs
    convention) -> (1,dim_gfeat) numpy, computed live from THIS rollout's own
    current state -- no teacher forcing. Uses x_base_raw, NOT the normalized
    x_base -- the predictor (e.g. predictability.py's q_A probe) was
    trained on raw physical-unit inputs."""
    def _fn(sim_id, step_k, t_k, t_final, vol_k, x_base, x_base_raw, h_true):
        h_pred = predict_fn(x_base_raw)
        return torch.from_numpy(np.asarray(h_pred, dtype=np.float32)).view(1, -1)
    return _fn


def make_shuffle_intervention(companion_map: dict, val_features: dict, scheme: str, seed: int = 12345):
    """Replace h_g with another simulation's h_g at a matching point in its
    own trajectory, rather than at the true current step -- if the NODE
    only cares about h_g's rough scale/progress and not its
    simulation-specific content, this should hurt little; if it cares about
    the specific growth state, this should hurt a lot. Three schemes, each
    parameterized by a fixed companion_map (one random permutation
    assigning every val sim i a companion pi(i)!=i, built by the caller and
    varied across repeats).

    scheme='progress': substitute the companion's h_g from the step with
        closest matching normalized progress s=t/t_f.
    scheme='volume': substitute from the step with closest matching current
        volume.
    scheme='random': substitute from a uniformly random step of the
        companion, ignoring s/volume matching entirely (harsher control).

    val_features: the loaded val_features.npz dict (needs sim_id,
    snapshot_id, time, volume, cnn_final_features).
    """
    sid_arr = val_features["sim_id"]
    time_arr = val_features["time"]
    vol_arr = val_features["volume"]
    feat_arr = val_features["cnn_final_features"]

    per_sim = {}
    for sid in np.unique(sid_arr):
        mask = sid_arr == sid
        t_this = time_arr[mask]
        per_sim[int(sid)] = dict(time=t_this, t_final=float(t_this.max()), vol=vol_arr[mask], feat=feat_arr[mask])

    rng = np.random.default_rng(seed)

    def _fn(sim_id, step_k, t_k, t_final, vol_k, x_base, x_base_raw, h_true):
        companion = companion_map[int(sim_id)]
        comp = per_sim[companion]
        n = comp["feat"].shape[0]
        if scheme == "random":
            j = int(rng.integers(0, n))
        elif scheme == "progress":
            s_query = t_k / t_final if t_final > 0 else 0.0
            s_comp = comp["time"] / (comp["t_final"] + 1e-12)
            j = int(np.argmin(np.abs(s_comp - s_query)))
        elif scheme == "volume":
            j = int(np.argmin(np.abs(comp["vol"] - vol_k)))
        else:
            raise ValueError(f"unknown scheme {scheme!r}")
        return comp["feat"][j:j + 1].astype(np.float32)
    return _fn


# ---------------------------------------------------------------------------
# The intervention-capable rollout itself
# ---------------------------------------------------------------------------

def build_constrained_companion_map(val_features: dict, seed: int, higher_final_volume_only: bool = True) -> dict:
    """Companion-selection variant for the volume-matched shuffle intervention
    (see make_shuffle_intervention's scheme='volume'): a plain
    random-derangement companion_map can pair a query sim with a companion
    whose own final volume is much LOWER than the query's, forcing the
    nearest-volume snapshot lookup to clip to the companion's own endpoint
    for the rest of that trajectory once the query's volume exceeds it
    (verified empirically: this affected 27.9% of (sim,step) pairs under
    plain random pairing, median mismatch ~132,000 volume units).

    This restricts each sim's companion pool to only OTHER sims whose final
    (max) volume is >= the query's own final volume -- guaranteeing (except
    for the single highest-final-volume sim in the set, which has no
    eligible companion and falls back to the next-highest) that the
    companion's own trajectory spans a volume range covering the query's,
    so nearest-neighbor volume matching no longer needs to extrapolate.
    Verified empirically to cut the clipping rate from 27.9% to 0.02%.
    """
    sid_arr = val_features["sim_id"]
    vol_arr = val_features["volume"]
    uniq = np.unique(sid_arr)
    final_vol = {int(s): float(vol_arr[sid_arr == s].max()) for s in uniq}

    rng = np.random.default_rng(seed)
    companion_map = {}
    for sid in uniq:
        sid = int(sid)
        if higher_final_volume_only:
            pool = [int(s) for s in uniq if final_vol[int(s)] >= final_vol[sid] and int(s) != sid]
        else:
            pool = [int(s) for s in uniq if int(s) != sid]
        if len(pool) == 0:
            others = [int(s) for s in uniq if int(s) != sid]
            pool = [max(others, key=lambda s: final_vol[s])]
        companion_map[sid] = int(rng.choice(pool))
    return companion_map


def rollout_cnn_node_intervened(
    r_modes: int,
    sim_id: int,
    nodes_csv: str,
    elems_csv: str,
    intervention_fn,
    max_steps=None,
    ckpt_path_template: str = "checkpoints/Model_D_v2_BEST.pt",
    make_plots: bool = False,
):
    """Byte-for-byte identical to rollout_single_sim_cnn_node_matched except
    for the one intervention point documented at module level."""
    device = torch.device("cpu")
    ckpt_path = ckpt_path_template.format(r=r_modes, r_modes=r_modes)
    ckpt = torch.load(ckpt_path, map_location=device)

    D_state = int(ckpt["latent_state_dim"])
    volume_idx = int(r_modes)

    z_mean = np.asarray(ckpt["z_mean"], dtype=np.float32).reshape(-1)
    z_std = np.asarray(ckpt["z_std"], dtype=np.float32).reshape(-1)
    sp_mean = float(np.asarray(ckpt["sp_mean"], dtype=np.float32).reshape(-1)[0])
    sp_std = float(np.asarray(ckpt["sp_std"], dtype=np.float32).reshape(-1)[0])
    design_mean = np.asarray(ckpt["design_mean"], dtype=np.float32).reshape(1, -1)
    design_std = np.asarray(ckpt["design_std"], dtype=np.float32).reshape(1, -1)
    e_mean = float(np.asarray(ckpt["e_mean"], dtype=np.float32).reshape(-1)[0])
    e_std = float(np.asarray(ckpt["e_std"], dtype=np.float32).reshape(-1)[0])
    I_mean = float(np.asarray(ckpt["I_mean"], dtype=np.float32).reshape(-1)[0])
    I_std = float(np.asarray(ckpt["I_std"], dtype=np.float32).reshape(-1)[0])
    g_mean = np.asarray(ckpt["g_mean"], dtype=np.float32).reshape(-1)
    g_std = np.asarray(ckpt["g_std"], dtype=np.float32).reshape(-1)

    model = build_cnn_node_from_ckpt(ckpt, device)
    model.eval()

    (U_lat, volume_snap, sim_index, time_vals, volume_SP, design_all, n_sims) = load_raw_data(r_modes)
    Z_state_all = np.concatenate([U_lat, volume_snap.reshape(-1, 1)], axis=1).astype(np.float32)

    idx = np.where(sim_index == sim_id)[0]
    t_s = time_vals[idx]
    order = np.argsort(t_s)
    idx_sorted = idx[order]
    t_sim = t_s[order]

    Z_true_sim = Z_state_all[idx_sorted, :]
    SP_sim = volume_SP[idx_sorted]
    design_sim = design_all[idx_sorted, :]

    T_true = Z_true_sim.shape[0]
    n_steps = (T_true - 1) if max_steps is None else min(max_steps, T_true - 1)
    t_final = float(t_sim[-1])

    design_raw = design_sim[0:1, :].astype(np.float32)
    design_norm = (design_raw - design_mean) / design_std
    design_tensor = torch.from_numpy(design_norm.astype(np.float32)).to(device)

    node_xyz_bot, elem_conn_bot, node_ids_bot, elem_node_ids_bot = load_bottom_surface_mesh_direct(nodes_csv, elems_csv)
    H, W, _ = get_bottom_grid_cache()

    decode_u = load_disp_decoder(
        r_modes, zarr_disp_path=ZARR_DISP, pod_group=DISP_POD_GROUP,
        pod_u_name=DISP_POD_U_NAME, pod_mean_name=DISP_POD_MEAN_NAME, verbose=False,
    )

    job_id, theta_crit, k1, k2 = get_growth_params_for_sim(sim_id)

    t_sim_g, Gx_sim_elem, Gy_sim_elem, _ = load_elem_growth_for_sim(
        sim_id=sim_id, sim_index=sim_index, time_vals=time_vals,
        zarr_sdv_path=ZARR_SDV, folder_x="snaps_SDV1", folder_y="snaps_SDV4",
    )
    Ne_full = Gx_sim_elem.shape[1]
    half = Ne_full // 2
    gx0 = Gx_sim_elem[0, half:]
    gy0 = Gy_sim_elem[0, half:]
    lamdag_elem = np.stack([gx0, gy0], axis=1).astype(np.float64)

    G_grid = np.zeros((1, 2, H, W), dtype=np.float32)
    G_grid[0, 0] = lamdag_elem[:, 0].reshape(H, W)
    G_grid[0, 1] = lamdag_elem[:, 1].reshape(H, W)
    G_grid = normalize_growth_grid(G_grid, g_mean, g_std)

    Z0_raw = Z_true_sim[0, :].astype(np.float32)
    V0_raw = float(Z0_raw[volume_idx])
    SP0_raw = float(SP_sim[0])
    e_raw = SP0_raw - V0_raw
    I_raw = 0.0

    z0_norm = ((Z0_raw - z_mean) / z_std).astype(np.float32)[None, :]
    z_curr = torch.from_numpy(z0_norm.astype(np.float32)).to(device)

    t_list = [float(t_sim[0])]
    z_true_list = [Z0_raw.copy()]
    z_pred_list = [Z0_raw.copy()]
    vol_true_list = [V0_raw]
    vol_pred_list = [V0_raw]
    e_hist = [e_raw]
    I_hist = [I_raw]
    u_pred_nodes = []
    g_pred_grid = []
    lamdag_pred_elem = []

    u0 = decode_u(Z0_raw[:r_modes])
    u_pred_nodes.append(u0.astype(np.float32))
    g_pred_grid.append(G_grid[0].copy())
    lamdag_pred_elem.append(lamdag_elem.copy())

    with torch.no_grad():
        for k in range(n_steps):
            t_k = float(t_sim[k])
            t_k1 = float(t_sim[k + 1])
            dt_k = float(t_k1 - t_k)

            SP_k_raw = float(SP_sim[k])
            e_norm = (e_raw - e_mean) / e_std
            I_norm = (I_raw - I_mean) / I_std
            sp_norm = (SP_k_raw - sp_mean) / sp_std

            e_tensor = torch.tensor([[e_norm]], dtype=torch.float32, device=device)
            I_tensor = torch.tensor([[I_norm]], dtype=torch.float32, device=device)
            sp_tensor = torch.tensor([[sp_norm]], dtype=torch.float32, device=device)

            x_base = torch.cat([z_curr, e_tensor, I_tensor, sp_tensor, design_tensor], dim=1)
            G_tensor = torch.from_numpy(G_grid.astype(np.float32)).to(device)

            # raw-physical-units version of x_base, matching extract_features.py's
            # modelA_inputs convention exactly -- [z(10 raw), e, I, sp, design(7 raw)]
            z_curr_raw = (z_curr.detach().cpu().numpy()[0] * z_std + z_mean).astype(np.float32)
            x_base_raw = np.concatenate([
                z_curr_raw, np.array([e_raw, I_raw, SP_k_raw], dtype=np.float32), design_raw.reshape(-1),
            ]).reshape(1, -1)

            # --- the only functional change vs. rollout_single_sim_cnn_node_matched ---
            vol_k_current = float(vol_pred_list[-1])
            h_true = model.growth_encoder(G_tensor)
            h_used = intervention_fn(sim_id, k, t_k, t_final, vol_k_current, x_base, x_base_raw, h_true)
            if not torch.is_tensor(h_used):
                h_used = torch.from_numpy(np.asarray(h_used, dtype=np.float32))
            h_used = h_used.view(1, -1)
            x_full = torch.cat([x_base, h_used], dim=-1)
            dz_norm = model.net(x_full)
            # ---------------------------------------------------------------

            z_next = z_curr + dt_k * dz_norm
            z_next_np = z_next.detach().cpu().numpy()[0]

            Z_next_raw = (z_next_np * z_std + z_mean).astype(np.float32)
            V_next_raw = float(Z_next_raw[volume_idx])
            V_true_raw = float(Z_true_sim[k + 1, volume_idx])

            z_pred_list.append(Z_next_raw.copy())
            z_true_list.append(Z_true_sim[k + 1, :].astype(np.float32).copy())
            t_list.append(t_k1)
            vol_pred_list.append(V_next_raw)
            vol_true_list.append(V_true_raw)

            u_nodes = decode_u(Z_next_raw[:r_modes])

            lamdag_elem = integrate_growth_matched(
                node_coords=node_xyz_bot, elem_conn=elem_conn_bot, node_u=u_nodes,
                elem_lamdag=lamdag_elem, k1=k1, k2=k2, theta_crit=theta_crit, time=t_k, dt=dt_k,
            )

            G_grid = np.zeros((1, 2, H, W), dtype=np.float32)
            G_grid[0, 0] = lamdag_elem[:, 0].reshape(H, W)
            G_grid[0, 1] = lamdag_elem[:, 1].reshape(H, W)
            G_grid = normalize_growth_grid(G_grid, g_mean, g_std)

            u_pred_nodes.append(u_nodes.astype(np.float32))
            g_pred_grid.append(G_grid[0].copy())
            lamdag_pred_elem.append(lamdag_elem.copy())

            I_raw = I_raw + dt_k * e_raw
            SP_next_raw = float(SP_sim[k + 1])
            e_raw = SP_next_raw - V_next_raw
            e_hist.append(e_raw)
            I_hist.append(I_raw)

            z_curr = z_next

    z_true_raw = np.vstack(z_true_list)
    z_pred_raw = np.vstack(z_pred_list)
    t_arr = np.array(t_list, dtype=np.float64)
    vol_true = np.array(vol_true_list, dtype=np.float64)
    vol_pred = np.array(vol_pred_list, dtype=np.float64)
    err_L2 = np.linalg.norm(z_pred_raw - z_true_raw, axis=1)
    err_vol = np.abs(vol_pred - vol_true)
    area_gain_pred_final = compute_net_area_gain_from_lamdag_elem(lamdag_pred_elem[-1], A0_cm2=A0_CM2)

    return {
        "t": t_arr, "z_true": z_true_raw, "z_pred": z_pred_raw,
        "vol_true": vol_true, "vol_pred": vol_pred, "err_L2": err_L2, "err_vol": err_vol,
        "e_hist": np.array(e_hist, dtype=np.float64), "I_hist": np.array(I_hist, dtype=np.float64),
        "u_pred_nodes": np.stack(u_pred_nodes, axis=0),
        "g_pred_grid": np.stack(g_pred_grid, axis=0),
        "lamdag_pred_elem": np.stack(lamdag_pred_elem, axis=0),
        "r_modes": r_modes, "sim_id": sim_id, "volume_idx": int(volume_idx), "volume_dim": int(volume_idx),
        "state_names": [*(f"u_lat_{i}" for i in range(r_modes)), "volume"],
        "H": int(H), "W": int(W),
        "area_gain_pred_final_cm2": float(area_gain_pred_final),
        "n_steps": int(n_steps),
    }
