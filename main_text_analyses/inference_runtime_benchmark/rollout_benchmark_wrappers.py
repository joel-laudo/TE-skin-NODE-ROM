"""Thin "pre-loaded model" rollout wrappers used by benchmark_runtime.py to
time Model A vs. Model D inference (Section 4.1) without the checkpoint
load/model-build cost leaking into the timed region.

Model A's `rollout_single_sim_vanilla_node` (eval_ablation_model_a_base_closureMLP.py)
and Model D's `rollout_single_sim_cnn_node_matched` (eval_rollout_cnn.py) both
call `torch.load(ckpt_path)` + rebuild the model INSIDE every single call --
fine for one-off evaluation, but wrong for a "load the checkpoint once, then
time each simulation individually" benchmark.

Each function below is a line-for-line duplicate of its authoritative
counterpart with the checkpoint-load + model-build lines removed
(the caller now passes in an already-loaded `ckpt` dict, an already-built
`model`, and `device`). Diagnostic-plotting branches (`make_plots`) and
per-step print statements are omitted since they are never exercised in a
batch/benchmark context (the authoritative functions are always called with
make_plots=False for batch work) and per-step prints would otherwise leak
I/O time into the timed region.

ONE further, deliberate deviation: rollout_A_preloaded calls the SAME
UMAT-matched implicit-Newton growth integrator (ed.integrate_growth_matched)
that Model D uses, instead of the older explicit ma.integrate_growth_fast
that the authoritative rollout_single_sim_vanilla_node actually calls. This
is intentional -- without it, an A-vs-D runtime comparison would conflate
two different things: the cost of Model D's CNN pathway, and a difference in
growth-integrator implementation that predates and is unrelated to this
benchmark. It's safe to do because Model A's own NODE dynamics never consume
growth feedback (lamdag_elem is diagnostic-only, computed but never fed back
into x) -- so u_pred_nodes/z_pred/vol_pred stay bit-exact vs. the
authoritative function regardless of which growth integrator runs, while
g_pred_grid/lamdag_pred_elem/area_gain_pred_final_cm2 legitimately differ (a
known, intended consequence, not a bug). rollout_D_preloaded has no such
deviation -- it matches rollout_single_sim_cnn_node_matched exactly in every
respect.

Bit-exact reproduction (on the fields each model's own dynamics actually
determine) against the authoritative functions is verified by
verify_bitexact.py before any benchmark timing is trusted.

FURTHER MEMORY/IO deviation: both wrappers take a required `growth_cache`
argument (built once by `prefetch_growth_data`, before any timing starts)
instead of calling the authoritative functions' own
get_growth_params_for_sim/load_elem_growth_for_sim internally. This removes
two things from the timed per-simulation call: (1) a genuine, uncached-per-
new-sim zarr+text-file read (get_growth_params_for_sim's cache is keyed by
sim_id, so every one of the 185 distinct validation sims was a fresh disk
read during timing, not a cache hit), and (2) load_elem_growth_for_sim's
~2.3 GB (both arrays) eager whole-store load into a module-global cache --
prefetch_growth_data reads only the one small column each sim actually
needs. Net effect: zero zarr/disk access happens anywhere inside the timed
loop, and peak memory drops by roughly the size of that eager cache.
"""
import numpy as np
import torch
import zarr

import eval_ablation_model_a_base_closureMLP as ma
import eval_rollout_cnn as ed


def prefetch_growth_data(val_sims, sim_index, time_vals,
                          zarr_sdv_path="ip_growth_elem.zarr",
                          folder_x="snaps_SDV1", folder_y="snaps_SDV4"):
    """Pre-fetch, for every val sim, the two per-sim quantities the rollout
    wrappers need from disk: growth-kinetics params (theta_crit, k1, k2) and
    the t=0 growth initial condition (gx0, gy0). Called ONCE, up front,
    before any timing starts -- the timed rollout calls then do a plain dict
    lookup, touching neither zarr nor any other file.

    This deliberately does NOT use the project's existing
    load_elem_growth_for_sim, which eagerly loads and module-caches the
    ENTIRE snaps_SDV1/snaps_SDV4 arrays (shape (7200, 42237), ~1.16 GB each
    decompressed) just to read one sim's first column -- a huge, unnecessary
    memory cost for what is a single 7200-element slice per sim. Instead,
    this opens the zarr arrays directly and reads only the columns actually
    needed. Confirmed chunk layout is (7200, 16) (one chunk = all elements x
    16 consecutive snapshots, ~450 KB decompressed).

    All ~185 needed columns are gathered into ONE batched orthogonal-index
    read per array (`.oindex[sorted_cols, :]`), not 185 separate individual
    reads -- zarr resolves the distinct chunks a batched fancy-index touches
    in a single call, which is both fewer total bytes decompressed than the
    old eager-whole-array approach AND far fewer individual
    open/seek/decompress round-trips than issuing one zarr read per
    simulation would be. Column indices are sorted first so chunk access is
    monotonic. This whole function runs once, before any timing starts, so
    even its own cost is never part of any reported per-simulation time --
    but batching still matters for keeping total wall-clock job time down.
    """
    g_sdv = zarr.open_group(zarr_sdv_path, mode="r")
    S = sim_index.shape[0]

    def _open_snap_array(folder_name):
        candidates = [f"{folder_name}/{folder_name}", f"{folder_name}/data",
                      f"{folder_name}/snaps", f"{folder_name}"]
        last_err = None
        for key in candidates:
            try:
                return g_sdv[key]
            except Exception as e:
                last_err = e
        raise KeyError(f"Could not open '{folder_name}'. Tried={candidates}. Last error={repr(last_err)}")

    gx_arr = _open_snap_array(folder_x)
    gy_arr = _open_snap_array(folder_y)
    # Orientation check mirrors load_elem_growth_for_sim's own logic, using
    # only .shape (metadata, no data read) to decide which axis is "element".
    gx_snapshot_major = gx_arr.shape[0] == S
    gy_snapshot_major = gy_arr.shape[0] == S

    sim_ids = [int(s) for s in val_sims]
    col0_by_sim = {}
    for sid in sim_ids:
        idx = np.where(sim_index == sid)[0]
        if idx.size < 2:
            raise ValueError(f"Sim {sid} has too few frames ({idx.size}).")
        t_s = time_vals[idx]
        col0_by_sim[sid] = int(idx[np.argsort(t_s)][0])  # first (t=0) snapshot's global column

    cols = np.array([col0_by_sim[sid] for sid in sim_ids])
    order = np.argsort(cols)
    sorted_cols = cols[order]
    sorted_sim_ids = [sim_ids[i] for i in order]

    gx_batch = np.asarray(
        gx_arr.oindex[sorted_cols, :] if gx_snapshot_major else gx_arr.oindex[:, sorted_cols].T,
        dtype=np.float64,
    )  # (n_sims, Ne_full), one batched read
    gy_batch = np.asarray(
        gy_arr.oindex[sorted_cols, :] if gy_snapshot_major else gy_arr.oindex[:, sorted_cols].T,
        dtype=np.float64,
    )

    Ne_full = gx_batch.shape[1]
    if Ne_full % 2 != 0:
        raise ValueError(f"Expected even Ne from growth arrays, got Ne={Ne_full}")
    half = Ne_full // 2

    cache = {}
    for i, sid in enumerate(sorted_sim_ids):
        job_id, theta_crit, k1, k2 = ma.get_growth_params_for_sim(sid)
        cache[sid] = dict(theta_crit=theta_crit, k1=k1, k2=k2,
                           gx0=gx_batch[i, half:].copy(), gy0=gy_batch[i, half:].copy())
    return cache


def rollout_A_preloaded(model, ckpt, device, r_modes, sim_id, nodes_csv, elems_csv, growth_cache, max_steps=None):
    """Duplicate of ma.rollout_single_sim_vanilla_node, minus the ckpt/model load."""
    D_state = int(ckpt["latent_state_dim"])

    volume_idx = int(r_modes)
    if D_state <= volume_idx:
        raise ValueError(f"D_state={D_state} too small for volume_idx=r_modes={r_modes}")

    z_mean = np.asarray(ckpt["z_mean"], dtype=np.float32).reshape(-1)
    z_std = np.asarray(ckpt["z_std"], dtype=np.float32).reshape(-1)
    if z_mean.shape[0] != D_state or z_std.shape[0] != D_state:
        raise ValueError(
            f"Checkpoint z_mean/z_std length mismatch with D_state={D_state}. "
            f"Got z_mean={z_mean.shape}, z_std={z_std.shape}"
        )

    sp_mean = float(np.asarray(ckpt["sp_mean"], dtype=np.float32).reshape(-1)[0])
    sp_std = float(np.asarray(ckpt["sp_std"], dtype=np.float32).reshape(-1)[0])

    design_mean = np.asarray(ckpt["design_mean"], dtype=np.float32).reshape(1, -1)
    design_std = np.asarray(ckpt["design_std"], dtype=np.float32).reshape(1, -1)

    e_mean = float(np.asarray(ckpt["e_mean"], dtype=np.float32).reshape(-1)[0])
    e_std = float(np.asarray(ckpt["e_std"], dtype=np.float32).reshape(-1)[0])

    I_mean = float(np.asarray(ckpt["I_mean"], dtype=np.float32).reshape(-1)[0])
    I_std = float(np.asarray(ckpt["I_std"], dtype=np.float32).reshape(-1)[0])

    dim_closure_ckpt = int(ckpt.get("dim_closure", 0))
    if dim_closure_ckpt > 0:
        closure_mlp, closure_z_mean_c, closure_z_std_c = ma.load_closure_mlp(ma.CLOSURE_MODEL_PATH, device=device)
    else:
        closure_mlp, closure_z_mean_c, closure_z_std_c = None, None, None

    z_mean_t = torch.from_numpy(z_mean).to(device).view(1, -1)
    z_std_t = torch.from_numpy(z_std).to(device).view(1, -1)

    (U_lat, volume_snap, sim_index, time_vals, volume_SP, design_all, n_sims) = ma.load_raw_data(r_modes)

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
    n_steps = (T_true - 1) if max_steps is None else min(max_steps, T_true - 1)

    design_raw = design_sim[0:1, :].astype(np.float32)
    design_norm = (design_raw - design_mean) / design_std
    design_tensor = torch.from_numpy(design_norm.astype(np.float32)).to(device)

    node_xyz_bot, elem_conn_bot, node_ids_bot, elem_node_ids_bot = ma.load_bottom_surface_mesh_direct(nodes_csv, elems_csv)

    H, W, _ = ma.get_bottom_grid_cache()
    if H * W != elem_conn_bot.shape[0]:
        raise ValueError(f"Expected bottom elems = H*W={H*W}, got {elem_conn_bot.shape[0]}")

    decode_u = ma.load_disp_decoder(
        r_modes,
        zarr_disp_path=ma.ZARR_DISP,
        pod_group=ma.DISP_POD_GROUP,
        pod_u_name=ma.DISP_POD_U_NAME,
        pod_mean_name=ma.DISP_POD_MEAN_NAME,
    )

    gc = growth_cache[sim_id]
    theta_crit, k1, k2 = gc["theta_crit"], gc["k1"], gc["k2"]
    gx0, gy0 = gc["gx0"], gc["gy0"]
    if gx0.shape[0] != H * W:
        raise ValueError(f"Expected bottom growth length H*W={H*W}, got {gx0.shape[0]}")

    lamdag_elem = np.stack([gx0, gy0], axis=1).astype(np.float64)

    g_pred_grid = []
    G0_raw = np.zeros((2, H, W), dtype=np.float32)
    G0_raw[0] = lamdag_elem[:, 0].reshape(H, W)
    G0_raw[1] = lamdag_elem[:, 1].reshape(H, W)
    g_pred_grid.append(G0_raw.copy())

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
    lamdag_pred_elem = []

    u0 = decode_u(Z0_raw[:r_modes])
    if u0.shape[0] != node_xyz_bot.shape[0]:
        raise ValueError(f"Decode nodes mismatch: u0={u0.shape}, node_xyz_bot={node_xyz_bot.shape}")
    u_pred_nodes.append(u0.astype(np.float32))
    lamdag_pred_elem.append(lamdag_elem.copy())

    with torch.no_grad():
        for k in range(n_steps):
            t_k = float(t_sim[k])
            t_k1 = float(t_sim[k + 1])
            dt_k = float(t_k1 - t_k)
            if (not np.isfinite(dt_k)) or (dt_k <= 0.0):
                raise ValueError(f"Bad dt at step {k}: dt_k={dt_k} (t_k={t_k}, t_k1={t_k1})")

            SP_k_raw = float(SP_sim[k])
            e_norm = (e_raw - e_mean) / e_std
            I_norm = (I_raw - I_mean) / I_std
            sp_norm = (SP_k_raw - sp_mean) / sp_std

            e_tensor = torch.tensor([[e_norm]], dtype=torch.float32, device=device)
            I_tensor = torch.tensor([[I_norm]], dtype=torch.float32, device=device)
            sp_tensor = torch.tensor([[sp_norm]], dtype=torch.float32, device=device)

            if closure_mlp is not None:
                Z_raw_curr_t = z_curr * z_std_t + z_mean_t
                alpha_raw_curr_t = Z_raw_curr_t[:, :r_modes]
                closure_feat = ma.compute_closure_feature(closure_mlp, alpha_raw_curr_t, closure_z_mean_c, closure_z_std_c)
                x = torch.cat([z_curr, e_tensor, I_tensor, sp_tensor, design_tensor, closure_feat], dim=1)
            else:
                x = torch.cat([z_curr, e_tensor, I_tensor, sp_tensor, design_tensor], dim=1)

            dz_norm = model(x)

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

            # DELIBERATE DEVIATION from the authoritative rollout_single_sim_vanilla_node,
            # which calls ma.integrate_growth_fast (the older explicit-Euler,
            # UMAT-mismatched integrator) here. This benchmark instead uses
            # the SAME UMAT-matched implicit-Newton integrator as Model D
            # (ed.integrate_growth_matched), so the A-vs-D runtime comparison
            # isolates the CNN pathway's cost rather than conflating it with
            # a difference in growth-integrator implementation. This is safe
            # for displacement accuracy: Model A's dynamics never consume
            # growth feedback (lamdag_elem is diagnostic-only, computed but
            # never fed into x), so u_pred_nodes/z_pred/vol_pred are
            # unaffected and remain bit-exact vs. the authoritative function.
            # g_pred_grid/lamdag_pred_elem/area_gain_pred_final_cm2 WILL
            # differ from the authoritative function's output as a direct,
            # intended consequence -- verify_bitexact.py checks only the
            # dynamics-relevant fields for Model A accordingly.
            lamdag_elem = ed.integrate_growth_matched(
                node_coords=node_xyz_bot, elem_conn=elem_conn_bot, node_u=u_nodes,
                elem_lamdag=lamdag_elem, k1=k1, k2=k2, theta_crit=theta_crit, time=t_k, dt=dt_k,
            )

            G_raw = np.zeros((2, H, W), dtype=np.float32)
            G_raw[0] = lamdag_elem[:, 0].reshape(H, W)
            G_raw[1] = lamdag_elem[:, 1].reshape(H, W)

            u_pred_nodes.append(u_nodes.astype(np.float32))
            g_pred_grid.append(G_raw.copy())
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

    area_gain_pred_final = ma.compute_net_area_gain_from_lamdag_elem(lamdag_pred_elem[-1], A0_cm2=0.25)

    rollout = {
        "t": t_arr, "z_true": z_true_raw, "z_pred": z_pred_raw,
        "vol_true": vol_true, "vol_pred": vol_pred,
        "err_L2": err_L2, "err_vol": err_vol,
        "e_hist": np.array(e_hist, dtype=np.float64), "I_hist": np.array(I_hist, dtype=np.float64),
        "u_pred_nodes": np.stack(u_pred_nodes, axis=0),
        "g_pred_grid": np.stack(g_pred_grid, axis=0),
        "lamdag_pred_elem": np.stack(lamdag_pred_elem, axis=0),
        "r_modes": r_modes, "sim_id": sim_id,
        "volume_idx": int(volume_idx), "volume_dim": int(volume_idx),
        "state_names": [*(f"u_lat_{i}" for i in range(r_modes)), "volume"],
        "H": int(H), "W": int(W),
        "area_gain_pred_final_cm2": float(area_gain_pred_final),
        "n_steps": int(n_steps),
    }
    return rollout


def rollout_D_preloaded(model, ckpt, device, r_modes, sim_id, nodes_csv, elems_csv, growth_cache, max_steps=None):
    """Duplicate of ed.rollout_single_sim_cnn_node_matched, minus the ckpt/model load."""
    D_state = int(ckpt["latent_state_dim"])

    volume_idx = int(r_modes)
    if D_state <= volume_idx:
        raise ValueError(f"D_state={D_state} too small for volume_idx=r_modes={r_modes}")

    z_mean = np.asarray(ckpt["z_mean"], dtype=np.float32).reshape(-1)
    z_std = np.asarray(ckpt["z_std"], dtype=np.float32).reshape(-1)
    if z_mean.shape[0] != D_state or z_std.shape[0] != D_state:
        raise ValueError(f"z_mean/z_std length mismatch with D_state={D_state}: {z_mean.shape}, {z_std.shape}")

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
    if g_mean.shape[0] != 2 or g_std.shape[0] != 2:
        raise ValueError(f"Expected g_mean/g_std shape (2,), got {g_mean.shape}, {g_std.shape}")

    (U_lat, volume_snap, sim_index, time_vals, volume_SP, design_all, n_sims) = ed.load_raw_data(r_modes)

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
    n_steps = (T_true - 1) if max_steps is None else min(max_steps, T_true - 1)

    design_raw = design_sim[0:1, :].astype(np.float32)
    design_norm = (design_raw - design_mean) / design_std
    design_tensor = torch.from_numpy(design_norm.astype(np.float32)).to(device)

    node_xyz_bot, elem_conn_bot, node_ids_bot, elem_node_ids_bot = ed.load_bottom_surface_mesh_direct(nodes_csv, elems_csv)

    H, W, _ = ed.get_bottom_grid_cache()
    if H * W != elem_conn_bot.shape[0]:
        raise ValueError(f"Expected bottom elems = H*W={H*W}, got {elem_conn_bot.shape[0]}")

    decode_u = ed.load_disp_decoder(
        r_modes, zarr_disp_path=ed.ZARR_DISP, pod_group=ed.DISP_POD_GROUP,
        pod_u_name=ed.DISP_POD_U_NAME, pod_mean_name=ed.DISP_POD_MEAN_NAME, verbose=False,
    )

    gc = growth_cache[sim_id]
    theta_crit, k1, k2 = gc["theta_crit"], gc["k1"], gc["k2"]
    gx0, gy0 = gc["gx0"], gc["gy0"]
    if gx0.shape[0] != H * W:
        raise ValueError(f"Expected bottom growth length H*W={H*W}, got {gx0.shape[0]}")

    lamdag_elem = np.stack([gx0, gy0], axis=1).astype(np.float64)

    G_grid = np.zeros((1, 2, H, W), dtype=np.float32)
    G_grid[0, 0] = lamdag_elem[:, 0].reshape(H, W)
    G_grid[0, 1] = lamdag_elem[:, 1].reshape(H, W)
    G_grid = ed.normalize_growth_grid(G_grid, g_mean, g_std)

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
    if u0.shape[0] != node_xyz_bot.shape[0]:
        raise ValueError(f"Decode nodes mismatch: u0={u0.shape}, node_xyz_bot={node_xyz_bot.shape}")
    u_pred_nodes.append(u0.astype(np.float32))
    g_pred_grid.append(G_grid[0].copy())
    lamdag_pred_elem.append(lamdag_elem.copy())

    with torch.no_grad():
        for k in range(n_steps):
            t_k = float(t_sim[k])
            t_k1 = float(t_sim[k + 1])
            dt_k = float(t_k1 - t_k)
            if (not np.isfinite(dt_k)) or (dt_k <= 0.0):
                raise ValueError(f"Bad dt at step {k}: dt_k={dt_k} (t_k={t_k}, t_k1={t_k1})")

            SP_k_raw = float(SP_sim[k])
            e_norm = (e_raw - e_mean) / e_std
            I_norm = (I_raw - I_mean) / I_std
            sp_norm = (SP_k_raw - sp_mean) / sp_std

            e_tensor = torch.tensor([[e_norm]], dtype=torch.float32, device=device)
            I_tensor = torch.tensor([[I_norm]], dtype=torch.float32, device=device)
            sp_tensor = torch.tensor([[sp_norm]], dtype=torch.float32, device=device)

            x_base = torch.cat([z_curr, e_tensor, I_tensor, sp_tensor, design_tensor], dim=1)
            G_tensor = torch.from_numpy(G_grid.astype(np.float32)).to(device)

            dz_norm = model(x_base, G_tensor)

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
            if u_nodes.shape[0] != node_xyz_bot.shape[0]:
                raise ValueError(f"u_nodes mismatch at step {k}: {u_nodes.shape} vs {node_xyz_bot.shape}")
            if elem_conn_bot.max() >= u_nodes.shape[0]:
                raise ValueError(f"elem_conn_bot out of range at step {k}")

            lamdag_elem = ed.integrate_growth_matched(
                node_coords=node_xyz_bot, elem_conn=elem_conn_bot, node_u=u_nodes,
                elem_lamdag=lamdag_elem, k1=k1, k2=k2, theta_crit=theta_crit, time=t_k, dt=dt_k,
            )

            G_grid = np.zeros((1, 2, H, W), dtype=np.float32)
            G_grid[0, 0] = lamdag_elem[:, 0].reshape(H, W)
            G_grid[0, 1] = lamdag_elem[:, 1].reshape(H, W)
            G_grid = ed.normalize_growth_grid(G_grid, g_mean, g_std)

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

    area_gain_pred_final = ed.compute_net_area_gain_from_lamdag_elem(lamdag_pred_elem[-1], A0_cm2=ed.A0_CM2)

    rollout = {
        "t": t_arr, "z_true": z_true_raw, "z_pred": z_pred_raw,
        "vol_true": vol_true, "vol_pred": vol_pred,
        "err_L2": err_L2, "err_vol": err_vol,
        "e_hist": np.array(e_hist, dtype=np.float64), "I_hist": np.array(I_hist, dtype=np.float64),
        "u_pred_nodes": np.stack(u_pred_nodes, axis=0),
        "g_pred_grid": np.stack(g_pred_grid, axis=0),
        "lamdag_pred_elem": np.stack(lamdag_pred_elem, axis=0),
        "r_modes": r_modes, "sim_id": sim_id,
        "volume_idx": int(volume_idx), "volume_dim": int(volume_idx),
        "state_names": [*(f"u_lat_{i}" for i in range(r_modes)), "volume"],
        "H": int(H), "W": int(W),
        "area_gain_pred_final_cm2": float(area_gain_pred_final),
        "n_steps": int(n_steps),
    }
    return rollout
