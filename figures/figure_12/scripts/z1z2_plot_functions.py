"""Plotting functions for Figure 12's z1-z2 latent phase-portrait panels
(Models A and D), imported and called by build_figure12_panels.py. Each
plot freezes the model's learned vector field (dz/dt as a function of the
first two latent coordinates z1, z2, with all other model inputs -- growth
history, design parameters, etc. -- held fixed at one rollout step) and
overlays that simulation's actual predicted/true trajectory through that
slice, for one selected simulation and one truncation step.

Model A (vanilla NODE) and Model D (CNN-growth-conditioned NODE) need
different plotting functions below because Model D's vector field also
depends on a spatial growth field tensor G, which must be threaded through
every forward pass alongside the usual scalar/latent inputs.

Checkpoints are loaded via `evaluation/eval_model_a.py`,
`evaluation/eval_model_d.py`, and `evaluation/shared_eval_utils.py`
(imported below with `evaluation/` added to `sys.path` relative to this
file's own location, so this resolves regardless of the caller's working
directory).
"""
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend -- must be set before any pyplot import
import matplotlib.pyplot as plt

# figures/figure_12/scripts -> figures/figure_12 -> figures -> repository root
_EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "evaluation")
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from eval_model_a import (
    build_vanilla_node_from_ckpt,
    load_raw_data,
    rollout_single_sim_vanilla_node,
)
from eval_model_d import rollout_single_sim_cnn_node_matched as rollout_single_sim_cnn_node
from shared_eval_utils import build_cnn_node_from_ckpt

import matplotlib as mpl


def set_pub_plot_style():
    """Apply the shared publication rcParams (serif math font, tick sizes,
    line widths, etc.) used as the baseline style for every Figure 12
    z1-z2 panel. build_figure12_panels.py calls this once, then layers a
    few panel-specific font-size overrides on top."""
    mpl.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.5,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })


def plot_selected_sim_vanilla_node_z1_z2_slice(
    r_modes: int,
    sim_id: int,
    rollout: dict,
    traj_end_step: int | None = None,
    ckpt_path_template: str = "checkpoints/Model_A_v2_BEST.pt",
    mode1_idx: int = 0,
    mode2_idx: int = 1,
    grid_n: int = 31,
    z1_lim: tuple | None = None,
    z2_lim: tuple | None = None,
    pad_frac: float = 0.10,
    normalize_arrows: bool = True,
    use_streamplot: bool = False,

    # quiver styling
    quiver_color: str = "black",
    quiver_alpha: float = 0.45,
    quiver_width: float = 0.0022,
    quiver_headwidth: float = 3.0,
    quiver_headlength: float = 5.0,
    quiver_headaxislength: float = 4.5,
    quiver_linewidth: float = 0.5,

    # streamplot styling
    stream_color: str = "0.35",
    stream_linewidth: float = 1.0,
    stream_density: float = 1.1,
    stream_arrowsize: float = 1.0,

    # trajectory styling
    pred_color=None,
    pred_linewidth: float = 2.8,
    pred_linestyle: str = "-",
    true_color=None,
    true_linewidth: float = 2.0,
    true_linestyle: str = "--",
    true_alpha: float = 0.9,
    show_true_traj: bool = False,

    # markers
    start_marker: str = "o",
    start_markersize: float = 40,
    end_marker: str = "x",
    end_markersize: float = 55,
    end_marker_linewidth: float | None = None,

    # axes / figure
    figsize=(8, 6),
    title: str | None = None,
    show_grid: bool = True,
    grid_alpha: float = 0.2,

    # autoscaling knob
    arrow_length_frac: float = 2.0,

    # terminal tangent arrow
    show_terminal_tangent: bool = True,
    terminal_tangent_color: str = "red",
    terminal_tangent_width: float = 0.004,
    terminal_tangent_headwidth: float = 4.5,
    terminal_tangent_headlength: float = 6.0,
    terminal_tangent_headaxislength: float = 5.0,
    terminal_tangent_alpha: float = 1.0,
    terminal_tangent_length_frac: float = 2.2,
    show_terminal_x_only_if_last: bool = True,

    save_path: str | None = None,
    save_dpi: int = 400,
    save_bbox_inches: str = "tight",
    save_pad_inches: float = 0.02,
):
    """
    Plot a z1-z2 latent vector field slice for ONE selected simulation, for
    Model A (the vanilla NODE).

    The vector field (drawn as a streamplot or quiver arrows, depending on
    use_streamplot) is frozen using the rollout's state/auxiliary values at
    traj_end_step -- i.e. it shows what the model would predict at nearby
    (z1, z2) points if everything else about the current state were held
    fixed. The solid line is this simulation's own predicted trajectory,
    truncated to steps [0, traj_end_step]; the dashed line (if
    show_true_traj) is the corresponding ground-truth FE trajectory. "o"
    marks the trajectory start, "x" marks the endpoint (only drawn if this
    is the final rollout step, unless show_terminal_x_only_if_last=False),
    and the separate red arrow (if show_terminal_tangent) is the model's
    instantaneous velocity at the endpoint -- distinct from the background
    vector field because it also reflects the terminal e/I/sp values
    exactly, not the grid point directly under the arrow.

    Model input order is kept exactly as in rollout_single_sim_vanilla_node:
        x = [z, e, I, sp, design]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # 1) Load checkpoint + model + normalization stats
    # --------------------------------------------------
    ckpt_path = ckpt_path_template.format(r=r_modes, r_modes=r_modes)
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_vanilla_node_from_ckpt(ckpt, device)
    model.eval()

    D_state = int(ckpt["latent_state_dim"])

    z_mean = np.asarray(ckpt["z_mean"], dtype=np.float32).reshape(-1)
    z_std  = np.asarray(ckpt["z_std"],  dtype=np.float32).reshape(-1)

    sp_mean = float(np.asarray(ckpt["sp_mean"], dtype=np.float32).reshape(-1)[0])
    sp_std  = float(np.asarray(ckpt["sp_std"],  dtype=np.float32).reshape(-1)[0])

    design_mean = np.asarray(ckpt["design_mean"], dtype=np.float32).reshape(1, -1)
    design_std  = np.asarray(ckpt["design_std"],  dtype=np.float32).reshape(1, -1)

    e_mean = float(np.asarray(ckpt["e_mean"], dtype=np.float32).reshape(-1)[0])
    e_std  = float(np.asarray(ckpt["e_std"],  dtype=np.float32).reshape(-1)[0])

    I_mean = float(np.asarray(ckpt["I_mean"], dtype=np.float32).reshape(-1)[0])
    I_std  = float(np.asarray(ckpt["I_std"],  dtype=np.float32).reshape(-1)[0])

    # --------------------------------------------------
    # 2) Load raw data so we can grab this sim's design + SP
    # --------------------------------------------------
    (
        U_lat,
        volume_snap,
        sim_index,
        time_vals,
        volume_SP,
        design_all,
        n_sims
    ) = load_raw_data(r_modes)

    Z_state_all = np.concatenate(
        [U_lat, volume_snap.reshape(-1, 1)],
        axis=1
    ).astype(np.float32)

    idx = np.where(sim_index == sim_id)[0]
    if idx.size < 2:
        raise ValueError(f"Sim {sim_id} has too few frames ({idx.size}).")

    t_s = time_vals[idx]
    order = np.argsort(t_s)
    idx_sorted = idx[order]

    Z_true_sim = Z_state_all[idx_sorted, :]
    SP_sim     = np.asarray(volume_SP[idx_sorted], dtype=np.float32)
    design_sim = np.asarray(design_all[idx_sorted, :], dtype=np.float32)

    design_raw  = design_sim[0:1, :]
    design_norm = (design_raw - design_mean) / design_std
    design_vec  = design_norm.reshape(-1).astype(np.float32)

    # --------------------------------------------------
    # 3) Pull rollout arrays
    # --------------------------------------------------
    z_pred = np.asarray(rollout["z_pred"], dtype=np.float32)
    z_true = np.asarray(rollout["z_true"], dtype=np.float32)
    e_hist = np.asarray(rollout["e_hist"], dtype=np.float32)
    I_hist = np.asarray(rollout["I_hist"], dtype=np.float32)
    t_roll = np.asarray(rollout["t"], dtype=np.float64)

    if z_pred.ndim != 2 or z_pred.shape[1] != D_state:
        raise ValueError(f"rollout['z_pred'] has shape {z_pred.shape}, expected (*, {D_state}).")

    T = z_pred.shape[0]
    if traj_end_step is None:
        traj_end_step = T - 1
    traj_end_step = int(np.clip(traj_end_step, 0, T - 1))

    # truncated trajectories to display
    z_pred_plot = z_pred[:traj_end_step + 1]
    z_true_plot = z_true[:traj_end_step + 1]

    # --------------------------------------------------
    # 4) Build conditioning point from selected sim/final plotted step
    # --------------------------------------------------
    z_ref_raw = z_pred[traj_end_step].copy()
    e_raw     = float(e_hist[traj_end_step])
    I_raw     = float(I_hist[traj_end_step])
    SP_raw    = float(SP_sim[traj_end_step])

    # --------------------------------------------------
    # 5) Determine plot limits from truncated trajectory
    # --------------------------------------------------
    z1_traj = z_pred_plot[:, mode1_idx]
    z2_traj = z_pred_plot[:, mode2_idx]

    if z1_lim is None:
        dz1 = z1_traj.max() - z1_traj.min()
        pad1 = pad_frac * (dz1 + 1e-12)
        z1_lim = (z1_traj.min() - pad1, z1_traj.max() + pad1)

    if z2_lim is None:
        dz2 = z2_traj.max() - z2_traj.min()
        pad2 = pad_frac * (dz2 + 1e-12)
        z2_lim = (z2_traj.min() - pad2, z2_traj.max() + pad2)

    z1_grid = np.linspace(z1_lim[0], z1_lim[1], grid_n)
    z2_grid = np.linspace(z2_lim[0], z2_lim[1], grid_n)
    Z1, Z2 = np.meshgrid(z1_grid, z2_grid)

    # --------------------------------------------------
    # 6) Evaluate vector field on grid
    # --------------------------------------------------
    X_list = []

    for a, b in zip(Z1.ravel(), Z2.ravel()):
        z_raw = z_ref_raw.copy()
        z_raw[mode1_idx] = a
        z_raw[mode2_idx] = b

        z_norm = (z_raw - z_mean) / z_std
        e_norm = (e_raw - e_mean) / e_std
        I_norm = (I_raw - I_mean) / I_std
        sp_norm = (SP_raw - sp_mean) / sp_std

        x = np.concatenate([
            z_norm.astype(np.float32),
            np.array([e_norm], dtype=np.float32),
            np.array([I_norm], dtype=np.float32),
            np.array([sp_norm], dtype=np.float32),
            design_vec.astype(np.float32),
        ])
        X_list.append(x)

    X = np.asarray(X_list, dtype=np.float32)

    with torch.no_grad():
        x_t = torch.from_numpy(X).to(device)
        dz_norm = model(x_t).detach().cpu().numpy()

    dz_raw = dz_norm * z_std[None, :]

    DZ1 = dz_raw[:, mode1_idx].reshape(Z1.shape)
    DZ2 = dz_raw[:, mode2_idx].reshape(Z2.shape)

    # --------------------------------------------------
    # 6b) Auto-scale arrows for quiver
    # --------------------------------------------------
    mag = np.sqrt(DZ1**2 + DZ2**2)

    if normalize_arrows:
        Udir = DZ1 / (mag + 1e-12)
        Vdir = DZ2 / (mag + 1e-12)
    else:
        Udir = DZ1.copy()
        Vdir = DZ2.copy()

    dx = float(z1_grid[1] - z1_grid[0]) if len(z1_grid) > 1 else 1.0
    dy = float(z2_grid[1] - z2_grid[0]) if len(z2_grid) > 1 else 1.0

    target_len = arrow_length_frac * max(dx, dy)

    if normalize_arrows:
        U_plot = target_len * Udir
        V_plot = target_len * Vdir
    else:
        mag_flat = mag.ravel()
        finite_mask = np.isfinite(mag_flat)
        mag_ref = np.percentile(mag_flat[finite_mask], 75) if np.any(finite_mask) else 1.0
        mag_ref = max(float(mag_ref), 1e-12)

        U_plot = target_len * (Udir / mag_ref)
        V_plot = target_len * (Vdir / mag_ref)


    # --------------------------------------------------
    # 6c) Evaluate model instantaneous velocity at terminal plotted point
    # --------------------------------------------------
    z_end_raw_full = z_pred[traj_end_step].copy()
    z_end_norm = (z_end_raw_full - z_mean) / z_std
    e_end_norm = (e_raw - e_mean) / e_std
    I_end_norm = (I_raw - I_mean) / I_std
    sp_end_norm = (SP_raw - sp_mean) / sp_std

    x_end = np.concatenate([
        z_end_norm.astype(np.float32),
        np.array([e_end_norm], dtype=np.float32),
        np.array([I_end_norm], dtype=np.float32),
        np.array([sp_end_norm], dtype=np.float32),
        design_vec.astype(np.float32),
    ])[None, :]

    with torch.no_grad():
        x_end_t = torch.from_numpy(x_end).to(device)
        dz_end_norm = model(x_end_t).detach().cpu().numpy()[0]

    dz_end_raw = dz_end_norm * z_std
    vz1_end = float(dz_end_raw[mode1_idx])
    vz2_end = float(dz_end_raw[mode2_idx])

    # --------------------------------------------------
    # 7) Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    if use_streamplot:
        ax.streamplot(
            Z1, Z2,
            DZ1, DZ2,
            density=stream_density,
            color=stream_color,
            linewidth=stream_linewidth,
            arrowsize=stream_arrowsize,
        )
    else:
        ax.quiver(
            Z1, Z2,
            U_plot, V_plot,
            color=quiver_color,
            alpha=quiver_alpha,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=quiver_width,
            headwidth=quiver_headwidth,
            headlength=quiver_headlength,
            headaxislength=quiver_headaxislength,
            linewidths=quiver_linewidth,
        )

    ax.plot(
        z_pred_plot[:, mode1_idx],
        z_pred_plot[:, mode2_idx],
        linewidth=pred_linewidth,
        linestyle=pred_linestyle,
        color=pred_color,
    )

    ax.scatter(
        z_pred_plot[0, mode1_idx], z_pred_plot[0, mode2_idx],
        s=start_markersize,
        marker=start_marker,
        color=pred_color,
        zorder=5,
    )

    # terminal plotted point
    z_end_1 = float(z_pred_plot[-1, mode1_idx])
    z_end_2 = float(z_pred_plot[-1, mode2_idx])

    # draw terminal tangent arrow
    if show_terminal_tangent:
        vmag = np.sqrt(vz1_end**2 + vz2_end**2)

        if vmag > 1e-12:
            tangent_len = terminal_tangent_length_frac * max(dx, dy)
            u_tan = tangent_len * vz1_end / vmag
            v_tan = tangent_len * vz2_end / vmag

            ax.quiver(
                [z_end_1], [z_end_2],
                [u_tan], [v_tan],
                color=terminal_tangent_color,
                alpha=terminal_tangent_alpha,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=terminal_tangent_width,
                headwidth=terminal_tangent_headwidth,
                headlength=terminal_tangent_headlength,
                headaxislength=terminal_tangent_headaxislength,
                zorder=7,
            )

    # draw red x only if this is the full rollout endpoint
    is_last_point = (traj_end_step == T - 1)

    if (not show_terminal_x_only_if_last) or is_last_point:
        ax.scatter(
            z_end_1, z_end_2,
            s=end_markersize,
            marker=end_marker,
            color="red",
            linewidths=end_marker_linewidth,
            zorder=8,
        )

    if show_true_traj:
        ax.plot(
            z_true_plot[:, mode1_idx],
            z_true_plot[:, mode2_idx],
            linestyle=true_linestyle,
            linewidth=true_linewidth,
            alpha=true_alpha,
            color=true_color,
        )

    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")

    if title is not None:
        ax.set_title(title)

    if show_grid:
        ax.grid(alpha=grid_alpha)

    plt.tight_layout()
    ax.set_xlim(z1_lim)
    ax.set_ylim(z2_lim)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=save_dpi,
            bbox_inches=save_bbox_inches,
            pad_inches=save_pad_inches,
        )
        print(f"[saved figure] {save_path}")

    plt.show()
    plt.show()

    print(f"[slice plot] sim={sim_id}, traj_end_step={traj_end_step}")
    print(f"  conditioning time t = {t_roll[traj_end_step]:.6g}")
    print(f"  SP_raw = {SP_raw:.6g}")
    print(f"  e_raw  = {e_raw:.6g}")
    print(f"  I_raw  = {I_raw:.6g}")

    return {
        "traj_end_step": traj_end_step,
        "z_ref_raw": z_ref_raw,
        "SP_raw": SP_raw,
        "e_raw": e_raw,
        "I_raw": I_raw,
        "design_raw": design_raw.copy(),
        "z1_lim": z1_lim,
        "z2_lim": z2_lim,
        "DZ1": DZ1,
        "DZ2": DZ2,
        "U_plot": U_plot,
        "V_plot": V_plot,
        "Z1": Z1,
        "Z2": Z2,
        "dx": dx,
        "dy": dy,
        "target_len": target_len,
    }


def plot_selected_sim_cnn_node_z1_z2_slice(
    r_modes: int,
    sim_id: int,
    rollout: dict,
    traj_end_step: int | None = None,
    ckpt_path_template: str = "checkpoints/Model_D_v2_BEST.pt",
    mode1_idx: int = 0,
    mode2_idx: int = 1,
    grid_n: int = 31,
    z1_lim: tuple | None = None,
    z2_lim: tuple | None = None,
    pad_frac: float = 0.10,
    normalize_arrows: bool = True,
    use_streamplot: bool = False,

    # growth-field key in rollout
    growth_key: str = "g_pred_grid",

    # quiver styling
    quiver_color: str = "black",
    quiver_alpha: float = 0.45,
    quiver_width: float = 0.0022,
    quiver_headwidth: float = 3.0,
    quiver_headlength: float = 5.0,
    quiver_headaxislength: float = 4.5,
    quiver_linewidth: float = 0.5,

    # streamplot styling
    stream_color: str = "0.35",
    stream_linewidth: float = 1.0,
    stream_density: float = 1.1,
    stream_arrowsize: float = 1.0,

    # trajectory styling
    pred_color=None,
    pred_linewidth: float = 2.8,
    pred_linestyle: str = "-",
    true_color="crimson",
    true_linewidth: float = 2.0,
    true_linestyle: str = "--",
    true_alpha: float = 0.9,
    show_true_traj: bool = False,

    # markers
    start_marker: str = "o",
    start_markersize: float = 40,
    end_marker: str = "x",
    end_markersize: float = 55,
    end_marker_linewidth: float | None = None,

    # axes / figure
    figsize=(8, 6),
    title: str | None = None,
    show_grid: bool = True,
    grid_alpha: float = 0.2,

    # autoscaling knob
    arrow_length_frac: float = 2.0,

    # terminal tangent arrow
    show_terminal_tangent: bool = True,
    terminal_tangent_color: str = "red",
    terminal_tangent_width: float = 0.004,
    terminal_tangent_headwidth: float = 4.5,
    terminal_tangent_headlength: float = 6.0,
    terminal_tangent_headaxislength: float = 5.0,
    terminal_tangent_alpha: float = 1.0,
    terminal_tangent_length_frac: float = 2.2,
    show_terminal_x_only_if_last: bool = True,

    save_path: str | None = None,
    save_dpi: int = 400,
    save_bbox_inches: str = "tight",
    save_pad_inches: float = 0.02,
):
    """
    Plot a z_i-z_j latent vector field slice for ONE selected simulation, for
    Model D (the CNN-growth-conditioned NODE). Same visual encoding as
    plot_selected_sim_vanilla_node_z1_z2_slice (see that docstring for what
    the streamplot/quiver field, solid/dashed trajectories, markers, and
    terminal-tangent arrow each represent) -- the only difference is that
    this model's vector field is additionally conditioned on a frozen
    spatial growth field G, which must be supplied alongside the scalar
    inputs.

    This version assumes the model signature is:
        dz_norm = model(x_base, G)

    where
        x_base = [z, e, I, sp, design]
        G      = growth field tensor [B,C,H,W]

    The vector field is frozen using rollout state/auxiliary values at traj_end_step,
    including the growth field G at traj_end_step.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # 1) Load checkpoint + model + normalization stats
    # --------------------------------------------------
    ckpt_path = ckpt_path_template.format(r=r_modes, r_modes=r_modes)
    ckpt = torch.load(ckpt_path, map_location=device)

    # Reconstruct the CNN-conditioned NODE model from the checkpoint.
    model = build_cnn_node_from_ckpt(ckpt, device)
    model.eval()

    D_state = int(ckpt["latent_state_dim"])

    z_mean = np.asarray(ckpt["z_mean"], dtype=np.float32).reshape(-1)
    z_std  = np.asarray(ckpt["z_std"], dtype=np.float32).reshape(-1)

    sp_mean = float(np.asarray(ckpt["sp_mean"], dtype=np.float32).reshape(-1)[0])
    sp_std  = float(np.asarray(ckpt["sp_std"],  dtype=np.float32).reshape(-1)[0])

    design_mean = np.asarray(ckpt["design_mean"], dtype=np.float32).reshape(1, -1)
    design_std  = np.asarray(ckpt["design_std"],  dtype=np.float32).reshape(1, -1)

    e_mean = float(np.asarray(ckpt["e_mean"], dtype=np.float32).reshape(-1)[0])
    e_std  = float(np.asarray(ckpt["e_std"],  dtype=np.float32).reshape(-1)[0])

    I_mean = float(np.asarray(ckpt["I_mean"], dtype=np.float32).reshape(-1)[0])
    I_std  = float(np.asarray(ckpt["I_std"],  dtype=np.float32).reshape(-1)[0])

    # --------------------------------------------------
    # 2) Load raw data so we can grab this sim's design + SP
    # --------------------------------------------------
    (
        U_lat,
        volume_snap,
        sim_index,
        time_vals,
        volume_SP,
        design_all,
        n_sims
    ) = load_raw_data(r_modes)

    Z_state_all = np.concatenate(
        [U_lat, volume_snap.reshape(-1, 1)],
        axis=1
    ).astype(np.float32)

    idx = np.where(sim_index == sim_id)[0]
    if idx.size < 2:
        raise ValueError(f"Sim {sim_id} has too few frames ({idx.size}).")

    t_s = time_vals[idx]
    order = np.argsort(t_s)
    idx_sorted = idx[order]

    Z_true_sim = Z_state_all[idx_sorted, :]
    SP_sim     = np.asarray(volume_SP[idx_sorted], dtype=np.float32)
    design_sim = np.asarray(design_all[idx_sorted, :], dtype=np.float32)

    design_raw  = design_sim[0:1, :]
    design_norm = (design_raw - design_mean) / design_std
    design_vec  = design_norm.reshape(-1).astype(np.float32)

    # --------------------------------------------------
    # 3) Pull rollout arrays
    # --------------------------------------------------
    z_pred = np.asarray(rollout["z_pred"], dtype=np.float32)
    z_true = np.asarray(rollout["z_true"], dtype=np.float32)
    e_hist = np.asarray(rollout["e_hist"], dtype=np.float32)
    I_hist = np.asarray(rollout["I_hist"], dtype=np.float32)
    t_roll = np.asarray(rollout["t"], dtype=np.float64)

    if growth_key not in rollout:
        raise KeyError(
            f"rollout is missing '{growth_key}'. "
            f"This CNN version needs the frozen growth field history in rollout['{growth_key}']."
        )

    G_hist = np.asarray(rollout[growth_key], dtype=np.float32)

    if z_pred.ndim != 2 or z_pred.shape[1] != D_state:
        raise ValueError(f"rollout['z_pred'] has shape {z_pred.shape}, expected (*, {D_state}).")

    T = z_pred.shape[0]
    if traj_end_step is None:
        traj_end_step = T - 1
    traj_end_step = int(np.clip(traj_end_step, 0, T - 1))

    z_pred_plot = z_pred[:traj_end_step + 1]
    z_true_plot = z_true[:traj_end_step + 1]

    # --------------------------------------------------
    # 4) Build conditioning point from selected sim/final plotted step
    # --------------------------------------------------
    z_ref_raw = z_pred[traj_end_step].copy()
    e_raw     = float(e_hist[traj_end_step])
    I_raw     = float(I_hist[traj_end_step])
    SP_raw    = float(SP_sim[traj_end_step])

    G_ref = G_hist[traj_end_step].copy()

    # Ensure G_ref is [C,H,W]
    if G_ref.ndim == 2:
        G_ref = G_ref[None, :, :]   # [1,H,W]
    elif G_ref.ndim == 3:
        pass                        # [C,H,W]
    else:
        raise ValueError(
            f"Expected growth field at one step to have shape [H,W] or [C,H,W], got {G_ref.shape}"
        )

    # --------------------------------------------------
    # 5) Determine plot limits from truncated trajectory
    # --------------------------------------------------
    z1_traj = z_pred_plot[:, mode1_idx]
    z2_traj = z_pred_plot[:, mode2_idx]

    if z1_lim is None:
        dz1 = z1_traj.max() - z1_traj.min()
        pad1 = pad_frac * (dz1 + 1e-12)
        z1_lim = (z1_traj.min() - pad1, z1_traj.max() + pad1)

    if z2_lim is None:
        dz2 = z2_traj.max() - z2_traj.min()
        pad2 = pad_frac * (dz2 + 1e-12)
        z2_lim = (z2_traj.min() - pad2, z2_traj.max() + pad2)

    z1_grid = np.linspace(z1_lim[0], z1_lim[1], grid_n)
    z2_grid = np.linspace(z2_lim[0], z2_lim[1], grid_n)
    Z1, Z2 = np.meshgrid(z1_grid, z2_grid)

    # --------------------------------------------------
    # 6) Evaluate vector field on grid
    # --------------------------------------------------
    Xbase_list = []

    for a, b in zip(Z1.ravel(), Z2.ravel()):
        z_raw = z_ref_raw.copy()
        z_raw[mode1_idx] = a
        z_raw[mode2_idx] = b

        z_norm = (z_raw - z_mean) / z_std
        e_norm = (e_raw - e_mean) / e_std
        I_norm = (I_raw - I_mean) / I_std
        sp_norm = (SP_raw - sp_mean) / sp_std

        x_base = np.concatenate([
            z_norm.astype(np.float32),
            np.array([e_norm], dtype=np.float32),
            np.array([I_norm], dtype=np.float32),
            np.array([sp_norm], dtype=np.float32),
            design_vec.astype(np.float32),
        ])
        Xbase_list.append(x_base)

    X_base = np.asarray(Xbase_list, dtype=np.float32)  # [Ngrid, Dbase]
    Ngrid = X_base.shape[0]

    # Repeat frozen growth field across all grid points
    G_batch = np.repeat(G_ref[None, ...], Ngrid, axis=0).astype(np.float32)  # [Ngrid,C,H,W]

    with torch.no_grad():
        x_base_t = torch.from_numpy(X_base).to(device)
        G_t = torch.from_numpy(G_batch).to(device)
        dz_norm = model(x_base_t, G_t).detach().cpu().numpy()

    dz_raw = dz_norm * z_std[None, :]

    DZ1 = dz_raw[:, mode1_idx].reshape(Z1.shape)
    DZ2 = dz_raw[:, mode2_idx].reshape(Z2.shape)

    # --------------------------------------------------
    # 6b) Auto-scale arrows for quiver
    # --------------------------------------------------
    mag = np.sqrt(DZ1**2 + DZ2**2)

    if normalize_arrows:
        Udir = DZ1 / (mag + 1e-12)
        Vdir = DZ2 / (mag + 1e-12)
    else:
        Udir = DZ1.copy()
        Vdir = DZ2.copy()

    dx = float(z1_grid[1] - z1_grid[0]) if len(z1_grid) > 1 else 1.0
    dy = float(z2_grid[1] - z2_grid[0]) if len(z2_grid) > 1 else 1.0

    target_len = arrow_length_frac * max(dx, dy)

    if normalize_arrows:
        U_plot = target_len * Udir
        V_plot = target_len * Vdir
    else:
        mag_flat = mag.ravel()
        finite_mask = np.isfinite(mag_flat)
        mag_ref = np.percentile(mag_flat[finite_mask], 75) if np.any(finite_mask) else 1.0
        mag_ref = max(float(mag_ref), 1e-12)

        U_plot = target_len * (Udir / mag_ref)
        V_plot = target_len * (Vdir / mag_ref)

    # --------------------------------------------------
    # 6c) Evaluate instantaneous velocity at terminal plotted point
    # --------------------------------------------------
    z_end_raw_full = z_pred[traj_end_step].copy()
    z_end_norm = (z_end_raw_full - z_mean) / z_std
    e_end_norm = (e_raw - e_mean) / e_std
    I_end_norm = (I_raw - I_mean) / I_std
    sp_end_norm = (SP_raw - sp_mean) / sp_std

    x_end = np.concatenate([
        z_end_norm.astype(np.float32),
        np.array([e_end_norm], dtype=np.float32),
        np.array([I_end_norm], dtype=np.float32),
        np.array([sp_end_norm], dtype=np.float32),
        design_vec.astype(np.float32),
    ])[None, :]

    G_end = G_ref[None, ...].astype(np.float32)

    with torch.no_grad():
        x_end_t = torch.from_numpy(x_end).to(device)
        G_end_t = torch.from_numpy(G_end).to(device)
        dz_end_norm = model(x_end_t, G_end_t).detach().cpu().numpy()[0]

    dz_end_raw = dz_end_norm * z_std
    vz1_end = float(dz_end_raw[mode1_idx])
    vz2_end = float(dz_end_raw[mode2_idx])

    # --------------------------------------------------
    # 7) Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    if use_streamplot:
        ax.streamplot(
            Z1, Z2,
            DZ1, DZ2,
            density=stream_density,
            color=stream_color,
            linewidth=stream_linewidth,
            arrowsize=stream_arrowsize,
        )
    else:
        ax.quiver(
            Z1, Z2,
            U_plot, V_plot,
            color=quiver_color,
            alpha=quiver_alpha,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=quiver_width,
            headwidth=quiver_headwidth,
            headlength=quiver_headlength,
            headaxislength=quiver_headaxislength,
            linewidths=quiver_linewidth,
        )

    ax.plot(
        z_pred_plot[:, mode1_idx],
        z_pred_plot[:, mode2_idx],
        linewidth=pred_linewidth,
        linestyle=pred_linestyle,
        color=pred_color,
    )

    ax.scatter(
        z_pred_plot[0, mode1_idx], z_pred_plot[0, mode2_idx],
        s=start_markersize,
        marker=start_marker,
        color=pred_color,
        zorder=5,
    )

    z_end_1 = float(z_pred_plot[-1, mode1_idx])
    z_end_2 = float(z_pred_plot[-1, mode2_idx])

    if show_terminal_tangent:
        vmag = np.sqrt(vz1_end**2 + vz2_end**2)

        if vmag > 1e-12:
            tangent_len = terminal_tangent_length_frac * max(dx, dy)
            u_tan = tangent_len * vz1_end / vmag
            v_tan = tangent_len * vz2_end / vmag

            ax.quiver(
                [z_end_1], [z_end_2],
                [u_tan], [v_tan],
                color=terminal_tangent_color,
                alpha=terminal_tangent_alpha,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=terminal_tangent_width,
                headwidth=terminal_tangent_headwidth,
                headlength=terminal_tangent_headlength,
                headaxislength=terminal_tangent_headaxislength,
                zorder=7,
            )

    is_last_point = (traj_end_step == T - 1)

    if (not show_terminal_x_only_if_last) or is_last_point:
        ax.scatter(
            z_end_1, z_end_2,
            s=end_markersize,
            marker=end_marker,
            color="red",
            linewidths=end_marker_linewidth,
            zorder=8,
        )

    if show_true_traj:
        ax.plot(
            z_true_plot[:, mode1_idx],
            z_true_plot[:, mode2_idx],
            linestyle=true_linestyle,
            linewidth=true_linewidth,
            alpha=true_alpha,
            color=true_color,
        )

    ax.set_xlabel(rf"$z_{{{mode1_idx+1}}}$")
    ax.set_ylabel(rf"$z_{{{mode2_idx+1}}}$")

    if title is not None:
        ax.set_title(title)

    if show_grid:
        ax.grid(alpha=grid_alpha)

    ax.set_xlim(z1_lim)
    ax.set_ylim(z2_lim)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=save_dpi,
            bbox_inches=save_bbox_inches,
            pad_inches=save_pad_inches,
        )
        print(f"[saved figure] {save_path}")

    plt.show()

    print(f"[slice plot] sim={sim_id}, traj_end_step={traj_end_step}")
    print(f"  conditioning time t = {t_roll[traj_end_step]:.6g}")
    print(f"  SP_raw = {SP_raw:.6g}")
    print(f"  e_raw  = {e_raw:.6g}")
    print(f"  I_raw  = {I_raw:.6g}")
    print(f"  G_ref shape = {tuple(G_ref.shape)}")

    return {
        "traj_end_step": traj_end_step,
        "z_ref_raw": z_ref_raw,
        "SP_raw": SP_raw,
        "e_raw": e_raw,
        "I_raw": I_raw,
        "design_raw": design_raw.copy(),
        "G_ref": G_ref.copy(),
        "z1_lim": z1_lim,
        "z2_lim": z2_lim,
        "DZ1": DZ1,
        "DZ2": DZ2,
        "U_plot": U_plot,
        "V_plot": V_plot,
        "Z1": Z1,
        "Z2": Z2,
        "dx": dx,
        "dy": dy,
        "target_len": target_len,
    }
