"""Build Figure 11's four displacement-trajectory panels (one per model
family A-D), using each model's revised main-paper BEST checkpoint
(epochs 356/185/353/357 for A/B/C/D respectively).

Each panel overlays every validation simulation's surface displacement
RMSE vs. time for one model family, then summarizes them with a median
curve and a shaded 10th-90th percentile band (individual per-simulation
curves are computed but not drawn by default -- see show_all_curves).

Data source: deliverables/{A,B,C,D}_v2_val_rollouts_r9.zarr, produced by
`evaluation/generate_deliverables.py` from each family's BEST checkpoint
(checkpoints/Model_{fam}_v2_BEST.pt, per the zarr's own
`ckpt_path_template` attribute).

Read-only with respect to models/data: no training, no checkpoint/zarr
modification. Only reads the zarrs and writes new figure files into this
script's own directory (figures/figure_11/).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import zarr

BASE = os.path.dirname(os.path.abspath(__file__))
# figures/figure_11 -> figures -> repository root
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE))
OUT_DIR = BASE
os.makedirs(OUT_DIR, exist_ok=True)

ZARRS = {
    "A": "deliverables/A_v2_val_rollouts_r9.zarr",
    "B": "deliverables/B_v2_val_rollouts_r9.zarr",
    "C": "deliverables/C_v2_val_rollouts_r9.zarr",
    "D": "deliverables/D_v2_val_rollouts_r9.zarr",
}


def plot_trajectory_panel(rollout_zarr_path, save_stem,
                           use_relative_time=False, linewidth=0.9, alpha=0.12,
                           show_all_curves=False, summary_linewidth=3.2,
                           band_alpha=0.18, n_interp=300, use_seaborn_style=True,
                           ylim=(0, 10), save_dpi=600):
    """Build and save one model family's displacement-RMSE-vs-time panel.

    Each simulation's trajectory is linearly interpolated onto a shared
    time grid (`n_interp` points spanning the min/max time across all
    simulations, or normalized 0-1 "relative time" if use_relative_time),
    so that per-timestep median and 10th/90th-percentile band statistics
    can be computed across simulations even though their raw time grids
    don't line up exactly. Individual per-simulation curves are only drawn
    if show_all_curves=True; by default only the median + band are shown."""
    if use_seaborn_style:
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except Exception:
            plt.style.use("default")

    root = zarr.open_group(rollout_zarr_path, mode="r")
    sims_grp = root["simulations"]
    sim_keys = sorted(list(sims_grp.group_keys()))
    model_name = root.attrs.get("model_name", "model")

    fig, ax = plt.subplots(figsize=(8.4, 5.6))

    traj_data = []
    all_t_min, all_t_max = [], []
    for sk in sim_keys:
        g = sims_grp[sk]
        y = np.asarray(g["disp_err_pointwise_rmse"], dtype=np.float64)
        t = np.asarray(g["t"], dtype=np.float64) if "t" in g else np.arange(y.shape[0], dtype=np.float64)
        if use_relative_time:
            x = np.zeros_like(t) if (len(t) == 1 or (t[-1] - t[0]) <= 0) else (t - t[0]) / (t[-1] - t[0])
        else:
            x = t
            all_t_min.append(np.min(t))
            all_t_max.append(np.max(t))
        traj_data.append((sk, x, y))

    x_common = (np.linspace(0.0, 1.0, n_interp) if use_relative_time
                else np.linspace(min(all_t_min), max(all_t_max), n_interp))

    interp_bank = []
    for sk, x, y in traj_data:
        x_unique, idx_unique = np.unique(x, return_index=True)
        y_unique = y[idx_unique]
        if x_unique.size < 2:
            continue
        if show_all_curves:
            ax.plot(x_unique, y_unique, linewidth=linewidth, alpha=alpha)
        y_interp = np.interp(x_common, x_unique, y_unique, left=np.nan, right=np.nan)
        interp_bank.append(y_interp)

    Y = np.vstack(interp_bank)
    y_p10 = np.nanpercentile(Y, 10, axis=0)
    y_med = np.nanpercentile(Y, 50, axis=0)
    y_p90 = np.nanpercentile(Y, 90, axis=0)

    ax.fill_between(x_common, y_p10, y_p90, color="tab:red", alpha=band_alpha, label="10th-90th percentile")
    ax.plot(x_common, y_med, color="black", linewidth=summary_linewidth, label="Median")

    ax.set_title(f"Validation surface displacement RMSE trajectories | {model_name} | n={len(sim_keys)}")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    print(f"[{model_name}] Final median RMSE = {y_med[-1]:.6f}")
    print(f"[{model_name}] Second-to-last median RMSE = {y_med[-2]:.6f}")

    fig.savefig(save_stem + ".pdf", dpi=save_dpi, bbox_inches="tight")
    fig.savefig(save_stem + ".png", dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save_stem}.pdf / .png")


for fam, zarr_path in ZARRS.items():
    save_stem = os.path.join(OUT_DIR, f"Model_{fam}_displacement_trajectory")
    plot_trajectory_panel(os.path.join(PROJECT_ROOT, zarr_path), save_stem)

print("DONE")
