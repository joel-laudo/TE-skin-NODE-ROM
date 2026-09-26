"""Build Figure 11's four final-time error-histogram panels (one per model
family A-D), using each model's revised main-paper BEST checkpoint.

Each panel is a histogram, across the 185-simulation validation set, of
per-simulation surface displacement RMSE evaluated at a single common
"final" time -- inferred as the maximum time value seen across all
simulations in that model's rollout zarr (`infer_support_time_from_global_max_t`),
since not every simulation's trajectory necessarily extends to exactly the
same last time point. Bin count and x-axis range differ deliberately by
panel: A/B/D use 20 bins over [0, 15] (A uses only 5, since it has very few
distinct low error values); C uses 60 bins with no x-limit, so its full
range -- including a ~25.9 outlier -- stays visible.

Read-only with respect to models/data: no training, no checkpoint/zarr
modification. Reads deliverables/{A,B,C,D}_v2_val_rollouts_r9.zarr
(produced by `evaluation/generate_deliverables.py`, run from the repository
root) and writes new figure files into this script's own directory
(figures/figure_11/).
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
    # (zarr_path, bins, xlim) -- xlim=None lets matplotlib auto-range to show all data
    "A": ("deliverables/A_v2_val_rollouts_r9.zarr", 5, (0, 15)),  # few bins: errors are tightly clustered at low values
    "B": ("deliverables/B_v2_val_rollouts_r9.zarr", 20, (0, 15)),
    "C": ("deliverables/C_v2_val_rollouts_r9.zarr", 60, None),  # more bins + no xlim so the full range (incl. the ~25.9 max outlier) is visible
    "D": ("deliverables/D_v2_val_rollouts_r9.zarr", 20, (0, 15)),
}


def plot_histogram_panel(rollout_zarr_path, bins, save_stem,
                          color="tab:red", alpha=0.8, edgecolor="black", linewidth=0.8,
                          xlim=(0, 15), show_mean=False, show_median=False,
                          infer_support_time_from_global_max_t=True, atol=1e-10,
                          figsize=(7.2, 5.0), tick_fontsize=24, use_seaborn_style=True,
                          save_dpi=600):
    """Build and save one model family's final-time RMSE histogram.

    Finds the common evaluation time (see module docstring), interpolates
    each simulation's error trajectory to that time, and histograms the
    resulting per-simulation values. Simulations whose trajectory doesn't
    span the target time, or whose interpolated value is non-finite, are
    skipped and reported in the printed summary rather than silently
    dropped."""
    if use_seaborn_style:
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except Exception:
            plt.style.use("default")

    root = zarr.open_group(rollout_zarr_path, mode="r")
    sims_grp = root["simulations"]
    sim_keys = sorted(list(sims_grp.group_keys()))
    model_name = root.attrs.get("model_name", "model")

    target_time = None
    if infer_support_time_from_global_max_t:
        all_t_max = []
        for sk in sim_keys:
            g = sims_grp[sk]
            y = np.asarray(g["disp_err_pointwise_rmse"], dtype=np.float64).squeeze()
            if y.size == 0:
                continue
            t = np.asarray(g["t"], dtype=np.float64).squeeze() if "t" in g else np.arange(y.shape[0], dtype=np.float64)
            if t.size == y.size and t.size > 0:
                all_t_max.append(np.max(t))
        target_time = float(np.max(all_t_max))

    rmse_vals, used, skipped_no_support, skipped_invalid = [], [], [], []
    for sk in sim_keys:
        g = sims_grp[sk]
        y = np.asarray(g["disp_err_pointwise_rmse"], dtype=np.float64).squeeze()
        t = np.asarray(g["t"], dtype=np.float64).squeeze() if "t" in g else np.arange(y.shape[0], dtype=np.float64)
        if t.size != y.size or t.size == 0:
            skipped_invalid.append(sk)
            continue
        t_unique, idx_unique = np.unique(t, return_index=True)
        y_unique = y[idx_unique]
        t_min, t_max = t_unique[0], t_unique[-1]
        if (target_time < t_min - atol) or (target_time > t_max + atol):
            skipped_no_support.append(sk)
            continue
        val = float(np.interp(target_time, t_unique, y_unique))
        if np.isfinite(val):
            rmse_vals.append(val)
            used.append(sk)
        else:
            skipped_invalid.append(sk)

    rmse_vals = np.asarray(rmse_vals, dtype=np.float64)
    mean_val, median_val = float(np.mean(rmse_vals)), float(np.median(rmse_vals))
    p10, p90 = float(np.percentile(rmse_vals, 10)), float(np.percentile(rmse_vals, 90))
    std_val = float(np.std(rmse_vals))

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(rmse_vals, bins=bins, density=False, color=color, alpha=alpha,
            edgecolor=edgecolor, linewidth=linewidth)
    if show_mean:
        ax.axvline(mean_val, color="tab:red", linestyle="--", linewidth=2.0, label=f"Mean = {mean_val:.3f}")
    if show_median:
        ax.axvline(median_val, color="black", linestyle="-", linewidth=2.2, label=f"Median = {median_val:.3f}")
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    print(f"Model: {model_name}")
    print(f"Evaluating RMSE only for simulations with support at t = {target_time:.6f}")
    print(f"Number of sims used      : {len(rmse_vals)}")
    print(f"Number skipped (support) : {len(skipped_no_support)}")
    print(f"Number skipped (invalid) : {len(skipped_invalid)}")
    print(f"Mean RMSE                : {mean_val:.6f}")
    print(f"Median RMSE              : {median_val:.6f}")
    print(f"Std RMSE                 : {std_val:.6f}")
    print(f"10th percentile          : {p10:.6f}")
    print(f"90th percentile          : {p90:.6f}")
    print(f"Min RMSE                 : {np.min(rmse_vals):.6f}")
    print(f"Max RMSE                 : {np.max(rmse_vals):.6f}")

    fig.savefig(save_stem + ".pdf", dpi=save_dpi, bbox_inches="tight")
    fig.savefig(save_stem + ".png", dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save_stem}.pdf / .png")
    print()


for fam, (zarr_path, bins, xlim) in ZARRS.items():
    save_stem = os.path.join(OUT_DIR, f"Model_{fam}_final_error_histogram")
    plot_histogram_panel(os.path.join(PROJECT_ROOT, zarr_path), bins, save_stem, xlim=xlim)

print("DONE")
