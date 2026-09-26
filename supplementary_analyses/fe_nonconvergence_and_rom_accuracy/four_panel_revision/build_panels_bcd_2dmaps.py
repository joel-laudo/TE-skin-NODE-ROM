"""4-panel supplement, Panels (b)-(d), 2D-map version: same visual style
as Panel (a), instead of the 1D kappa-binned line/IQR-band presentation
used in build_panels_bcd.py. Uses the same underlying per-simulation data
as those panels (step9_master_per_sim_table.csv's
e_pod/e_node_A/e_node_D columns -- no new analysis, no recomputation of
the reported medians/correlations). The only new computation is a smooth
2-feature (kappa, mu) surface (Gaussian-kernel local average; see
fit_surface below) computed purely for this visualization, in the same
spirit as Panel (a)'s 2-feature classifier surface -- not the canonical
8-feature classifiers used for the quantitative failure-risk analysis
elsewhere in this repo; a smooth 2D surface is simply needed to render a
contour map.

Each panel additionally overlays the 73 non-converged (failed) simulations'
(kappa, mu) locations as black X markers -- identical marker/style to Panel
(a) -- so a reader can see directly, in the same 2D layout across all four
panels, whether the non-convergence region coincides with elevated
reconstruction/prediction error.

Reads: step9_master_per_sim_table.csv (from step9_master_join.py) and
all_1000_jobs_with_convergence_flag.csv (from
step1_recover_failed_jobs.py), both found in this script's parent
directory (fe_nonconvergence_and_rom_accuracy/, via CONV_DIR).

Writes (into this four_panel_revision/ folder): panel_b_pod_kappa_2dmap,
panel_c_modelA_kappa_2dmap, panel_d_modelD_kappa_2dmap (plain and lettered
variants) -- combined with panel (a) into the 4-panel supplement figure.

Run from the repository root.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONV_DIR = os.path.dirname(SCRIPT_DIR)
OUT_DIR = SCRIPT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# Identical rcParams/figsize to build_panel_a.py, for visual consistency.
PANEL_RC = {
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.2,
    "lines.markersize": 6,
    "grid.alpha": 0.25,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
}
plt.rcParams.update(PANEL_RC)
PANEL_FIGSIZE = (6.4, 5.8)

master = pd.read_csv(os.path.join(CONV_DIR, "step9_master_per_sim_table.csv"))
all_jobs = pd.read_csv(os.path.join(CONV_DIR, "all_1000_jobs_with_convergence_flag.csv"))
val = master[master["is_validation"]].copy()

# Fixed axis ranges shared with Panel (a): the FULL 1000-job (kappa, mu)
# range, not just the converged/validation subset's own range, so all four
# panels are directly comparable on identical axes/parameter ranges.
KAPPA_RANGE = (all_jobs["kappa"].min(), all_jobs["kappa"].max())
MU_RANGE = (all_jobs["mu"].min(), all_jobs["mu"].max())
xg = np.linspace(*KAPPA_RANGE, 200)
yg = np.linspace(*MU_RANGE, 200)
XG, YG = np.meshgrid(xg, yg)
GRID_XY = np.column_stack([XG.ravel(), YG.ravel()])

failed = all_jobs.loc[~all_jobs["converged"].astype(bool), ["kappa", "mu"]]

# Gaussian-kernel local-average smoother (Nadaraya-Watson), not a
# random-forest regressor: with only ~185 (Model A/D) or 927 (POD)
# irregularly-scattered points evaluated on a 200x200 grid, a tree-ensemble
# regressor produces visible axis-aligned striping artifacts at individual
# sample coordinates (each tree can isolate a single point with a thin
# rectangular leaf), which is visually distracting. A kernel-weighted local
# average is smooth by construction and is the standard way to turn
# irregular scattered (x, y, z) samples into a continuous 2D map for
# visualization. Bandwidth is 12% of each parameter's full 1000-job range
# (a conventional default for this sample size/data density) -- purely a
# visualization choice, exactly like Panel (a)'s 2-feature classifier
# surface; the reported quantitative results (medians, correlations,
# kappa-bin table) are untouched.
BW_KAPPA = 0.12 * (KAPPA_RANGE[1] - KAPPA_RANGE[0])
BW_MU = 0.12 * (MU_RANGE[1] - MU_RANGE[0])


def fit_surface(df, target_col):
    """Nadaraya-Watson kernel regression: at each of the 200x200 grid
    points, average df[target_col] weighted by a 2D Gaussian kernel
    centered on that grid point (bandwidths BW_KAPPA, BW_MU along kappa
    and mu respectively), turning the scattered (kappa, mu, target_col)
    samples into a smooth interpolated surface."""
    kx = df["kappa"].values[:, None]
    my = df["mu"].values[:, None]
    z = df[target_col].values[:, None]
    gx = XG.ravel()[None, :]
    gy = YG.ravel()[None, :]
    w = np.exp(-0.5 * (((kx - gx) / BW_KAPPA) ** 2 + ((my - gy) / BW_MU) ** 2))
    surface = (w * z).sum(axis=0) / w.sum(axis=0)
    return surface.reshape(XG.shape)


PANELS = [
    ("b", master, "e_pod", "black", "steelblue",
     "POD reconstruction error", "Predicted $e_{POD}$ (cm)", "panel_b_pod_kappa_2dmap",
     f"converged (n={len(master)})"),
    ("c", val, "e_node_A", "tab:blue", "tab:blue",
     "Model A displacement error", r"Predicted $e_{NODE,A}$ (cm)", "panel_c_modelA_kappa_2dmap",
     f"validation (n={len(val)})"),
    ("d", val, "e_node_D", "tab:red", "tab:red",
     "Model D displacement error", r"Predicted $e_{NODE,D}$ (cm)", "panel_d_modelD_kappa_2dmap",
     f"validation (n={len(val)})"),
]


def render(letter, df, target_col, marker_color, scatter_label, title, cbar_label, stem, with_letter):
    surface = fit_surface(df, target_col)

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    cf = ax.contourf(XG, YG, surface, levels=15, cmap="Reds", alpha=0.75)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(cbar_label)

    ax.scatter(df["kappa"], df["mu"], s=18, color=marker_color, alpha=0.6,
               edgecolors="none", label=scatter_label)
    ax.scatter(failed["kappa"], failed["mu"], s=45, color="black", marker="x",
               linewidths=1.5, label=f"non-converged (n={len(failed)})")

    ax.set_xlim(KAPPA_RANGE)
    ax.set_ylim(MU_RANGE)
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\mu$")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    if with_letter:
        ax.text(-0.12, 1.04, f"({letter})", transform=ax.transAxes, fontsize=15, fontweight="bold",
                 va="bottom", ha="right")
    fig.tight_layout()
    suffix = "_lettered" if with_letter else ""
    fig.savefig(os.path.join(OUT_DIR, stem + suffix + ".pdf"))
    fig.savefig(os.path.join(OUT_DIR, stem + suffix + ".png"))
    plt.close(fig)
    print(f"saved {stem}{suffix}.pdf / .png")


for letter, df, target_col, marker_color, scatter_label, title, cbar_label, stem, label_text in PANELS:
    render(letter, df.dropna(subset=[target_col]), target_col, marker_color, label_text, title, cbar_label, stem, with_letter=False)
    render(letter, df.dropna(subset=[target_col]), target_col, marker_color, label_text, title, cbar_label, stem, with_letter=True)

print("DONE")
