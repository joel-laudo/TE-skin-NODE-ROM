"""Assembles the final 4-panel supplement figure, 2D-map version: all four
panels -- (a) FE non-convergence, (b) POD reconstruction error, (c) Model A
displacement error, (d) Model D displacement error -- rendered as 2D
parameter-space maps in the same (kappa, mu) plane, with identical
axes/ranges and matching visual style (contourf, Reds colormap,
converged/validation scatter plus non-converged X-marker overlay,
colorbar, panel lettering).

Panel (a)'s surface uses the identical RandomForestClassifier fit as
build_panel_a.py (kappa+mu only, n_estimators=500, max_depth=4,
class_weight="balanced", random_state=123 -- deterministic, so this
reproduces that panel exactly). Panels (b)-(d) use the identical
Gaussian-kernel local-average surfaces as build_panels_bcd_2dmaps.py. No
new analysis beyond what those two scripts already do -- this script only
lays the four panels out together as one true-vector PDF (not a raster
composite of the individual panel images), for direct side-by-side
comparison with the 1D/bin-based version of this figure assembled from
build_panel_a.py and build_panels_bcd.py.

Reads: step9_master_per_sim_table.csv (from step9_master_join.py) and
all_1000_jobs_with_convergence_flag.csv (from
step1_recover_failed_jobs.py), both found in this script's parent
directory (fe_nonconvergence_and_rom_accuracy/, via CONV_DIR).

Writes: Nonconverged_sim_analysis_supp_rev_2Dmaps.{pdf,png} -- the
assembled 4-panel supplement figure for SI Section 2 / Figure 2.

Run from the repository root.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONV_DIR = os.path.dirname(SCRIPT_DIR)
OUT_DIR = SCRIPT_DIR

PANEL_RC = {
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "axes.linewidth": 1.0,
    "grid.alpha": 0.25,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
}
plt.rcParams.update(PANEL_RC)

master = pd.read_csv(os.path.join(CONV_DIR, "step9_master_per_sim_table.csv"))
all_jobs = pd.read_csv(os.path.join(CONV_DIR, "all_1000_jobs_with_convergence_flag.csv"))
val = master[master["is_validation"]].copy()

KAPPA_RANGE = (all_jobs["kappa"].min(), all_jobs["kappa"].max())
MU_RANGE = (all_jobs["mu"].min(), all_jobs["mu"].max())
xg = np.linspace(*KAPPA_RANGE, 200)
yg = np.linspace(*MU_RANGE, 200)
XG, YG = np.meshgrid(xg, yg)
GRID_XY = np.column_stack([XG.ravel(), YG.ravel()])

conv_mask = all_jobs["converged"].values.astype(bool)
failed = all_jobs.loc[~conv_mask, ["kappa", "mu"]]

BW_KAPPA = 0.12 * (KAPPA_RANGE[1] - KAPPA_RANGE[0])
BW_MU = 0.12 * (MU_RANGE[1] - MU_RANGE[0])


def kernel_surface(df, target_col):
    """Nadaraya-Watson kernel regression (same construction as
    build_panels_bcd_2dmaps.py's fit_surface): a Gaussian-kernel-weighted
    local average of df[target_col] over the (kappa, mu) grid."""
    kx = df["kappa"].values[:, None]
    my = df["mu"].values[:, None]
    z = df[target_col].values[:, None]
    gx = XG.ravel()[None, :]
    gy = YG.ravel()[None, :]
    w = np.exp(-0.5 * (((kx - gx) / BW_KAPPA) ** 2 + ((my - gy) / BW_MU) ** 2))
    surface = (w * z).sum(axis=0) / w.sum(axis=0)
    return surface.reshape(XG.shape)


# Panel (a) surface: 2-feature classifier, identical to build_panel_a.py
rf2 = RandomForestClassifier(n_estimators=500, max_depth=4, class_weight="balanced",
                               random_state=123, n_jobs=-1)
rf2.fit(all_jobs[["kappa", "mu"]].values, (~all_jobs["converged"]).astype(int).values)
surf_a = rf2.predict_proba(GRID_XY)[:, 1].reshape(XG.shape)

surf_b = kernel_surface(master, "e_pod")
surf_c = kernel_surface(val.dropna(subset=["e_node_A"]), "e_node_A")
surf_d = kernel_surface(val.dropna(subset=["e_node_D"]), "e_node_D")

# Panels (c) and (d) plot the same quantity (validation displacement error)
# for two different models, so they share one color scale (spanning both
# surfaces' combined range) for a direct visual comparison. Panels (a)/(b)
# plot different quantities and keep their own independent scales.
CD_VMIN = min(surf_c.min(), surf_d.min())
CD_VMAX = max(surf_c.max(), surf_d.max())
CD_LEVELS = np.linspace(CD_VMIN, CD_VMAX, 16)

PANELS = [
    ("a", surf_a, None, "FE non-convergence", "Predicted P(FE non-convergence)",
     [(all_jobs.loc[conv_mask, "kappa"], all_jobs.loc[conv_mask, "mu"], "steelblue",
       f"converged (n={conv_mask.sum()})")]),
    ("b", surf_b, None, "POD reconstruction error", r"Predicted $e_{POD}$ (cm)",
     [(master["kappa"], master["mu"], "steelblue", f"converged (n={len(master)})")]),
    ("c", surf_c, CD_LEVELS, "Model A displacement error", r"Predicted $e_{NODE,A}$ (cm)",
     [(val["kappa"], val["mu"], "tab:blue", f"validation (n={len(val)})")]),
    ("d", surf_d, CD_LEVELS, "Model D displacement error", r"Predicted $e_{NODE,D}$ (cm)",
     [(val["kappa"], val["mu"], "tab:red", f"validation (n={len(val)})")]),
]

fig, axes = plt.subplots(2, 2, figsize=(13.2, 11.6))
for ax, (letter, surf, shared_levels, title, cbar_label, scatters) in zip(axes.flat, PANELS):
    levels = shared_levels if shared_levels is not None else 15
    cf = ax.contourf(XG, YG, surf, levels=levels, cmap="Reds", alpha=0.75)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(cbar_label)

    for xvals, yvals, color, label in scatters:
        ax.scatter(xvals, yvals, s=16, color=color, alpha=0.6, edgecolors="none", label=label)
    ax.scatter(failed["kappa"], failed["mu"], s=38, color="black", marker="x",
               linewidths=1.4, label=f"non-converged (n={len(failed)})")

    ax.set_xlim(KAPPA_RANGE)
    ax.set_ylim(MU_RANGE)
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\mu$")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.text(-0.14, 1.05, f"({letter})", transform=ax.transAxes, fontsize=15, fontweight="bold",
             va="bottom", ha="right")

fig.suptitle("Non-convergence risk and reconstruction/prediction error across parameter space "
             r"($\kappa$, $\mu$)", fontsize=14, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.98])

out_pdf = os.path.join(OUT_DIR, "Nonconverged_sim_analysis_supp_rev_2Dmaps.pdf")
out_png = os.path.join(OUT_DIR, "Nonconverged_sim_analysis_supp_rev_2Dmaps.png")
fig.savefig(out_pdf)
fig.savefig(out_png)
plt.close(fig)
print(f"saved {out_pdf}")
print(f"saved {out_png}")
print("DONE")
