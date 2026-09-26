"""4-panel supplement, Panel (a): FE non-convergence map in the (kappa, mu)
plane, restyled for publication. Uses the same underlying data and model
as step4_2d_failure_map.py (identical RandomForestClassifier fit: kappa+mu
features only, n_estimators=500, max_depth=4, class_weight="balanced",
random_state=123 -- deterministic, so the probability shading here is
pixel-identical to that figure; only the styling changes).

This is explicitly a 2-feature visualization (kappa, mu only), not the
canonical 8-feature classifiers used for the quantitative failure-risk
analysis elsewhere in this repo (step2_classifier.py,
step6_failure_risk_scores.py) -- worth noting in a caption, but
intentionally left out of the panel title/labels to keep the figure
uncluttered.

Reads: all_1000_jobs_with_convergence_flag.csv (written by
step1_recover_failed_jobs.py; CONV_DIR points at this script's parent
directory, fe_nonconvergence_and_rom_accuracy/, where the step1-11
pipeline outputs live).

Writes (into this four_panel_revision/ folder): panel_a_fe_nonconvergence.
{pdf,png} and a lettered variant, panel_a_fe_nonconvergence_lettered.
{pdf,png}, combined with panels (b)-(d) into the 4-panel supplement
figure.

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
os.makedirs(OUT_DIR, exist_ok=True)

# Shared styling for all 4 panels of this supplement figure
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
PANEL_FIGSIZE = (6.4, 5.8)  # shared target footprint for all 4 panels

df = pd.read_csv(os.path.join(CONV_DIR, "all_1000_jobs_with_convergence_flag.csv"))
y = (~df["converged"]).astype(int).values

rf2 = RandomForestClassifier(n_estimators=500, max_depth=4, class_weight="balanced",
                               random_state=123, n_jobs=-1)
rf2.fit(df[["kappa", "mu"]].values, y)

xg = np.linspace(df["kappa"].min(), df["kappa"].max(), 200)
yg = np.linspace(df["mu"].min(), df["mu"].max(), 200)
XG, YG = np.meshgrid(xg, yg)
proba = rf2.predict_proba(np.column_stack([XG.ravel(), YG.ravel()]))[:, 1].reshape(XG.shape)


def render(with_letter, save_stem):
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    cf = ax.contourf(XG, YG, proba, levels=15, cmap="Reds", alpha=0.75)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Predicted P(FE non-convergence)")

    conv_mask = df["converged"].values.astype(bool)
    ax.scatter(df.loc[conv_mask, "kappa"], df.loc[conv_mask, "mu"],
               s=18, color="steelblue", alpha=0.6, label=f"converged (n={conv_mask.sum()})",
               edgecolors="none")
    ax.scatter(df.loc[~conv_mask, "kappa"], df.loc[~conv_mask, "mu"],
               s=45, color="black", marker="x", linewidths=1.5,
               label=f"non-converged (n={(~conv_mask).sum()})")

    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\mu$")
    ax.set_title("FE non-convergence")
    # Upper-right (high kappa, high mu) is one of the three failure-dense
    # corners -- upper-left is the clearest region, so the legend goes there.
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    if with_letter:
        ax.text(-0.12, 1.04, "(a)", transform=ax.transAxes, fontsize=15, fontweight="bold",
                 va="bottom", ha="right")
    fig.tight_layout()
    fig.savefig(save_stem + ".pdf")
    fig.savefig(save_stem + ".png")
    plt.close(fig)
    print(f"saved {save_stem}.pdf / .png")


render(False, os.path.join(OUT_DIR, "panel_a_fe_nonconvergence"))
render(True, os.path.join(OUT_DIR, "panel_a_fe_nonconvergence_lettered"))
print("DONE")
