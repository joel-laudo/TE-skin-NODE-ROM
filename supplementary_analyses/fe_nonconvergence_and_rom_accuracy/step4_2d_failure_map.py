"""2D failure maps with a smooth predicted-probability contour overlay.

Two feature pairs are visualized: (a) k1 x mu, the pair with the strongest
linear interaction (see step2_classifier.py), and (b) kappa x mu. Kappa's
relationship with failure is 1D/non-monotonic on its own (see
step2b_investigate_kappa_discrepancy.py and step3_marginal_panel.py), so
pairing it with mu -- its next-most relevant partner per the held-out
permutation importance ranking -- gives a genuine 2D view of its effect.

For each pair, a small 2-feature random forest (fit only for this
visualization, separate from the full classifier in step2_classifier.py)
is trained on all 1000 LHS points, and its predicted failure probability is
contoured over a fine grid with the actual converged/non-converged points
overlaid.

Reads: all_1000_jobs_with_convergence_flag.csv (written by
step1_recover_failed_jobs.py).

Writes: fig_2d_failure_map_k1_mu.{pdf,png} and
fig_2d_failure_map_kappa_mu.{pdf,png} -- the 2D failure-probability maps
referenced in SI Section 2 / Figure 2.

Run from the repository root; all paths resolve relative to this script.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(OUT_DIR, "all_1000_jobs_with_convergence_flag.csv"))
y = (~df["converged"]).astype(int).values

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                       "savefig.dpi": 600, "savefig.bbox": "tight"})


def make_2d_map(xcol, ycol, save_stem):
    X2 = df[[xcol, ycol]].values
    rf2 = RandomForestClassifier(n_estimators=500, max_depth=4, class_weight="balanced",
                                   random_state=123, n_jobs=-1)
    rf2.fit(X2, y)

    xg = np.linspace(df[xcol].min(), df[xcol].max(), 200)
    yg = np.linspace(df[ycol].min(), df[ycol].max(), 200)
    XG, YG = np.meshgrid(xg, yg)
    grid = np.column_stack([XG.ravel(), YG.ravel()])
    proba = rf2.predict_proba(grid)[:, 1].reshape(XG.shape)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    cf = ax.contourf(XG, YG, proba, levels=15, cmap="Reds", alpha=0.75)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Predicted P(non-convergence)")

    conv_mask = df["converged"].values.astype(bool)
    ax.scatter(df.loc[conv_mask, xcol], df.loc[conv_mask, ycol],
               s=18, color="steelblue", alpha=0.6, label=f"converged (n={conv_mask.sum()})",
               edgecolors="none")
    ax.scatter(df.loc[~conv_mask, xcol], df.loc[~conv_mask, ycol],
               s=45, color="black", marker="x", linewidths=1.5,
               label=f"non-converged (n={(~conv_mask).sum()})")
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(f"FE non-convergence probability: {xcol} vs {ycol}\n(2-feature random forest, all 1000 LHS points)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_stem + ".pdf")
    fig.savefig(save_stem + ".png")
    plt.close(fig)
    print(f"saved {save_stem}.pdf / .png")


make_2d_map("k1", "mu", os.path.join(OUT_DIR, "fig_2d_failure_map_k1_mu"))
make_2d_map("kappa", "mu", os.path.join(OUT_DIR, "fig_2d_failure_map_kappa_mu"))
print("DONE")
