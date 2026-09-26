"""Manuscript Figure 14: publication-quality 2-row x 7-column spatial
parameter-effect figure ("Mechanical Influence of Growth", Section 4.3).
Uses ONLY the already-computed coefficient fields in
spatial_effect_maps.npz (produced by build_spatial_effect_maps.py) -- the
regression itself is not re-run, re-defined, or altered in any way.

Column order: growth parameters first (theta_crit, kk1, kk2), then
mechanical/boundary-condition parameters (mu, k1_fiber, kappa, tol) -- a
subtle visual gap separates the two groups. Row order: early (t=12.6) then
late (t=68.6). Color scale: matched WITHIN each row (not independently
autoscaled per panel), with a single labeled colorbar per row so the
shared/row-only scale is unambiguous -- cross-row comparison is not implied
by color alone.

Loads spatial_effect_maps.npz from this script's own directory (via
`__file__`) so it runs correctly regardless of the caller's working
directory.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend -- must be set before any pyplot import
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 16,
    "axes.labelsize": 17,
    "legend.fontsize": 14,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "spatial_effect_maps.npz"))

# (internal_key, display_label, group) -- group 0 = growth, 1 = mechanical/BC
PARAMS = [
    ("theta_crit", r"$\lambda^{\mathrm{crit}}$", 0),
    ("kk1",        r"$k_1^{g}$",                 0),
    ("kk2",        r"$k_2^{g}$",                 0),
    ("mu",         r"$\mu$",                     1),
    ("k1_fiber",   r"$k_1$",                     1),
    ("kappa",      r"$\kappa$",                  1),
    ("tol",        r"$\mathrm{tol}$",            1),
]

cmap = mpl.colormaps["magma"].copy()
cmap.set_bad("white")  # NaN (outside tissue domain) renders as clean white -- shows the true domain outline

vmax_early = max(np.nanmax(d[f"early_{k}"]) for k, _, _ in PARAMS)
vmax_late = max(np.nanmax(d[f"late_{k}"]) for k, _, _ in PARAMS)

fig = plt.figure(figsize=(17.5, 6.2))
# 7 param columns + 1 slim colorbar column, per row; extra gap after column 3 (growth|mechanical divider)
width_ratios = [1, 1, 1, 0.32, 1, 1, 1, 1, 0.62]
gs = fig.add_gridspec(2, 9, width_ratios=width_ratios, hspace=0.32, wspace=0.06,
                       left=0.105, right=0.965, top=0.82, bottom=0.05)

col_slots = [0, 1, 2, 4, 5, 6, 7]  # skip index 3 (the gap) and 8 (colorbar)

for row, (stage_key, stage_label, vmax) in enumerate([
    ("early", "Early\n($t=12.6$)", vmax_early),
    ("late", "Late\n($t=68.6$)", vmax_late),
]):
    im = None
    for slot, (pkey, plabel, group) in zip(col_slots, PARAMS):
        ax = fig.add_subplot(gs[row, slot])
        field = d[f"{stage_key}_{pkey}"]
        im = ax.imshow(field, cmap=cmap, origin="lower", vmin=0, vmax=vmax,
                        aspect="equal", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if row == 0:
            ax.set_title(plabel, fontsize=20, pad=10)
    cax = fig.add_subplot(gs[row, 8])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Effect (cm)", fontsize=16)
    cb.ax.tick_params(labelsize=13)
    # row label to the left of the first panel
    fig.text(0.012, gs[row, 0].get_position(fig).y0 + gs[row, 0].get_position(fig).height / 2,
              stage_label, fontsize=17, fontweight="bold", ha="left", va="center", linespacing=1.35)

# group headers above the top row
pos_growth_left = gs[0, 0].get_position(fig).x0
pos_growth_right = gs[0, 2].get_position(fig).x1
pos_mech_left = gs[0, 4].get_position(fig).x0
pos_mech_right = gs[0, 7].get_position(fig).x1
fig.text((pos_growth_left + pos_growth_right) / 2, 0.945, "Growth parameters",
          fontsize=18, fontweight="bold", ha="center", va="bottom")
fig.text((pos_mech_left + pos_mech_right) / 2, 0.945, "Mechanical / boundary-condition parameters",
          fontsize=18, fontweight="bold", ha="center", va="bottom")

fig.savefig(os.path.join(HERE, "mechanical_influence_growth_early_late.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(HERE, "mechanical_influence_growth_early_late.png"), dpi=600, bbox_inches="tight")
print("saved mechanical_influence_growth_early_late.pdf and .png")
