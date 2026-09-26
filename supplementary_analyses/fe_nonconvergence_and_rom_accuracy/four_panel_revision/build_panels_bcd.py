"""4-panel supplement, Panels (b)-(d): kappa-binned error trends, split
into standalone panels and restyled for publication. Uses the same bin
medians as step10_kappa_bins.csv (produced by
step10_error_vs_risk_analysis.py; step11_figures.py's
fig2_kappa_bin_error_trend.png is the equivalent combined figure) -- no new
analysis. One addition: an IQR-band alternate version of each panel,
computed directly from the already-loaded step9 master table (25th/75th
percentile per bin) -- a trivial re-aggregation, not a new statistical
analysis.

Color convention: Model A = tab:blue, Model D = tab:red, matching this
repo's manuscript-wide model color convention (different from the
orange/green scheme used in step11_figures.py's fig2, which predates that
convention). POD is not one of the paper's architecture models, so it uses
a neutral color (black) to avoid implying any A/B/C/D association.

Reads: step9_master_per_sim_table.csv (from step9_master_join.py) and
step10_kappa_bins.csv (from step10_error_vs_risk_analysis.py), both found
in this script's parent directory (fe_nonconvergence_and_rom_accuracy/,
via CONV_DIR).

Writes (into this four_panel_revision/ folder): kappa_bin_iqr.csv (the
newly computed IQR bands) plus panel_b_pod_kappa / panel_c_modelA_kappa /
panel_d_modelD_kappa, each in plain, lettered, and IQR-band variants --
combined with panel (a) into the 4-panel supplement figure.

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
kappa_df = pd.read_csv(os.path.join(CONV_DIR, "step10_kappa_bins.csv"))

# Re-derive per-bin IQR (25th/75th pct) directly from the master table, using
# the exact same bin edges (qcut, q=8) already used for the medians.
master["kappa_bin"] = pd.qcut(master["kappa"], q=8, duplicates="drop")
val = master[master["is_validation"]]

iqr_rows = []
for bin_label, g in master.groupby("kappa_bin", observed=True):
    gv = g[g["is_validation"]]
    iqr_rows.append(dict(
        kappa_mid=g["kappa"].mean(),
        pod_p25=g["e_pod"].quantile(0.25), pod_p75=g["e_pod"].quantile(0.75),
        a_p25=gv["e_node_A"].quantile(0.25) if len(gv) else np.nan,
        a_p75=gv["e_node_A"].quantile(0.75) if len(gv) else np.nan,
        d_p25=gv["e_node_D"].quantile(0.25) if len(gv) else np.nan,
        d_p75=gv["e_node_D"].quantile(0.75) if len(gv) else np.nan,
    ))
iqr_df = pd.DataFrame(iqr_rows).sort_values("kappa_mid")
iqr_df.to_csv(os.path.join(OUT_DIR, "kappa_bin_iqr.csv"), index=False)

PANELS = [
    ("b", "median_e_pod", "pod_p25", "pod_p75", "black", "POD reconstruction error",
     "panel_b_pod_kappa"),
    ("c", "median_e_node_A", "a_p25", "a_p75", "tab:blue", "Model A displacement error",
     "panel_c_modelA_kappa"),
    ("d", "median_e_node_D", "d_p25", "d_p75", "tab:red", "Model D displacement error",
     "panel_d_modelD_kappa"),
]


def render(letter, med_col, p25_col, p75_col, color, title, stem, with_iqr, with_letter):
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    x = kappa_df["kappa_mid"]
    y = kappa_df[med_col]
    if with_iqr:
        ax.fill_between(iqr_df["kappa_mid"], iqr_df[p25_col], iqr_df[p75_col],
                         color=color, alpha=0.15, label="IQR (25th-75th pct.)")
    ax.plot(x, y, marker="o", color=color, linewidth=2.2)
    ax.set_xlabel(r"Fiber dispersion, $\kappa$")
    ax.set_ylabel("Median RMSE (cm)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if with_iqr:
        ax.legend(loc="upper center", fontsize=9)
    if with_letter:
        ax.text(-0.12, 1.04, f"({letter})", transform=ax.transAxes, fontsize=15, fontweight="bold",
                 va="bottom", ha="right")
    fig.tight_layout()
    suffix = "_iqr" if with_iqr else ""
    suffix += "_lettered" if with_letter else ""
    fig.savefig(os.path.join(OUT_DIR, stem + suffix + ".pdf"))
    fig.savefig(os.path.join(OUT_DIR, stem + suffix + ".png"))
    plt.close(fig)
    print(f"saved {stem}{suffix}.pdf / .png")


for letter, med_col, p25_col, p75_col, color, title, stem in PANELS:
    render(letter, med_col, p25_col, p75_col, color, title, stem, with_iqr=False, with_letter=False)
    render(letter, med_col, p25_col, p75_col, color, title, stem, with_iqr=False, with_letter=True)
    render(letter, med_col, p25_col, p75_col, color, title, stem, with_iqr=True, with_letter=False)

print("DONE")
