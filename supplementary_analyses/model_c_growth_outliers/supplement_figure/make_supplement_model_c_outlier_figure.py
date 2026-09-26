"""Publication-ready supplement figure: Model C growth-prediction outliers
and the associated residual growth-POD amplification.

Reads already-computed artifacts from the Model C outlier investigation --
no new rollout, no POD basis refit, no recomputation of the underlying
analysis. Only the plotting/styling is new here. ("POD" is used here
rather than "PCA" for consistency with this repo's naming convention for
the reduced growth basis -- mathematically the same mean-centered SVD
operation as the paper's displacement POD basis.)

Reads:
  - model_c_growth_outliers/step1_mode_decomposition_v2_results.csv
      (sim_id, res_ratio_final, ag_abs_err for all 185 validation sims)
  - model_c_growth_outliers/step2_outlier_summary.json
      (top10_sims, used here only for the top-5 highlighted set)

Panel (a): dominant-mode norm ratio ||g_pred||/||g_true|| (final rollout
step, restricted to mode 1 of each growth channel; see
model_c_growth_outliers/step1_mode_decomposition_v2.py for the exact
definition) vs. |Ag error|, 185 validation sims, 5 worst highlighted in
red, with a Pearson r=0.02 / Spearman rho=-0.12 correlation annotation
(i.e. no meaningful correlation).
Panel (b): residual-mode norm ratio ||g_pred||/||g_true|| (final rollout
step, restricted to modes 2-8 of each growth channel) vs. |Ag error|, same
5 sims highlighted, with a Pearson r=0.870 / Spearman rho=0.521
correlation annotation. The contrast between (a) and (b) is the point of
this figure: the growth-area outliers are driven by amplification of the
higher-order residual growth modes, not by the dominant (large-scale)
growth pattern being wrong.

Writes (into this supplement_figure/ folder): fig_model_c_outliers.
{pdf,png} (combined 2-panel figure, plain and lettered variants) and
panel_a_model_c_dominant_modes / panel_b_model_c_residual_modes
(standalone panels, plain and lettered) -- the figure for SI Section 3 /
Figure 3.

Run from the repository root.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
CONV_DIR = os.path.dirname(BASE)  # model_c_growth_outliers/
OUT_DIR = BASE
os.makedirs(OUT_DIR, exist_ok=True)

mode_df = pd.read_csv(os.path.join(CONV_DIR, "step1_mode_decomposition_v2_results.csv"))
outlier_summary = json.load(open(os.path.join(CONV_DIR, "step2_outlier_summary.json")))
TOP5 = outlier_summary["top10_sims"][:5]  # [36, 847, 585, 48, 117]
assert TOP5 == [36, 847, 585, 48, 117], f"unexpected top-5 set: {TOP5}"

PEARSON_R = 0.870   # from step1_mode_decomposition_v2_summary.json: pearson_res_vs_ag_err
SPEARMAN_RHO = 0.52  # from step1_mode_decomposition_v2_summary.json: spearman_res_vs_ag_err = 0.5213, rounded to 2 d.p.
PEARSON_R_DOM = 0.02    # pearson_dom_vs_ag_err = 0.01635, rounded to 2 d.p.
SPEARMAN_RHO_DOM = -0.12  # spearman_dom_vs_ag_err = -0.12398, rounded to 2 d.p.

PANEL_RC = {
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.titlesize": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.0,
    "grid.alpha": 0.25,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    # Computer Modern math glyphs (bundled with matplotlib, no external
    # LaTeX install needed) instead of the default DejaVu Sans mathtext --
    # gives math-italic letters like g their usual closed-loop LaTeX shape.
    "mathtext.fontset": "cm",
}
plt.rcParams.update(PANEL_RC)

COLOR_OTHER = "#6E9BC7"   # muted blue-gray
COLOR_WORST = "#C0392B"   # red

mode_df = mode_df.sort_values("ag_abs_err", ascending=False).reset_index(drop=True)
is_worst = mode_df["sim_id"].isin(TOP5)


def draw_panel_a(ax, with_legend):
    ax.scatter(mode_df.loc[~is_worst, "dom_ratio_final"], mode_df.loc[~is_worst, "ag_abs_err"],
               s=28, color=COLOR_OTHER, alpha=0.7, edgecolors="none",
               label="Other validation simulations")
    ax.scatter(mode_df.loc[is_worst, "dom_ratio_final"], mode_df.loc[is_worst, "ag_abs_err"],
               s=90, color=COLOR_WORST, marker="X", linewidths=0.5, edgecolors="black",
               label=r"Five largest $A^g$ errors")
    ax.set_xlabel(r"$\|\mathbf{g}^{\mathrm{dom}}_{\mathrm{pred}}\| \, / \, \|\mathbf{g}^{\mathrm{dom}}_{\mathrm{true}}\|$")
    ax.set_ylabel(r"$|A^g_{\mathrm{pred}}-A^g_{\mathrm{true}}|$ (cm$^2$)")
    ax.set_title("Dominant growth-POD amplification")
    ax.grid(True, alpha=0.25)
    xmax = mode_df["dom_ratio_final"].max()
    ymax = mode_df["ag_abs_err"].max()
    ax.text(0.97 * xmax, 0.5 * ymax,
            rf"$r={PEARSON_R_DOM:.2f}$" + "\n" + rf"$\rho={SPEARMAN_RHO_DOM:.2f}$",
            ha="right", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.6", alpha=0.9))
    if with_legend:
        ax.legend(loc="upper center", fontsize=9, framealpha=0.9)


def draw_panel_b(ax, with_legend):
    ax.scatter(mode_df.loc[~is_worst, "res_ratio_final"], mode_df.loc[~is_worst, "ag_abs_err"],
               s=28, color=COLOR_OTHER, alpha=0.7, edgecolors="none",
               label="Other validation simulations")
    ax.scatter(mode_df.loc[is_worst, "res_ratio_final"], mode_df.loc[is_worst, "ag_abs_err"],
               s=90, color=COLOR_WORST, marker="X", linewidths=0.5, edgecolors="black",
               label=r"Five largest $A^g$ errors")
    ax.set_xlabel(r"$\|\mathbf{g}^{\mathrm{res}}_{\mathrm{pred}}\| \, / \, \|\mathbf{g}^{\mathrm{res}}_{\mathrm{true}}\|$")
    ax.set_ylabel(r"$|A^g_{\mathrm{pred}}-A^g_{\mathrm{true}}|$ (cm$^2$)")
    ax.set_title("Residual growth-POD amplification")
    ax.grid(True, alpha=0.25)
    xmax = mode_df["res_ratio_final"].max()
    ymax = mode_df["ag_abs_err"].max()
    ax.text(0.97 * xmax, 0.5 * ymax,
            rf"$r={PEARSON_R:.2f}$" + "\n" + rf"$\rho={SPEARMAN_RHO:.2f}$",
            ha="right", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.6", alpha=0.9))
    if with_legend:
        ax.legend(loc="upper center", fontsize=9, framealpha=0.9)


def make_combined(with_letters, out_stem):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    draw_panel_a(axes[0], with_legend=True)
    draw_panel_b(axes[1], with_legend=True)
    if with_letters:
        axes[0].text(-0.14, 1.05, "(a)", transform=axes[0].transAxes, fontsize=15,
                      fontweight="bold", va="bottom", ha="right")
        axes[1].text(-0.14, 1.05, "(b)", transform=axes[1].transAxes, fontsize=15,
                      fontweight="bold", va="bottom", ha="right")
    fig.tight_layout()
    fig.savefig(out_stem + ".pdf")
    fig.savefig(out_stem + ".png")
    plt.close(fig)
    print(f"saved {out_stem}.pdf / .png")


def make_single_panel(draw_fn, with_legend, out_stem, letter):
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    draw_fn(ax, with_legend)
    fig.tight_layout()
    fig.savefig(out_stem + ".pdf")
    fig.savefig(out_stem + ".png")
    plt.close(fig)
    print(f"saved {out_stem}.pdf / .png")
    if letter:
        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        draw_fn(ax, with_legend)
        ax.text(-0.14, 1.05, f"({letter})", transform=ax.transAxes, fontsize=15,
                  fontweight="bold", va="bottom", ha="right")
        fig.tight_layout()
        fig.savefig(out_stem + "_lettered.pdf")
        fig.savefig(out_stem + "_lettered.png")
        plt.close(fig)
        print(f"saved {out_stem}_lettered.pdf / .png")


make_combined(with_letters=False, out_stem=os.path.join(OUT_DIR, "fig_model_c_outliers"))
make_combined(with_letters=True, out_stem=os.path.join(OUT_DIR, "fig_model_c_outliers_lettered"))

make_single_panel(draw_panel_a, with_legend=True,
                   out_stem=os.path.join(OUT_DIR, "panel_a_model_c_dominant_modes"), letter="a")
make_single_panel(draw_panel_b, with_legend=True,
                   out_stem=os.path.join(OUT_DIR, "panel_b_model_c_residual_modes"), letter="b")

print("DONE")
