"""Archived development-time diagnostic (see ../README.md, "dev_audit_scripts/"):
an earlier draft of the Figure 9 panel composition -- 2x2 panels of median
validation displacement RMSE, mean pre-cap rollout loss, a derived "cap
attenuation" panel, and mean capped rollout loss, in linear- and log-axis
variants -- superseded by build_figure9_v2_final.py's different, final
4-panel composition (val-loss / disp-RMSE / pre-cap rollout / capped
rollout, all log-scaled). Not part of reproducing the published figure;
its outputs remain in this folder for provenance only.

Read-only with respect to models: no training, no checkpoint modification.
Only reads `*_loss_data.txt` and `disp_rmse_eval_log.txt` from each v2
directory (under the private-project-only `FINAL_models/` layout, which
does not exist in this repo) and writes new figure/CSV artifacts into this
directory.

Panel definitions (same schema across all 4 model families' training logs):
  (a) Median validation displacement RMSE -- parsed directly from each
      model's own disp_rmse_eval_log.txt ("epoch=N median_disp_rmse=X"
      lines only; "-> new BEST" continuation lines are skipped). Evaluated
      epochs only -- every 5 epochs during 41-300, every epoch 301-360.
      No interpolation of unevaluated epochs.
  (b) Mean pre-cap rollout loss = loss_data.txt column
      "rollout_plus_ag_pre_cap" (full-trajectory mean rollout loss, i.e.
      loss_raw + ag_addon, BEFORE the SIM_LOSS_CAP=0.033 soft cap).
  (c) Cap attenuation = rollout_plus_ag_pre_cap - rollout_capped -- a
      derived quantity (never logged directly by the training scripts)
      showing how much the soft cap reduced the rollout loss each epoch.
  (d) Mean capped rollout loss = loss_data.txt column "rollout_capped"
      (the value actually backpropagated, before multiplication by
      LAMBDA_ROLLOUT=1.8e-3).
"""
import os
import re
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE)

MODEL_DIRS = {
    "A": "Model_A_ablation_fullrollout_v2",
    "B": "Model_B_ablation_fullrollout_v2",
    "C": "Model_C_ablation_fullrollout_v2",
    "D": "Model_D_ablation_fullrollout_v2",
}
LOSS_DATA_PREFIX = {
    "A": "Model_A_Vanilla_r9",
    "B": "Model_B_Ag_r9",
    "C": "Model_C_PCA_r9",
    "D": "Model_D_CNN_r9",
}

# Consistent color/label convention used for Models A-D throughout the paper.
MODEL_COLORS = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green", "D": "tab:red"}
MODEL_LABELS = {"A": "Model A", "B": "Model B", "C": "Model C", "D": "Model D"}

# Font sizes/line widths matching the rest of the manuscript's figures.
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 2.2,
    "axes.linewidth": 1.0,
    "grid.linewidth": 0.8,
    "grid.alpha": 0.25,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})


def moving_average_nan(y, window=11):
    """Centered moving average over a window of `window` samples; NaN-aware
    (NaNs inside a window are dropped before averaging rather than
    propagating)."""
    y = np.asarray(y, dtype=float)
    out = np.full_like(y, np.nan, dtype=float)
    if window <= 1:
        return y.copy()
    half = window // 2
    n = len(y)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        vals = y[lo:hi]
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            out[i] = vals.mean()
    return out


def load_loss_data(fam):
    """Load one model family's per-epoch training log as a structured
    array with named columns (epoch, train_loss, val_loss, rollout_raw,
    rollout_plus_ag_pre_cap, rollout_capped, ...)."""
    path = os.path.join(PROJECT_ROOT, "FINAL_models", MODEL_DIRS[fam],
                         f"{LOSS_DATA_PREFIX[fam]}_loss_data.txt")
    return np.genfromtxt(path, delimiter=",", names=True)


def load_disp_rmse_log(fam):
    """Parse 'epoch=N median_disp_rmse=X' lines only; skip '-> new BEST' lines."""
    path = os.path.join(PROJECT_ROOT, "FINAL_models", MODEL_DIRS[fam], "disp_rmse_eval_log.txt")
    pat = re.compile(r"^epoch=(\d+)\s+median_disp_rmse=([\d.eE+\-]+)\s*$")
    epochs, vals = [], []
    with open(path) as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                epochs.append(int(m.group(1)))
                vals.append(float(m.group(2)))
    order = np.argsort(epochs)
    return np.asarray(epochs, dtype=float)[order], np.asarray(vals, dtype=float)[order]


def first_valid_epoch(histories, key):
    """Find the earliest epoch (across all model families) at which column
    `key` has a finite value -- used to trim the rollout-loss panels to the
    epoch range where the rollout loss term is actually active."""
    starts = []
    for fam, d in histories.items():
        y = d[key]
        finite = np.isfinite(y)
        if finite.any():
            starts.append(float(d["epoch"][finite][0]))
    return min(starts) if starts else None


loss_hist = {fam: load_loss_data(fam) for fam in "ABCD"}
disp_hist = {fam: load_disp_rmse_log(fam) for fam in "ABCD"}

print("Global-minimum disp-RMSE cross-check (must match the audit's known values):")
EXPECTED = {"A": (356, 0.133681), "B": (185, 0.288688), "C": (353, 0.197508), "D": (357, 0.143313)}
for fam in "ABCD":
    ep, val = disp_hist[fam]
    i = int(np.argmin(val))
    exp_ep, exp_val = EXPECTED[fam]
    match = "OK" if (int(ep[i]) == exp_ep and abs(val[i] - exp_val) < 1e-5) else "MISMATCH"
    print(f"  {fam}: epoch={int(ep[i])} median_disp_rmse={val[i]:.6f}  (expected epoch={exp_ep}, {exp_val:.6f})  [{match}]")


def check_positivity_for_log(panel_key, values_by_fam):
    """Report (never alter) zero/negative values before a column is put on a log axis."""
    any_issue = False
    for fam, y in values_by_fam.items():
        finite = y[np.isfinite(y)]
        n_zero = int(np.sum(finite == 0.0))
        n_neg = int(np.sum(finite < 0.0))
        if n_zero or n_neg:
            any_issue = True
            print(f"  [log-axis check] panel {panel_key}, model {fam}: "
                  f"n_zero={n_zero} n_negative={n_neg} min={finite.min():.6e} "
                  f"(out of {finite.size} finite values)")
    if not any_issue:
        mins = {fam: float(y[np.isfinite(y)].min()) for fam, y in values_by_fam.items()}
        print(f"  [log-axis check] panel {panel_key}: all values strictly positive "
              f"(per-model min: {mins})")
    return any_issue


def make_figure(log_panels, out_stem):
    """log_panels: set of letters among {'a','b','c','d'} to render with a log y-axis."""
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Panel (a): median validation displacement RMSE -- evaluated epochs only, no
    # smoothing. The eval cadence changes (every 5 epochs during 41-300, every
    # epoch 301-360), so an index-based moving average would mix points from very
    # different epoch spacings; we plot the raw evaluated points instead.
    for fam in "ABCD":
        ep, val = disp_hist[fam]
        ax_a.plot(ep, val, color=MODEL_COLORS[fam], marker="o", markersize=3.2,
                   linewidth=1.1, alpha=0.85, label=MODEL_LABELS[fam])
    ax_a.set_title("Median validation displacement RMSE")
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("Median displacement RMSE (cm)")
    ax_a.grid(True)
    if "a" in log_panels:
        ax_a.set_yscale("log")

    start_epoch = first_valid_epoch(loss_hist, "rollout_raw")

    panel_specs = [
        ("b", ax_b, "rollout_plus_ag_pre_cap", "Mean pre-cap rollout loss", "Rollout loss (pre-cap)", 11, 0.18),
        ("c", ax_c, "__cap_atten__", "Cap attenuation", "Pre-cap $-$ capped rollout loss", 11, 0.20),
        ("d", ax_d, "rollout_capped", "Mean capped rollout loss", "Rollout loss (capped)", 11, 0.18),
    ]
    for letter, ax, key, title, ylabel, window, raw_alpha in panel_specs:
        for fam in "ABCD":
            d = loss_hist[fam]
            x = d["epoch"].astype(float)
            if key == "__cap_atten__":
                y = d["rollout_plus_ag_pre_cap"] - d["rollout_capped"]
            else:
                y = d[key]
            mask = x >= start_epoch
            xx, yy = x[mask], y[mask]
            color = MODEL_COLORS[fam]
            ax.plot(xx, yy, color=color, alpha=raw_alpha, linewidth=1.0)
            ys = moving_average_nan(yy, window=window)
            ax.plot(xx, ys, color=color, alpha=1.0, linewidth=2.4, label=MODEL_LABELS[fam])
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        if letter in log_panels:
            ax.set_yscale("log")

    for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], "abcd"):
        ax.text(-0.14, 1.06, f"({letter})", transform=ax.transAxes,
                 fontsize=14, fontweight="bold", va="top", ha="left")

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True,
               bbox_to_anchor=(0.5, -0.015))

    plt.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(out_stem + ".pdf")
    fig.savefig(out_stem + ".png", dpi=600)
    plt.close(fig)
    print(f"saved {out_stem}.pdf / .png")


make_figure(log_panels=set(), out_stem=os.path.join(BASE, "figure9_v2_linear"))
make_figure(log_panels={"a"}, out_stem=os.path.join(BASE, "figure9_v2_log_panelA"))

print()
print("Positivity check for panels (b)/(c)/(d) before applying a log y-axis to all four panels:")
start_epoch = first_valid_epoch(loss_hist, "rollout_raw")
for panel_key, col in (("b", "rollout_plus_ag_pre_cap"), ("d", "rollout_capped")):
    vals_by_fam = {}
    for fam in "ABCD":
        d = loss_hist[fam]
        x = d["epoch"].astype(float)
        mask = x >= start_epoch
        vals_by_fam[fam] = d[col][mask]
    check_positivity_for_log(panel_key, vals_by_fam)
cap_vals_by_fam = {}
for fam in "ABCD":
    d = loss_hist[fam]
    x = d["epoch"].astype(float)
    mask = x >= start_epoch
    cap_vals_by_fam[fam] = (d["rollout_plus_ag_pre_cap"] - d["rollout_capped"])[mask]
cap_has_issue = check_positivity_for_log("c", cap_vals_by_fam)
if cap_has_issue:
    print("  [log-axis check] panel c (cap attenuation) contains exact zeros -- these are REAL")
    print("  (epochs where no sampled sim exceeded the soft cap that epoch, so pre-cap == capped")
    print("  exactly). Per instructions, these are NOT shifted/clipped/transformed. On a pure log")
    print("  axis, matplotlib cannot render non-positive points -- both the raw and (in stretches")
    print("  where an entire smoothing window is all-zero) the smoothed curve will show real gaps")
    print("  at those epochs for the affected models. This is disclosed, not silently handled.")

make_figure(log_panels={"a", "b", "c", "d"}, out_stem=os.path.join(BASE, "figure9_v2_log_all_panels"))

# Plotted-data CSV
csv_path = os.path.join(BASE, "figure9_v2_data.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["family", "epoch", "median_disp_rmse_if_evaluated",
                "rollout_raw", "rollout_plus_ag_pre_cap", "rollout_capped", "cap_attenuation"])
    for fam in "ABCD":
        d = loss_hist[fam]
        cap_atten = d["rollout_plus_ag_pre_cap"] - d["rollout_capped"]
        disp_ep, disp_val = disp_hist[fam]
        disp_map = {int(e): v for e, v in zip(disp_ep, disp_val)}
        for i, ep in enumerate(d["epoch"].astype(int)):
            w.writerow([
                fam, ep,
                disp_map.get(ep, ""),
                d["rollout_raw"][i],
                d["rollout_plus_ag_pre_cap"][i],
                d["rollout_capped"][i],
                cap_atten[i],
            ])
print(f"saved {csv_path}")
print("DONE")
