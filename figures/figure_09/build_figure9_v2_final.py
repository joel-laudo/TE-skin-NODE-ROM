"""Build the published Figure 9 (training diagnostics for Models A-D):
a 2x2 panel figure of (a) validation one-step loss, (b) median validation
displacement RMSE, (c) mean pre-cap rollout objective, and (d) mean capped
rollout objective, all vs. training epoch and all on a log y-axis, one
curve per model family A-D.

Read-only with respect to models: no training, no checkpoint modification.
Only reads each model family's per-epoch training logs --
`checkpoints/Model_{A,B,C,D}_v2_loss_data.txt` and
`checkpoints/Model_{A,B,C,D}_v2_disp_rmse_eval_log.txt` -- resolved relative
to this script's own directory (via `__file__`) so it runs correctly
regardless of the caller's working directory. Writes
`figure9_v2_final.pdf`/`.png` and `figure9_v2_final_data.csv` (the plotted
data) into this same directory.

Panel definitions (same schema/semantics across all 4 model families'
training logs):
  (a) Validation one-step loss = loss_data.txt column "val_loss": the
      one-step-ahead prediction MSE on the validation set, evaluated every
      epoch.
  (b) Median validation displacement RMSE -- parsed directly from each
      model's own disp_rmse_eval_log.txt ("epoch=N median_disp_rmse=X"
      lines only; "-> new BEST" continuation lines are skipped). Evaluated
      epochs only -- every 5 epochs during 41-300, every epoch 301-360.
      No interpolation of unevaluated epochs, no smoothing.
  (c) Mean pre-cap rollout objective = loss_data.txt column
      "rollout_plus_ag_pre_cap" (full-trajectory mean rollout loss, i.e.
      loss_raw + ag_addon, BEFORE the SIM_LOSS_CAP=0.033 soft cap applied
      during training to keep any single bad rollout from dominating the
      gradient).
  (d) Mean capped rollout objective = loss_data.txt column
      "rollout_capped" (the value actually backpropagated during training,
      before multiplication by the rollout-loss weight LAMBDA_ROLLOUT).
"""
import os
import re
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
# figures/figure_09 -> figures -> repository root
REPO_ROOT = os.path.dirname(os.path.dirname(BASE))

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
    """Centered moving average over a window of `window` samples, used to
    smooth the noisy per-epoch curves for display. NaN-aware: any NaNs
    inside a window are dropped before averaging rather than propagating,
    since some panels have short stretches without a valid value."""
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
    """Load one model family's ("A"/"B"/"C"/"D") per-epoch training log as a
    structured array with named columns (epoch, train_loss, val_loss,
    rollout_raw, rollout_plus_ag_pre_cap, rollout_capped, ...)."""
    path = os.path.join(REPO_ROOT, "checkpoints", f"Model_{fam}_v2_loss_data.txt")
    return np.genfromtxt(path, delimiter=",", names=True)


def load_disp_rmse_log(fam):
    """Parse 'epoch=N median_disp_rmse=X' lines only; skip '-> new BEST' lines."""
    path = os.path.join(REPO_ROOT, "checkpoints", f"Model_{fam}_v2_disp_rmse_eval_log.txt")
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
    `key` has a finite value. Used to trim the rollout-objective panels to
    the epoch range where the rollout loss term is actually active (it is
    NaN/unused before rollout training begins)."""
    starts = []
    for fam, d in histories.items():
        y = d[key]
        finite = np.isfinite(y)
        if finite.any():
            starts.append(float(d["epoch"][finite][0]))
    return min(starts) if starts else None


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

print()
print("Positivity check for all log-scaled panels:")
check_positivity_for_log("a (val_loss)", {fam: loss_hist[fam]["val_loss"] for fam in "ABCD"})
check_positivity_for_log("b (median_disp_rmse)", {fam: disp_hist[fam][1] for fam in "ABCD"})
start_epoch = first_valid_epoch(loss_hist, "rollout_raw")
for panel_key, col in (("c (rollout_plus_ag_pre_cap)", "rollout_plus_ag_pre_cap"),
                       ("d (rollout_capped)", "rollout_capped")):
    vals_by_fam = {}
    for fam in "ABCD":
        d = loss_hist[fam]
        x = d["epoch"].astype(float)
        mask = x >= start_epoch
        vals_by_fam[fam] = d[col][mask]
    check_positivity_for_log(panel_key, vals_by_fam)


def make_figure(out_stem):
    """Draw the 2x2 Figure 9 layout: one subplot per panel (a)-(d) described
    in the module docstring, each with all four model families A-D overlaid
    (color-coded per MODEL_COLORS) as a faint raw curve plus a bold smoothed
    curve, all on log y-axes. Saves both a .pdf and a .png at `out_stem`."""
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Panel (a): validation one-step loss -- full 1-360 range, dense per-epoch
    # data (no trimming to the rollout-training window), smoothed with a
    # window=9 moving average (narrower than the rollout metrics' window=11
    # since this curve is denser and less noisy).
    for fam in "ABCD":
        d = loss_hist[fam]
        x = d["epoch"].astype(float)
        y = d["val_loss"]
        color = MODEL_COLORS[fam]
        ax_a.plot(x, y, color=color, alpha=0.18, linewidth=1.0)
        ys = moving_average_nan(y, window=9)
        ax_a.plot(x, ys, color=color, alpha=1.0, linewidth=2.4, label=MODEL_LABELS[fam])
    ax_a.set_title("Validation one-step loss")
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("Validation one-step loss")
    ax_a.grid(True)
    ax_a.set_yscale("log")

    # Panel (b): median validation displacement RMSE -- evaluated epochs only, no
    # smoothing. The eval cadence itself changes (every 5 epochs during 41-300,
    # every epoch 301-360), so an index-based moving average would mix points
    # from very different epoch spacings and give a misleading result; we plot
    # the raw evaluated points instead.
    for fam in "ABCD":
        ep, val = disp_hist[fam]
        ax_b.plot(ep, val, color=MODEL_COLORS[fam], marker="o", markersize=3.2,
                   linewidth=1.1, alpha=0.85, label=MODEL_LABELS[fam])
    ax_b.set_title("Median validation displacement RMSE")
    ax_b.set_xlabel("Epoch")
    ax_b.set_ylabel("Median displacement RMSE (cm)")
    ax_b.grid(True)
    ax_b.set_yscale("log")

    start_epoch = first_valid_epoch(loss_hist, "rollout_raw")
    panel_specs = [
        (ax_c, "rollout_plus_ag_pre_cap", "Mean pre-cap rollout objective", "Rollout objective (pre-cap)", 11, 0.18),
        (ax_d, "rollout_capped", "Mean capped rollout objective", "Rollout objective (capped)", 11, 0.18),
    ]
    for ax, key, title, ylabel, window, raw_alpha in panel_specs:
        for fam in "ABCD":
            d = loss_hist[fam]
            x = d["epoch"].astype(float)
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


make_figure(out_stem=os.path.join(BASE, "figure9_v2_final"))

# Plotted-data CSV
csv_path = os.path.join(BASE, "figure9_v2_final_data.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["family", "epoch", "val_loss", "median_disp_rmse_if_evaluated",
                "rollout_plus_ag_pre_cap", "rollout_capped"])
    for fam in "ABCD":
        d = loss_hist[fam]
        disp_ep, disp_val = disp_hist[fam]
        disp_map = {int(e): v for e, v in zip(disp_ep, disp_val)}
        for i, ep in enumerate(d["epoch"].astype(int)):
            w.writerow([
                fam, ep,
                d["val_loss"][i],
                disp_map.get(ep, ""),
                d["rollout_plus_ag_pre_cap"][i],
                d["rollout_capped"][i],
            ])
print(f"saved {csv_path}")
print("DONE")
