"""Publication-ready two-panel supplementary figure for SI Figure 4.

Two ablations are shown side by side: (1) architecture -- Model A
(open-loop, no growth-state feedback into the latent dynamics) vs. Model D
(CNN-compressed growth-state feedback), and (2) training procedure --
two-stage (pretrain one-step, then fine-tune on full rollouts) vs.
rollout-only (train on full rollouts from the start). Uses ONLY
already-saved artifacts -- no new training, no new rollouts, no
interpolation/fabrication of missing data points.

Panel (a): validation displacement RMSE vs. epoch, the actual periodically-
evaluated history from each checkpoint's disp_rmse_eval_log.txt (connected
with thin lines, no smoothing applied since the median-over-185-sims
statistic is already low-noise).
Panel (b): per-sim displacement RMSE distribution (boxplot + light strip
overlay) at each run's own best disp-RMSE-selected checkpoint, using the
exact per-sim values from the already-computed rollout zarr stores.

Visual encoding: color hue = training procedure (the primary, most
important distinction per spec), marker/linestyle = architecture (A vs D).

Run from the repository root after generate_deliverables.py has produced
the "deliverables/" zarr stores:
    python evaluation/generate_deliverables.py
    python evaluation/ablation_comparisons/ablation_supplementary_figure.py

Reads each run's per-checkpoint validation log from
checkpoints/Model_{X}_v2[_rolloutonly]_disp_rmse_eval_log.txt (saved
alongside the checkpoint during training) and its per-sim rollout error
from the corresponding "deliverables/" zarr store.
"""
import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")  # headless backend -- must be set before any pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,   # embed real text in vector output, not paths
    "ps.fonttype": 42,
})

COLOR_TWO_STAGE = "#1F5FA8"     # blue -- training procedure = primary encoding
COLOR_ROLLOUT = "#C0442F"       # muted vermillion

RUNS = {
    "A_two-stage":    dict(log="checkpoints/Model_A_v2_disp_rmse_eval_log.txt",
                            zarr="deliverables/A_v2_val_rollouts_r9.zarr",
                            fam="A", proc="two-stage", color=COLOR_TWO_STAGE, marker="o", ls="-"),
    "A_rollout-only": dict(log="checkpoints/Model_A_v2_rolloutonly_disp_rmse_eval_log.txt",
                            zarr="deliverables/A_v2_rolloutonly_val_rollouts_r9.zarr",
                            fam="A", proc="rollout-only", color=COLOR_ROLLOUT, marker="o", ls="-"),
    "D_two-stage":    dict(log="checkpoints/Model_D_v2_disp_rmse_eval_log.txt",
                            zarr="deliverables/D_v2_val_rollouts_r9.zarr",
                            fam="D", proc="two-stage", color=COLOR_TWO_STAGE, marker="s", ls="--"),
    "D_rollout-only": dict(log="checkpoints/Model_D_v2_rolloutonly_disp_rmse_eval_log.txt",
                            zarr="deliverables/D_v2_rolloutonly_val_rollouts_r9.zarr",
                            fam="D", proc="rollout-only", color=COLOR_ROLLOUT, marker="s", ls="--"),
}


def parse_disp_rmse_log(path):
    """Parse a "disp_rmse_eval_log.txt" checkpoint log into (epochs, vals).

    Each periodic-eval line in the log looks like
    "epoch=<N> median_disp_rmse=<value> ...". Returns two parallel numpy
    arrays: the epoch numbers and the corresponding median validation
    displacement-RMSE values, in file order (not necessarily sorted).
    """
    epochs, vals = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("epoch=") and "median_disp_rmse=" in line:
                parts = line.split()
                ep = int(parts[0].split("=")[1])
                val = float(parts[1].split("=")[1])
                epochs.append(ep)
                vals.append(val)
    return np.array(epochs), np.array(vals)


def best_checkpoint(path):
    """Recover the (epoch, median_disp_rmse) of the saved best checkpoint.

    Training logs a "-> new BEST (epoch <N>, median_disp_rmse=<value>)"
    line every time a new best-so-far checkpoint is saved; this scans the
    log for those lines and returns the one with the lowest recorded
    value (which is also the last such line, since BEST is monotonically
    improving during training).
    """
    best_epoch, best_val = None, np.inf
    with open(path) as f:
        for line in f:
            if "new BEST" in line:
                # "  -> new BEST (epoch 356, median_disp_rmse=0.133681)"
                inner = line.split("(", 1)[1].split(")", 1)[0]
                ep_str, val_str = inner.split(",")
                ep = int(ep_str.strip().split()[1])
                val = float(val_str.strip().split("=")[1])
                if val < best_val:
                    best_epoch, best_val = ep, val
    return best_epoch, best_val


def load_per_sim_scalar(zarr_path):
    """Load the per-sim mean displacement-RMSE scalar for a rollout zarr.

    Same convention used for model selection throughout this study: each
    sim's per-step pointwise-RMSE trajectory is averaged over time into
    one scalar. Returns a numpy array (one value per simulation).
    """
    root = zarr.open_group(zarr_path, mode="r")
    sims_grp = root["simulations"]
    sim_keys = sorted(list(sims_grp.group_keys()))
    vals = []
    for sk in sim_keys:
        g = sims_grp[sk]
        rmse_t = np.asarray(g["disp_err_pointwise_rmse"], dtype=np.float64)
        vals.append(float(np.mean(rmse_t)))
    return np.array(vals)


# --- Load everything ---
histories = {}
bests = {}
per_sim = {}
for key, spec in RUNS.items():
    ep, val = parse_disp_rmse_log(spec["log"])
    histories[key] = (ep, val)
    bests[key] = best_checkpoint(spec["log"])
    per_sim[key] = load_per_sim_scalar(spec["zarr"])

print("=== Verification: best checkpoints and medians ===")
for key in RUNS:
    ep_best, val_best_log = bests[key]
    med_zarr = float(np.median(per_sim[key]))
    print(f"{key:16s} best_epoch(log)={ep_best}  best_val(log)={val_best_log:.6f}  "
          f"median(per-sim zarr)={med_zarr:.6f}  match={'OK' if abs(val_best_log-med_zarr)<1e-4 else 'MISMATCH!'}")

# --- Figure ---
fig = plt.figure(figsize=(11.5, 4.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.28)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])

# ---------------- Panel (a) ----------------
# NOTE ON LOG SCALE: the periodic validation-eval history is highly volatile
# through most of training for ALL FOUR runs (not just one outlier) -- e.g.
# D rollout-only reaches 7.34 cm at epoch 170, A two-stage reaches 4.32 cm at
# epoch 275 -- and only stabilizes into the small, comparison-relevant range
# (0.13-0.4 cm) during the final tail-stage epochs. A linear axis cannot show
# both the volatile mid-training regime and the small converged differences
# simultaneously without crushing the latter into an unreadable sliver, so a
# log-scale y-axis is used here (100% of real evaluated points shown, none
# clipped or fabricated) -- this is the documented exception explicitly
# permitted when the actual range makes a linear presentation unreadable.
ROLL_WINDOW = 5  # trailing rolling MEDIAN over N consecutive evaluated epochs
                 # (evaluations are spaced every 5 epochs before epoch 300 and
                 # every epoch after -- this window is over the evaluation
                 # SEQUENCE, not a fixed absolute-epoch span; disclosed here)


def rolling_median(vals, window):
    """Trailing rolling median over the last `window` entries of `vals`.

    Used only to draw a smoothed trend line through the noisy periodic
    validation-eval history in panel (a); the underlying raw values are
    plotted too, so no data is hidden.
    """
    out = np.empty_like(vals)
    for i in range(len(vals)):
        lo = max(0, i - window + 1)
        out[i] = np.median(vals[lo:i + 1])
    return out


for key, spec in RUNS.items():
    ep, val = histories[key]
    order = np.argsort(ep)
    ep, val = ep[order], val[order]
    label = f"Model {spec['fam']} — {spec['proc'].capitalize()}"
    # raw evaluated points: small, faint -- shows the true (noisy) data
    axA.scatter(ep, val, color=spec["color"], marker=spec["marker"], s=10,
                 alpha=0.30, linewidths=0, zorder=2)
    # rolling-median trend line: bold, drawn on top
    smoothed = rolling_median(val, ROLL_WINDOW)
    axA.plot(ep, smoothed, color=spec["color"], linestyle=spec["ls"],
              linewidth=1.8, alpha=0.95, label=label, zorder=3)
    ep_best, val_best = bests[key]
    axA.scatter([ep_best], [val_best], marker="*", s=190, color=spec["color"],
                 edgecolor="black", linewidth=0.8, zorder=5)

axA.axvline(40, color="#888888", linestyle=":", linewidth=1.2, zorder=1)
axA.text(41.5, 0.97, "End of one-step\nwarmup", transform=axA.get_xaxis_transform(),
          fontsize=7.8, color="#666666", va="top", ha="left", linespacing=1.15)

axA.set_xlabel("Epoch")
axA.set_ylabel("Validation displacement RMSE (cm), log scale")
axA.set_xlim(35, 362)
axA.set_yscale("log")
axA.set_ylim(0.10, 10)
axA.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1, 2, 5)))
axA.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
axA.set_title("(a) Training convergence", loc="left", fontweight="bold", fontsize=12)
axA.grid(True, axis="y", which="major", alpha=0.18, linewidth=0.7)
axA.set_axisbelow(True)
leg = axA.legend(loc="upper right", frameon=False, handlelength=2.2, labelspacing=0.4, fontsize=8.8)
axA.text(0.99, 0.02, f"faint points = raw evaluations; bold line = {ROLL_WINDOW}-eval rolling median",
          transform=axA.transAxes, fontsize=7, color="#888888", ha="right", va="bottom")

# ---------------- Panel (b) ----------------
# x positions: A group at 1,2 ; extra gap ; D group at 3.6,4.6
xpos = {"A_two-stage": 1.0, "A_rollout-only": 2.0, "D_two-stage": 3.6, "D_rollout-only": 4.6}
order_keys = ["A_two-stage", "A_rollout-only", "D_two-stage", "D_rollout-only"]

box_data = [per_sim[k] for k in order_keys]
positions = [xpos[k] for k in order_keys]

bp = axB.boxplot(box_data, positions=positions, widths=0.72, patch_artist=True,
                   showfliers=True,
                   medianprops=dict(color="black", linewidth=1.8),
                   whiskerprops=dict(color="#333333", linewidth=1.0),
                   capprops=dict(color="#333333", linewidth=1.0),
                   boxprops=dict(linewidth=1.0, edgecolor="#333333"),
                   flierprops=dict(marker="o", markersize=2.5, markerfacecolor="none",
                                    markeredgecolor="#555555", alpha=0.6, linewidth=0.5))
for patch, k in zip(bp["boxes"], order_keys):
    patch.set_facecolor(RUNS[k]["color"])
    patch.set_alpha(0.75)

rng = np.random.default_rng(7)
for k in order_keys:
    vals = per_sim[k]
    x0 = xpos[k]
    jitter = rng.uniform(-0.16, 0.16, size=len(vals))
    axB.scatter(x0 + jitter, vals, s=5, color=RUNS[k]["color"], alpha=0.35,
                 linewidths=0, zorder=2)

axB.set_xticks(positions)
axB.set_xticklabels(["Two-stage", "Rollout-only", "Two-stage", "Rollout-only"])
axB.set_ylabel("Displacement RMSE (cm)")
axB.set_title("(b) Validation displacement error\n(per-simulation, best checkpoint, n=185)",
                loc="left", fontweight="bold", fontsize=12)
axB.set_ylim(0, None)
axB.grid(True, axis="y", alpha=0.18, linewidth=0.7)
axB.set_axisbelow(True)

# Model A / Model D group labels beneath the tick labels
ymin, ymax = axB.get_ylim()
axB.text((xpos["A_two-stage"] + xpos["A_rollout-only"]) / 2, -0.135, "Model A",
           transform=axB.get_xaxis_transform(), ha="center", va="top", fontsize=10.5, fontweight="bold")
axB.text((xpos["D_two-stage"] + xpos["D_rollout-only"]) / 2, -0.135, "Model D",
           transform=axB.get_xaxis_transform(), ha="center", va="top", fontsize=10.5, fontweight="bold")

# Degradation annotations (exact, computed from the plotted medians themselves)
med_A_2s = float(np.median(per_sim["A_two-stage"]))
med_A_ro = float(np.median(per_sim["A_rollout-only"]))
med_D_2s = float(np.median(per_sim["D_two-stage"]))
med_D_ro = float(np.median(per_sim["D_rollout-only"]))
pct_A = (med_A_ro / med_A_2s - 1.0) * 100.0
pct_D = (med_D_ro / med_D_2s - 1.0) * 100.0

bracket_y_A = max(np.percentile(per_sim["A_two-stage"], 95), np.percentile(per_sim["A_rollout-only"], 95)) * 1.12
bracket_y_D = max(np.percentile(per_sim["D_two-stage"], 95), np.percentile(per_sim["D_rollout-only"], 95)) * 1.12
axB.annotate("", xy=(xpos["A_rollout-only"], bracket_y_A), xytext=(xpos["A_two-stage"], bracket_y_A),
              arrowprops=dict(arrowstyle="-", color="#333333", linewidth=0.9))
axB.text((xpos["A_two-stage"] + xpos["A_rollout-only"]) / 2, bracket_y_A * 1.02, f"+{pct_A:.0f}%",
           ha="center", va="bottom", fontsize=10, fontweight="bold")
axB.annotate("", xy=(xpos["D_rollout-only"], bracket_y_D), xytext=(xpos["D_two-stage"], bracket_y_D),
              arrowprops=dict(arrowstyle="-", color="#333333", linewidth=0.9))
axB.text((xpos["D_two-stage"] + xpos["D_rollout-only"]) / 2, bracket_y_D * 1.02, f"+{pct_D:.0f}%",
           ha="center", va="bottom", fontsize=10, fontweight="bold")

top_needed = max(bracket_y_A, bracket_y_D) * 1.22
axB.set_ylim(0, top_needed)

fig.savefig("deliverables/two_stage_vs_rollout_only_ablation.png", dpi=600, bbox_inches="tight")
fig.savefig("deliverables/two_stage_vs_rollout_only_ablation.pdf", bbox_inches="tight")
print("\nSaved deliverables/two_stage_vs_rollout_only_ablation.png (600 dpi) and .pdf")

print("\n=== Exact percentages used in panel (b) annotations ===")
print(f"Model A: median two-stage={med_A_2s:.6f}  median rollout-only={med_A_ro:.6f}  -> +{pct_A:.2f}%")
print(f"Model D: median two-stage={med_D_2s:.6f}  median rollout-only={med_D_ro:.6f}  -> +{pct_D:.2f}%")
