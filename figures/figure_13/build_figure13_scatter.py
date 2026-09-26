"""Build Figure 13's four Ag (final net area gain) true-vs-predicted scatter
panels, one per model family A-D, using each model's revised main-paper
BEST checkpoint.

Each panel plots true vs. predicted Ag for all 185 validation simulations,
with the y=x identity line and a shaded "clinical tolerance" band
(+/- max(sqrt(Ag_true), 5) cm^2) marking which predictions count as
"captured". The title reports the identity-line R^2 (see NOTE below) and
the percentage of points inside the tolerance band. Axis limits are
independent per panel (auto-scaled to that model's own data, not shared
across panels).

NOTE on the R^2 definition used: this is the coefficient of determination
relative to the y=x identity line -- R^2 = 1 - SS_res/SS_tot, where
SS_res = sum((Ag_pred - Ag_true)^2) and SS_tot = sum((Ag_true - mean(Ag_true))^2).
This penalizes systematic bias and scale error (unlike (Pearson r)^2, which
would be blind to both), which matters here since the four models show
different systematic over/under-prediction biases.

Data source: deliverables/deliverables_results.json (produced by
evaluation/generate_deliverables.py), keys {A,B,C,D}_v2, computed from
each family's BEST checkpoint (checkpoints/Model_{fam}_v2_BEST.pt, per
the JSON's own `ckpt` field).

Read-only with respect to models/data. Writes into this script's own
directory (figures/figure_13/) via `__file__`.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = BASE
os.makedirs(OUT_DIR, exist_ok=True)

RESULTS_PATH = os.path.join("deliverables", "deliverables_results.json")
KEYS = {"A": "A_v2", "B": "B_v2", "C": "C_v2", "D": "D_v2"}


def scatter_panel(ag_true, ag_pred, save_stem, save_dpi=600):
    """Build and save one model family's Ag true-vs-predicted scatter panel
    (see module docstring for the tolerance band and R^2 definitions).
    Also prints the capture rate and median absolute error for a quick
    console summary alongside the saved figure."""
    ag_true = np.asarray(ag_true, dtype=np.float64)
    ag_pred = np.asarray(ag_pred, dtype=np.float64)
    tol = np.maximum(np.sqrt(ag_true), 5.0)
    within = np.abs(ag_pred - ag_true) <= tol
    capture_rate = 100 * within.mean()

    xmax = max(ag_true.max(), ag_pred.max()) * 1.05
    xs = np.linspace(0, xmax, 400)
    tol_line = np.maximum(np.sqrt(xs), 5.0)
    ss_res = float(np.sum((ag_pred - ag_true) ** 2))
    ss_tot = float(np.sum((ag_true - ag_true.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot  # identity-line (y=x) coefficient of determination; see module docstring

    fig = plt.figure(figsize=(7, 7))
    plt.plot(xs, xs, "b--", linewidth=1.5, zorder=2)
    plt.plot(xs, xs + tol_line, "r-", linewidth=3.0, zorder=2)
    plt.plot(xs, xs - tol_line, "r-", linewidth=3.0, zorder=2)
    plt.fill_between(xs, xs - tol_line, xs + tol_line, color="lightgreen", alpha=0.4, zorder=1)
    plt.scatter(ag_true, ag_pred, s=80, alpha=1.0, zorder=3)
    plt.xlim(0, xmax)
    plt.ylim(0, xmax)
    plt.xlabel("True final net area gain [cm$^2$]")
    plt.ylabel("Pred final net area gain [cm$^2$]")
    plt.title(f"Validation: Final net area gain (true vs pred) | r=9 | n={len(ag_true)}\n"
              f"R$^2$ = {r_squared:.3f} | capture rate = {capture_rate:.1f}%")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_stem + ".pdf", dpi=save_dpi)
    plt.savefig(save_stem + ".png", dpi=save_dpi)
    plt.close(fig)

    med = float(np.median(np.abs(ag_pred - ag_true)))
    print(f"saved {save_stem}.pdf / .png  capture_rate={capture_rate:.1f}%  median_abs_err={med:.4f}")


results = json.load(open(RESULTS_PATH))
for fam, key in KEYS.items():
    entry = results[key]
    save_stem = os.path.join(OUT_DIR, f"Model_{fam}_panel")
    scatter_panel(entry["Ag_true"], entry["Ag_pred"], save_stem)

print("DONE")
