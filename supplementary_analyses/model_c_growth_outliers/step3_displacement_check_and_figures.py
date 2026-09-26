"""Displacement-RMSE trajectory for Model C's worst growth-error
simulations (does displacement stay accurate while growth error blows
up?), plus two figures: the sorted Ag-error distribution highlighting the
worst cases, and the residual-mode-ratio vs. Ag-error scatter -- the
mechanistic figure linking step1_mode_decomposition_v2.py's R_res
diagnostic to the actual growth-area error.

Reads:
  - deliverables/C_v2_val_rollouts_r9.zarr (Model C's per-simulation
    pointwise displacement RMSE trajectory, disp_err_pointwise_rmse) and
    deliverables/deliverables_results.json (Ag_true/Ag_pred). Both are
    produced by running `python evaluation/generate_deliverables.py` from
    the repository root (with the FE dataset in place); run that first if
    they don't exist yet.
  - step2_outlier_summary.json (from step2_outlier_and_crosscheck.py, in
    this same folder), for the worst-error simulation ids.
  - step1_mode_decomposition_v2_results.csv (from
    step1_mode_decomposition_v2.py), for the residual-mode ratio (R_res).

Writes:
  - step3_displacement_check.csv: mean/final displacement RMSE for the 5
    worst growth-error simulations
  - fig_outliers_and_mechanism.{pdf,png}: the sorted Ag-error distribution
    and R_res-vs-Ag-error figures for SI Section 3 / Figure 3

Run from the repository root.
"""
import os
import json
import numpy as np
import pandas as pd
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

root_roll = zarr.open_group(os.path.join("deliverables", "C_v2_val_rollouts_r9.zarr"), mode="r")
sims_grp = root_roll["simulations"]

worst = json.load(open(os.path.join(OUT_DIR, "step2_outlier_summary.json")))
top5 = worst["top10_sims"][:5]
top5_errs = worst["top10_errs"][:5]

print("=== Section 5B: displacement RMSE (mean-over-trajectory) for Model C's worst growth-error sims ===")
disp_rows = []
for sid in top5:
    key = None
    for sk in sims_grp.group_keys():
        if int(sims_grp[sk].attrs["sim_id"]) == sid:
            key = sk
            break
    g = sims_grp[key]
    rmse_t = np.asarray(g["disp_err_pointwise_rmse"])
    disp_rows.append(dict(sim_id=sid, e_disp_mean=float(np.mean(rmse_t)), e_disp_final=float(rmse_t[-1])))
    print(f"  sim {sid}: mean disp-RMSE={np.mean(rmse_t):.4f} cm, final={rmse_t[-1]:.4f} cm "
          f"(population median for this checkpoint: ~0.198 cm)")
pd.DataFrame(disp_rows).to_csv(os.path.join(OUT_DIR, "step3_displacement_check.csv"), index=False)

# --- Figure 1: sorted Ag-error distribution, worst cases highlighted ---
c = json.load(open(os.path.join("deliverables", "deliverables_results.json")))["C_v2"]
ag_true = np.array(c["Ag_true"]); ag_pred = np.array(c["Ag_pred"]); sim_ids = np.array(c["sim_id"])
abs_err = np.abs(ag_pred - ag_true)
order = np.argsort(-abs_err)

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                       "savefig.dpi": 600, "savefig.bbox": "tight"})

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
ax = axes[0]
sorted_err = abs_err[order]
colors = ["tab:red" if i < 5 else "steelblue" for i in range(len(sorted_err))]
ax.bar(range(len(sorted_err)), sorted_err, color=colors, width=1.0)
ax.set_xlabel("Validation sim, sorted by |Ag error| (descending)")
ax.set_ylabel(r"$|A^g_{pred}-A^g_{true}|$ (cm$^2$)")
ax.set_title("Model C: sorted validation Ag-error distribution\n(5 worst cases in red)")
ax.grid(True, alpha=0.25)

mode_df = pd.read_csv(os.path.join(OUT_DIR, "step1_mode_decomposition_v2_results.csv")).set_index("sim_id")
ag_err_by_sim = {int(s): e for s, e in zip(sim_ids, abs_err)}
ax = axes[1]
x = mode_df["res_ratio_final"].values
y = np.array([ag_err_by_sim.get(sid, np.nan) for sid in mode_df.index])
is_top5 = np.isin(mode_df.index, top5)
ax.scatter(x[~is_top5], y[~is_top5], s=25, alpha=0.5, color="steelblue", label="other validation sims")
ax.scatter(x[is_top5], y[is_top5], s=60, color="tab:red", marker="X", label="5 worst Ag-error sims")
ax.set_xlabel("Residual-mode (2-8) ratio, ||pred||/||true|| at final step")
ax.set_ylabel(r"$|A^g_{pred}-A^g_{true}|$ (cm$^2$)")
ax.set_title("Residual growth-PCA-mode blow-up vs. Ag error\n(Spearman rho=0.52, Pearson r=0.87)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_outliers_and_mechanism.pdf"))
fig.savefig(os.path.join(OUT_DIR, "fig_outliers_and_mechanism.png"))
plt.close(fig)
print("\nsaved fig_outliers_and_mechanism.{pdf,png}")
print("DONE")
