"""Growth-prediction outlier analysis for Model C (v2, epoch 353): baseline
error metrics, how much of the total error is concentrated in a few
outlier simulations, a cross-check against the residual-mode-ratio finding
from step1_mode_decomposition_v2.py, how Models A and D perform on the
same worst simulations, an "agthroughout" ablation comparison, and where
the outlier simulations sit in parameter-space / trajectory-regime terms.

All of this reuses already-computed, cached data -- no new rollouts are
run here.

Reads:
  - deliverables/deliverables_results.json (final-step growth-area
    Ag_true/Ag_pred/median_abs_err/mean_abs_err/mean_signed_err for Models
    C_v2, A_v2, D_v2, and the C_v2_agthroughout ablation). Produced by
    running `python evaluation/generate_deliverables.py` from the
    repository root (with the FE dataset in place); run that first if it
    doesn't exist yet.
  - step1_mode_decomposition_v2_results.csv (from
    step1_mode_decomposition_v2.py, in this same folder), for the
    residual-mode-ratio cross-check.
  - step9_master_per_sim_table.csv and
    step6_failure_risk_scores_all_1000.csv, from the sibling
    supplementary_analyses/fe_nonconvergence_and_rom_accuracy/ folder
    (produced by that pipeline's step9_master_join.py and
    step6_failure_risk_scores.py), for locating the outliers in parameter
    space.

Writes: step2_outlier_summary.json (the baseline metrics and outlier
contribution numbers referenced in SI Section 3 / Figure 3).

Run from the repository root.
"""
import os
import json
import numpy as np
import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

deliverables = json.load(open(os.path.join("deliverables", "deliverables_results.json")))

# ---- Baseline metrics, direct from cached data ----
c = deliverables["C_v2"]
ag_true = np.array(c["Ag_true"]); ag_pred = np.array(c["Ag_pred"]); sim_ids = np.array(c["sim_id"])
abs_err = np.abs(ag_pred - ag_true)
tol = np.maximum(np.sqrt(ag_true), 5.0)
capture = 100 * (abs_err <= tol).mean()
print("=== Section 1: authoritative Model C (v2, epoch 353) growth performance ===")
print(f"  n={len(ag_true)}  median|err|={np.median(abs_err):.4f}  mean|err|={np.mean(abs_err):.4f}  "
      f"mean_signed={np.mean(ag_pred-ag_true):+.4f}  capture={capture:.1f}%  max|err|={abs_err.max():.4f}  "
      f"n(err>20)={int((abs_err>20).sum())}")
assert abs(np.median(abs_err) - c["median_abs_err"]) < 1e-6
assert abs(np.mean(abs_err) - c["mean_abs_err"]) < 1e-6
print("  cross-check vs deliverables_results.json: exact match")

# ---- Outlier contribution: how much of the total |Ag| error comes from
#      a handful of worst-case simulations? ----
order = np.argsort(-abs_err)
top_sids = sim_ids[order]
top_errs = abs_err[order]
total_abs_err_sum = abs_err.sum()
print("\n=== Section 4: outlier contribution ===")
print("Top 10 by |Ag error|:")
for sid, e in zip(top_sids[:10], top_errs[:10]):
    print(f"  sim {sid}: |err|={e:.2f}")
for k in [1, 2, 5]:
    contrib = 100 * top_errs[:k].sum() / total_abs_err_sum
    print(f"  top-{k} contribution to total sum(|err|): {contrib:.1f}%")
    rest = abs_err[order[k:]]
    print(f"    median|err| excluding top-{k}: {np.median(rest):.4f}   mean|err| excluding top-{k}: {np.mean(rest):.4f}")

# ---- Cross-check: do the worst Ag-error sims also have the worst
#      residual-mode ratio (R_res, computed in
#      step1_mode_decomposition_v2.py -- see that script for what R_res
#      means physically)? ----
mode_df = pd.read_csv(os.path.join(OUT_DIR, "step1_mode_decomposition_v2_results.csv"))
mode_df = mode_df.set_index("sim_id")
print("\n=== Cross-check: top-10 Ag-error sims' residual-mode ratio (final step) ===")
for sid, e in zip(top_sids[:10], top_errs[:10]):
    r = mode_df.loc[int(sid), "res_ratio_final"] if int(sid) in mode_df.index else np.nan
    print(f"  sim {sid}: |Ag err|={e:.2f}   residual-mode ratio={r:.3f}")

# ---- agthroughout ablation, exact cached values ----
print("\n=== Section 10: agthroughout ablation (already computed, reused) ===")
if "C_v2_agthroughout" in deliverables:
    ca = deliverables["C_v2_agthroughout"]
    print(f"  primary:      median={c['median_abs_err']:.4f} mean={c['mean_abs_err']:.4f} "
          f"mean_signed={c['mean_signed_err']:+.4f} capture={capture:.1f}%")
    ca_true = np.array(ca["Ag_true"]); ca_pred = np.array(ca["Ag_pred"])
    ca_abs = np.abs(ca_pred - ca_true)
    ca_tol = np.maximum(np.sqrt(ca_true), 5.0)
    ca_capture = 100 * (ca_abs <= ca_tol).mean()
    print(f"  agthroughout: median={ca['median_abs_err']:.4f} mean={ca['mean_abs_err']:.4f} "
          f"mean_signed={ca['mean_signed_err']:+.4f} capture={ca_capture:.1f}%")
    ca_sids = np.array(ca["sim_id"])
    ca_order = np.argsort(-ca_abs)
    print("  agthroughout top-5 worst sims:", list(zip(ca_sids[ca_order[:5]].tolist(),
                                                          ca_abs[ca_order[:5]].round(2).tolist())))
    overlap = set(top_sids[:10].tolist()) & set(ca_sids[ca_order[:10]].tolist())
    print(f"  overlap between primary's top-10 worst and agthroughout's top-10 worst: {len(overlap)}/10 -> {overlap}")
else:
    print("  [FLAG] C_v2_agthroughout key not found in deliverables_results.json")

# ---- Model A/D error on the same worst simulations (Model C's top-10) ----
print("\n=== Sections 8/9: Model A and D on Model C's worst sims ===")
a = deliverables["A_v2"]; d = deliverables["D_v2"]
a_map = {int(s): (t, p) for s, t, p in zip(a["sim_id"], a["Ag_true"], a["Ag_pred"])}
d_map = {int(s): (t, p) for s, t, p in zip(d["sim_id"], d["Ag_true"], d["Ag_pred"])}
for sid, e in zip(top_sids[:10], top_errs[:10]):
    sid = int(sid)
    at, ap = a_map.get(sid, (np.nan, np.nan))
    dt, dp = d_map.get(sid, (np.nan, np.nan))
    a_err = abs(ap - at); d_err = abs(dp - dt)
    print(f"  sim {sid}: C|err|={e:6.2f}   A|err|={a_err:6.2f} (true={at:.1f},pred={ap:.1f})   "
          f"D|err|={d_err:6.2f} (true={dt:.1f},pred={dp:.1f})")

# ---- Parameter-space / trajectory-regime location of C's worst sims ----
print("\n=== Sections 12/13: parameter-space / trajectory-regime location of C's worst sims ===")
conv_dir = os.path.join("supplementary_analyses", "fe_nonconvergence_and_rom_accuracy")
master = pd.read_csv(os.path.join(conv_dir, "step9_master_per_sim_table.csv")).set_index("sim_id")
risk = pd.read_csv(os.path.join(conv_dir, "step6_failure_risk_scores_all_1000.csv"))

worst10 = top_sids[:10].astype(int)
worst_rows = master.loc[master.index.intersection(worst10)]
print(worst_rows[["k1", "mu", "kappa", "Vf", "p_fail", "d_fail"]].describe() if "p_fail" in master.columns
      else worst_rows[["k1", "mu", "kappa", "Vf"]])
print("\nFull-population percentiles for comparison:")
for col in ["k1", "mu", "kappa", "Vf"]:
    pct = [float((master[col] < worst_rows[col].median()).mean() * 100)]
    print(f"  {col}: worst-10 median={worst_rows[col].median():.4f}  "
          f"(this value sits at the {pct[0]:.0f}th percentile of the full 927-sim population)")

results = dict(
    section1=dict(median=float(np.median(abs_err)), mean=float(np.mean(abs_err)),
                    mean_signed=float(np.mean(ag_pred-ag_true)), capture=float(capture),
                    max_err=float(abs_err.max())),
    top10_sims=[int(s) for s in top_sids[:10]],
    top10_errs=[float(e) for e in top_errs[:10]],
    contribution_top1=float(100*top_errs[:1].sum()/total_abs_err_sum),
    contribution_top2=float(100*top_errs[:2].sum()/total_abs_err_sum),
    contribution_top5=float(100*top_errs[:5].sum()/total_abs_err_sum),
)
with open(os.path.join(OUT_DIR, "step2_outlier_summary.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nsaved step2_outlier_summary.json")
print("DONE")
