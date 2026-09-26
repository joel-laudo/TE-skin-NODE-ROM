"""Build the Figure 9 supplementary table (figures/figure_09/README.md,
"Supplementary table" section): a 360-vs-560-epoch BEST-checkpoint
comparison between Model A and Model D, plus a paired A-vs-D bootstrap
significance test, using ONLY each run's own disp-RMSE-selected BEST
checkpoint for every row (both displacement and growth-parameter Ag
metrics).

Read-only: no training, no checkpoint modification. Reuses already-cached
zarr rollouts and JSON results -- no new rollout is run. Must be run from
the repository root, after both of the following have produced their
output in `deliverables/`:

    python evaluation/generate_deliverables.py
    python evaluation/ablation_comparisons/delayeddecay_eval_and_comparison.py

Writes `table_360_vs_560_best_results.json` into this script's own
directory (figures/figure_09/).
"""
import os
import json
import numpy as np
import zarr

BASE = os.path.dirname(os.path.abspath(__file__))

RNG_SEED = 12345
N_BOOT = 5000


def load_per_sim_disp_rmse(zarr_path):
    """Open a cached rollout zarr and return {sim_id: time-averaged
    displacement RMSE} for every simulation stored in it."""
    root = zarr.open_group(zarr_path, mode="r")
    sims_grp = root["simulations"]
    sim_keys = sorted(list(sims_grp.group_keys()))
    out = {}
    for sk in sim_keys:
        g = sims_grp[sk]
        sid = int(g.attrs["sim_id"])
        rmse_t = np.asarray(g["disp_err_pointwise_rmse"], dtype=np.float64)
        out[sid] = float(np.mean(rmse_t))
    return out


def ag_stats(ag_true, ag_pred):
    """Summarize growth-parameter (Ag) prediction error across simulations:
    median/mean absolute error, mean signed error (bias), the fraction of
    simulations "captured" within a per-simulation tolerance band, and
    counts of simulations exceeding several absolute-error thresholds."""
    ag_true = np.asarray(ag_true, dtype=np.float64)
    ag_pred = np.asarray(ag_pred, dtype=np.float64)
    err = ag_pred - ag_true
    abs_err = np.abs(err)
    # Tolerance band grows with sqrt(true Ag) (larger targets tolerate
    # proportionally larger absolute error) but is floored at 5.0 so very
    # small Ag values still get a meaningful band.
    tol = np.maximum(np.sqrt(ag_true), 5.0)
    within = abs_err <= tol
    return dict(
        median_abs_err=float(np.median(abs_err)),
        mean_abs_err=float(np.mean(abs_err)),
        mean_signed_err=float(np.mean(err)),
        capture_rate_pct=float(100 * within.mean()),
        n_err_gt_20=int(np.sum(abs_err > 20)),
        n_err_gt_50=int(np.sum(abs_err > 50)),
        n_err_gt_100=int(np.sum(abs_err > 100)),
        max_err=float(abs_err.max()),
    )


def paired_bootstrap(rmse_a_by_sim, rmse_d_by_sim, label):
    """Paired bootstrap test on the difference in median per-simulation
    displacement RMSE between Model D and Model A, restricted to the
    simulations both models were evaluated on (all 4 model families share
    the same 185-simulation validation split). Resamples simulation IDs
    with replacement N_BOOT times to build a 95% CI on
    median(D) - median(A); the sign/significance tells us whether adding
    feedback (D) measurably changes displacement accuracy relative to the
    open-loop baseline (A)."""
    common = sorted(set(rmse_a_by_sim) & set(rmse_d_by_sim))
    assert len(common) == 185, f"{label}: expected 185 common sims, got {len(common)}"
    a = np.array([rmse_a_by_sim[s] for s in common])
    d = np.array([rmse_d_by_sim[s] for s in common])
    point = float(np.median(d) - np.median(a))
    rng = np.random.default_rng(RNG_SEED)
    n = len(common)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        boots[i] = np.median(d[idx]) - np.median(a[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    excludes_zero = (lo > 0) or (hi < 0)
    print(f"[{label}] n_sims={n} median(A)={np.median(a):.6f} median(D)={np.median(d):.6f} "
          f"delta_feedback={point:+.6f}  95% CI=[{lo:+.6f}, {hi:+.6f}]  excludes_zero={excludes_zero}")
    return dict(point_estimate=point, ci_lo=float(lo), ci_hi=float(hi),
                excludes_zero=bool(excludes_zero), n_sims=n)


# ---- Load 360-epoch BEST-checkpoint displacement RMSE. "BEST" here means the
# checkpoint each training run itself selected as lowest-median-disp-RMSE
# (see dev_audit_scripts/verify_checkpoint_identity.py for the archived
# tensor-identity check confirming these cached rollouts really do correspond
# to that saved checkpoint and not some other epoch). ----
rmse_a360 = load_per_sim_disp_rmse("deliverables/A_v2_val_rollouts_r9.zarr")
rmse_d360 = load_per_sim_disp_rmse("deliverables/D_v2_val_rollouts_r9.zarr")
median_a360 = float(np.median(list(rmse_a360.values())))
median_d360 = float(np.median(list(rmse_d360.values())))
print(f"Cross-check vs disp_rmse_eval_log.txt: A_360_BEST median={median_a360:.6f} (expect 0.133681), "
      f"D_360_BEST median={median_d360:.6f} (expect 0.143313)")

# ---- Load 360-epoch BEST-checkpoint Ag stats from deliverables_results.json ----
deliverables_results = json.load(open("deliverables/deliverables_results.json"))
a360_ag_raw = deliverables_results["A_v2"]
d360_ag_raw = deliverables_results["D_v2"]
a360_ag = ag_stats(a360_ag_raw["Ag_true"], a360_ag_raw["Ag_pred"])
d360_ag = ag_stats(d360_ag_raw["Ag_true"], d360_ag_raw["Ag_pred"])
print(f"Cross-check vs deliverables_results.json: A_360_BEST Ag median={a360_ag['median_abs_err']:.6f} "
      f"(expect {a360_ag_raw['median_abs_err']:.6f}), D_360_BEST Ag median={d360_ag['median_abs_err']:.6f} "
      f"(expect {d360_ag_raw['median_abs_err']:.6f})")

# ---- Load 560-epoch delayed-decay BEST checkpoint results, computed on each
# run's own best checkpoint by
# evaluation/ablation_comparisons/delayeddecay_eval_and_comparison.py ----
delayeddecay = json.load(open("deliverables/delayeddecay560_results.json"))
a_delayed = delayeddecay["A_delayed"]
d_delayed = delayeddecay["D_delayed"]
# per_sim_disp_rmse is stored as a dict keyed by sim_id (string keys after JSON round-trip)
rmse_a_delayed = {int(k): float(v) for k, v in a_delayed["per_sim_disp_rmse"].items()}
rmse_d_delayed = {int(k): float(v) for k, v in d_delayed["per_sim_disp_rmse"].items()}

print()
print("=== Paired A-vs-D bootstrap, 360-epoch BEST checkpoints (epoch 356 vs epoch 357) ===")
boot_360 = paired_bootstrap(rmse_a360, rmse_d360, "360-epoch BEST")

print()
print("=== Paired A-vs-D bootstrap, 560-epoch delayed-decay BEST checkpoints (epoch 560 vs epoch 558) ===")
boot_560 = paired_bootstrap(rmse_a_delayed, rmse_d_delayed, "560-epoch delayed-decay BEST")

# ---- Final consistent table ----
rows = [
    ("A", "360 epochs", 356, median_a360, a360_ag["median_abs_err"], a360_ag["capture_rate_pct"],
     a360_ag["mean_abs_err"], a360_ag["mean_signed_err"]),
    ("D", "360 epochs", 357, median_d360, d360_ag["median_abs_err"], d360_ag["capture_rate_pct"],
     d360_ag["mean_abs_err"], d360_ag["mean_signed_err"]),
    ("A", "560 epochs, delayed decay", 560, a_delayed["disp_rmse_median"], a_delayed["ag_median_abs_err"],
     a_delayed["ag_capture_rate_pct"], a_delayed["ag_mean_abs_err"], a_delayed["ag_mean_signed_err"]),
    ("D", "560 epochs, delayed decay", 558, d_delayed["disp_rmse_median"], d_delayed["ag_median_abs_err"],
     d_delayed["ag_capture_rate_pct"], d_delayed["ag_mean_abs_err"], d_delayed["ag_mean_signed_err"]),
]

print()
print("=== Final consistent supplement table (all metrics from the SAME BEST checkpoint per row) ===")
print(f"{'Model':6s} {'Schedule':28s} {'Epoch':>6s} {'DispRMSE':>10s} {'Ag|err|med':>11s} {'Capture%':>9s} {'Ag|err|mean':>12s} {'AgSigned':>10s}")
for r in rows:
    print(f"{r[0]:6s} {r[1]:28s} {r[2]:6d} {r[3]:10.6f} {r[4]:11.4f} {r[5]:9.1f} {r[6]:12.4f} {r[7]:10.4f}")

out = dict(
    rows=[dict(model=r[0], schedule=r[1], epoch=r[2], median_disp_rmse=r[3], ag_median_abs_err=r[4],
               capture_rate_pct=r[5], ag_mean_abs_err=r[6], ag_mean_signed_err=r[7]) for r in rows],
    bootstrap_360_best=boot_360,
    bootstrap_560_delayed_best=boot_560,
)
out_path = os.path.join(BASE, "table_360_vs_560_best_results.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print()
print(f"saved {out_path}")
print("DONE")
