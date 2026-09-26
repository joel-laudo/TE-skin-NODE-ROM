"""Paired A-vs-D significance test supporting SI Figure 4: is the
growth-feedback ablation (Model A, open-loop / no growth feedback, vs.
Model D, CNN-compressed growth-state feedback) a real effect, or within
noise?

Compares the two models' validation displacement error across the same 185
validation sims, under each of two training procedures (two-stage vs.
rollout-only), using a paired bootstrap over sims (see
paired_bootstrap_delta below for the resampling scheme).

Per-sim scalar metric matches the exact model-selection convention used
throughout this study: mean over the per-step surface-displacement RMSE
trajectory, then compared across sims via the median. Uses already-computed
rollout zarrs (produced by evaluation/generate_deliverables.py) -- no new
rollouts, no retraining.

Run from the repository root after generate_deliverables.py has produced
the "deliverables/" zarr stores:
    python evaluation/generate_deliverables.py
    python evaluation/ablation_comparisons/paired_A_vs_D_bootstrap.py
"""
import json
import numpy as np
import zarr

RUNS = {
    "A_two-stage":    "deliverables/A_v2_val_rollouts_r9.zarr",
    "D_two-stage":    "deliverables/D_v2_val_rollouts_r9.zarr",
    "A_rollout-only": "deliverables/A_v2_rolloutonly_val_rollouts_r9.zarr",
    "D_rollout-only": "deliverables/D_v2_rolloutonly_val_rollouts_r9.zarr",
}

N_BOOTSTRAP = 5000
RNG_SEED = 12345  # fixed for reproducibility of this specific report


def load_per_sim_scalar(zarr_path):
    """Load the per-sim displacement-error scalar used for model selection.

    For each simulation stored in the rollout zarr at zarr_path, reads its
    per-step pointwise-RMSE trajectory and averages over time to get one
    scalar per sim. Returns a dict {sim_id: mean_disp_rmse}.
    """
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


data = {name: load_per_sim_scalar(path) for name, path in RUNS.items()}

for name, d in data.items():
    vals = np.array(list(d.values()))
    print(f"{name}: n={len(d)}  median={np.median(vals):.6f}  mean={np.mean(vals):.6f}")

# Confirm identical sim_id sets across all 4 runs (required for pairing)
sim_sets = [set(d.keys()) for d in data.values()]
common = set.intersection(*sim_sets)
all_union = set.union(*sim_sets)
print(f"\ncommon sim_ids across all 4 runs: {len(common)}  (union: {len(all_union)})")
assert len(common) == 185, f"expected 185 common sims, got {len(common)}"
sim_ids = np.array(sorted(common))

rng = np.random.default_rng(RNG_SEED)


def paired_bootstrap_delta(vals_D, vals_A, sim_ids, n_boot=N_BOOTSTRAP):
    """Paired bootstrap confidence interval for median(D) - median(A).

    vals_D and vals_A must be the same length and in matching sim order
    (index i in both arrays is the same sim_id) -- "paired" here means each
    bootstrap resample draws sim indices once and applies that same set of
    indices to both models, so the natural sim-to-sim difficulty variation
    cancels out and the resulting interval reflects only the A-vs-D effect.

    Returns (point_delta, ci_lo, ci_hi, boot_deltas): the observed
    median(D) - median(A) on the real data, the 95% bootstrap CI bounds
    (2.5th/97.5th percentiles of the resampled deltas), and the full array
    of n_boot resampled deltas.
    """
    n = len(sim_ids)
    point_delta = float(np.median(vals_D) - np.median(vals_A))
    idx_arr = np.arange(n)
    boot_deltas = np.empty(n_boot)
    for b in range(n_boot):
        resample = rng.choice(idx_arr, size=n, replace=True)
        boot_deltas[b] = np.median(vals_D[resample]) - np.median(vals_A[resample])
    ci_lo, ci_hi = np.percentile(boot_deltas, [2.5, 97.5])
    return point_delta, float(ci_lo), float(ci_hi), boot_deltas


results = {}
for cond in ["two-stage", "rollout-only"]:
    vals_A = np.array([data[f"A_{cond}"][sid] for sid in sim_ids])
    vals_D = np.array([data[f"D_{cond}"][sid] for sid in sim_ids])
    point_delta, ci_lo, ci_hi, boot_deltas = paired_bootstrap_delta(vals_D, vals_A, sim_ids)
    print(f"\n=== {cond}: Delta_feedback = median(D) - median(A) ===")
    print(f"  median(A)={np.median(vals_A):.6f}  median(D)={np.median(vals_D):.6f}")
    print(f"  point Delta = {point_delta:+.6f}")
    print(f"  95% CI (paired bootstrap, n_boot={N_BOOTSTRAP}) = [{ci_lo:+.6f}, {ci_hi:+.6f}]")
    print(f"  CI excludes zero: {not (ci_lo <= 0 <= ci_hi)}")
    results[cond] = {
        "median_A": float(np.median(vals_A)),
        "median_D": float(np.median(vals_D)),
        "mean_A": float(np.mean(vals_A)),
        "mean_D": float(np.mean(vals_D)),
        "point_delta_feedback": point_delta,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "ci_excludes_zero": bool(not (ci_lo <= 0 <= ci_hi)),
        "n_sims": int(len(sim_ids)),
        "n_bootstrap": N_BOOTSTRAP,
        "rng_seed": RNG_SEED,
    }

with open("deliverables/paired_A_vs_D_bootstrap_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nDONE -- wrote deliverables/paired_A_vs_D_bootstrap_results.json")
