"""Does Model D's NODE dynamics actually USE h_g (the CNN-encoded growth
feature), or is the growth-feedback pathway functionally inert? Section 2A
showed h_g faithfully encodes the growth state (growth_decoding.py);
this script tests whether that information actually influences the
predicted rollout, by substituting something else for h_g at every step and
seeing whether accuracy degrades.

Runs the frozen (no retraining) Model D checkpoint through the
intervention-capable rollout (rollout_cnn_node_intervened.py) at all 185
validation sims, under:
  - identity (sanity-check control -- must reproduce the baseline
    displacement RMSE of 0.143313 reported for Model D)
  - zero: replace h_g with the zero vector
  - mean: replace h_g with the training-set mean h_g
  - shuffle: substitute another (random "companion") simulation's h_g at a
    matching point in its trajectory -- progress-matched / volume-matched /
    fully-random, each repeated over 20 random companion-sim permutations.
    Matching by progress or volume (rather than pairing fully at random)
    keeps the substituted feature at a comparable growth stage to the true
    one, so a large error increase can be attributed to h_g's specific
    per-simulation content rather than merely to feeding in a vector of the
    wrong overall scale.

Reports median/mean displacement RMSE, final Ag error, Ag capture rate, and
a paired bootstrap (5000 resamples, simulation as the independent unit) of
every intervention vs. baseline (identity) Model D.

NOTE: must be run with the working directory set to the repository root
(where the FE dataset from data/DATA_AVAILABILITY.md has been placed) --
this script does not chdir. It adds evaluation/ to sys.path relative to its
own file location.
"""
import os
import sys
import json
import time

import numpy as np
import zarr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")
sys.path.insert(0, EVAL_DIR)
sys.path.insert(0, SCRIPT_DIR)

import eval_model_a as ma  # noqa: E402
from shared_eval_utils import compute_surface_disp_rmse_trajectory  # noqa: E402
from rollout_cnn_node_intervened import (  # noqa: E402
    rollout_cnn_node_intervened, identity_intervention, zero_intervention,
    make_mean_intervention, make_shuffle_intervention, build_constrained_companion_map,
)

t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


R_MODES = 9
A0_CM2 = 0.25
NODES_CSV = "GOH_Nodes_Test_for_Visualization.csv"
ELEMS_CSV = "GOH_Elements_Test_for_Visualization.csv"
D_CKPT = "checkpoints/Model_D_v2_BEST.pt"
D_VAL_ZARR = "deliverables/D_v2_val_rollouts_r9.zarr"
N_SHUFFLE_REPEATS = 20

OUT_DIR = SCRIPT_DIR

val_features = dict(np.load(os.path.join(OUT_DIR, "val_features.npz")))
train_features = dict(np.load(os.path.join(OUT_DIR, "train_features.npz")))
h_mean = train_features["cnn_final_features"].mean(axis=0)

val_sims = np.asarray(ma.val_sims, dtype=np.int64)
log(f"n_val_sims={len(val_sims)}")

# --- true displacement + true Ag, reused from the already-cached zarr / helper ---
root = zarr.open_group(D_VAL_ZARR, mode="r")
sims_grp = root["simulations"]
u_true_by_sim = {}
for sk in sims_grp.group_keys():
    g = sims_grp[sk]
    u_true_by_sim[int(g.attrs["sim_id"])] = np.asarray(g["u_true_nodes"])

(_, _, sim_index, time_vals, _, _, _) = ma.load_raw_data(R_MODES)
H, W, _ = ma.get_bottom_grid_cache()
ag_true_by_sim = {}
for sid in val_sims:
    ag_true_by_sim[int(sid)] = float(ma.compute_true_net_area_gain_final_for_sim(
        sim_id=int(sid), sim_index=sim_index, time_vals=time_vals,
        H=H, W=W, zarr_sdv_path=ma.ZARR_SDV, folder_x="snaps_SDV1", folder_y="snaps_SDV4", A0_cm2=A0_CM2,
    ))


def run_condition(name, intervention_fn):
    per_sim_disp_rmse = np.zeros(len(val_sims))
    per_sim_ag_err = np.zeros(len(val_sims))
    for i, sid in enumerate(val_sims):
        roll = rollout_cnn_node_intervened(
            r_modes=R_MODES, sim_id=int(sid), nodes_csv=NODES_CSV, elems_csv=ELEMS_CSV,
            intervention_fn=intervention_fn, ckpt_path_template=D_CKPT,
        )
        rmse_t = compute_surface_disp_rmse_trajectory(roll["u_pred_nodes"], u_true_by_sim[int(sid)])
        per_sim_disp_rmse[i] = np.mean(rmse_t)
        per_sim_ag_err[i] = abs(roll["area_gain_pred_final_cm2"] - ag_true_by_sim[int(sid)])
    return per_sim_disp_rmse, per_sim_ag_err


def paired_bootstrap(delta, n_resamples=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(delta)
    boot_medians = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, n)
        boot_medians[b] = np.median(delta[idx])
    lo, hi = np.percentile(boot_medians, [2.5, 97.5])
    return float(np.median(delta)), float(lo), float(hi)


results = {}
RESULTS_PATH = os.path.join(OUT_DIR, "intervention_sweep_results.json")
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    log(f"resuming: found existing results for {list(results.keys())}")

# --- fixed conditions ---
fixed_conditions = {
    "identity": identity_intervention,
    "zero": zero_intervention,
    "mean": make_mean_intervention(h_mean),
}
for name, fn in fixed_conditions.items():
    if name in results:
        log(f"=== {name}: already done, skipping ===")
        continue
    log(f"=== running condition: {name} ===")
    disp_rmse, ag_err = run_condition(name, fn)
    results[name] = {
        "disp_rmse_per_sim": disp_rmse.tolist(),
        "ag_err_per_sim": ag_err.tolist(),
        "disp_rmse_median": float(np.median(disp_rmse)),
        "disp_rmse_mean": float(np.mean(disp_rmse)),
        "ag_median_abs_err": float(np.median(ag_err)),
        "ag_mean_abs_err": float(np.mean(ag_err)),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log(f"{name}: disp_rmse_median={results[name]['disp_rmse_median']:.4f}  "
        f"ag_median={results[name]['ag_median_abs_err']:.4f}")

if results["identity"]["disp_rmse_median"] > 0.15:
    log("!!! WARNING: identity condition did not reproduce ~0.1433 baseline -- STOP and debug before trusting anything below !!!")

# --- shuffle conditions: 3 schemes x N_SHUFFLE_REPEATS companion permutations ---
sim_list = val_sims.tolist()
for scheme in ("progress", "volume", "random"):
    key = f"shuffle_{scheme}"
    if key in results:
        log(f"=== {key}: already done, skipping ===")
        continue
    log(f"=== running condition: {key} ({N_SHUFFLE_REPEATS} repeats) ===")
    all_disp = np.zeros((N_SHUFFLE_REPEATS, len(val_sims)))
    all_ag = np.zeros((N_SHUFFLE_REPEATS, len(val_sims)))
    for rep in range(N_SHUFFLE_REPEATS):
        rng = np.random.default_rng(1000 + rep)
        # derangement: random permutation with no fixed points
        perm = rng.permutation(sim_list)
        while np.any(perm == np.array(sim_list)):
            perm = rng.permutation(sim_list)
        companion_map = {int(a): int(b) for a, b in zip(sim_list, perm)}
        fn = make_shuffle_intervention(companion_map, val_features, scheme, seed=2000 + rep)
        disp_rmse, ag_err = run_condition(f"{key}_rep{rep}", fn)
        all_disp[rep] = disp_rmse
        all_ag[rep] = ag_err
        log(f"  {key} rep {rep+1}/{N_SHUFFLE_REPEATS}: disp_rmse_median={np.median(disp_rmse):.4f}")
    results[key] = {
        "disp_rmse_median_per_rep": np.median(all_disp, axis=1).tolist(),
        "ag_median_abs_err_per_rep": np.median(all_ag, axis=1).tolist(),
        "disp_rmse_median_overall": float(np.median(all_disp)),
        "disp_rmse_mean_overall": float(np.mean(all_disp)),
        "ag_median_abs_err_overall": float(np.median(all_ag)),
        "ag_mean_abs_err_overall": float(np.mean(all_ag)),
        "disp_rmse_per_sim_meanoverreps": np.mean(all_disp, axis=0).tolist(),
        "ag_err_per_sim_meanoverreps": np.mean(all_ag, axis=0).tolist(),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log(f"{key}: disp_rmse_median_overall={results[key]['disp_rmse_median_overall']:.4f}")

# --- volume-constrained variant of the volume-matched shuffle: companion
# restricted to sims with final volume >= query's own. With a plain random
# companion pairing, the nearest-volume lookup can be forced to clip to the
# companion's own endpoint once the query's volume exceeds it (this affected
# 27.9% of (sim,step) pairs); requiring the companion's final volume to
# dominate the query's own cuts that clipping rate to 0.02%. Reuses the
# existing scheme="volume" nearest-snapshot-match logic; only companion
# SELECTION differs. ---
key = "shuffle_volume_constrained"
if key not in results:
    log(f"=== running condition: {key} ({N_SHUFFLE_REPEATS} repeats) ===")
    all_disp = np.zeros((N_SHUFFLE_REPEATS, len(val_sims)))
    all_ag = np.zeros((N_SHUFFLE_REPEATS, len(val_sims)))
    for rep in range(N_SHUFFLE_REPEATS):
        companion_map = build_constrained_companion_map(val_features, seed=4000 + rep, higher_final_volume_only=True)
        fn = make_shuffle_intervention(companion_map, val_features, "volume", seed=5000 + rep)
        disp_rmse, ag_err = run_condition(f"{key}_rep{rep}", fn)
        all_disp[rep] = disp_rmse
        all_ag[rep] = ag_err
        log(f"  {key} rep {rep+1}/{N_SHUFFLE_REPEATS}: disp_rmse_median={np.median(disp_rmse):.4f}")
    results[key] = {
        "disp_rmse_median_per_rep": np.median(all_disp, axis=1).tolist(),
        "ag_median_abs_err_per_rep": np.median(all_ag, axis=1).tolist(),
        "disp_rmse_median_overall": float(np.median(all_disp)),
        "disp_rmse_mean_overall": float(np.mean(all_disp)),
        "ag_median_abs_err_overall": float(np.median(all_ag)),
        "ag_mean_abs_err_overall": float(np.mean(all_ag)),
        "disp_rmse_per_sim_meanoverreps": np.mean(all_disp, axis=0).tolist(),
        "ag_err_per_sim_meanoverreps": np.mean(all_ag, axis=0).tolist(),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log(f"{key}: disp_rmse_median_overall={results[key]['disp_rmse_median_overall']:.4f}")
else:
    log(f"=== {key}: already done, skipping ===")

# --- paired bootstrap vs baseline ---
log("=== paired bootstrap (5000 resamples) vs identity baseline ===")
baseline_disp = np.array(results["identity"]["disp_rmse_per_sim"])
bootstrap_results = {}
for name in ("zero", "mean"):
    delta = np.array(results[name]["disp_rmse_per_sim"]) - baseline_disp
    med, lo, hi = paired_bootstrap(delta)
    bootstrap_results[name] = {"median_delta": med, "ci_lo": lo, "ci_hi": hi}
    log(f"  {name}: median_delta_disp_rmse={med:.4f}  95% CI=[{lo:.4f},{hi:.4f}]")
for scheme in ("progress", "volume", "random"):
    key = f"shuffle_{scheme}"
    delta = np.array(results[key]["disp_rmse_per_sim_meanoverreps"]) - baseline_disp
    med, lo, hi = paired_bootstrap(delta)
    bootstrap_results[key] = {"median_delta": med, "ci_lo": lo, "ci_hi": hi}
    log(f"  {key}: median_delta_disp_rmse={med:.4f}  95% CI=[{lo:.4f},{hi:.4f}]")

key = "shuffle_volume_constrained"
delta = np.array(results[key]["disp_rmse_per_sim_meanoverreps"]) - baseline_disp
med, lo, hi = paired_bootstrap(delta)
bootstrap_results[key] = {"median_delta": med, "ci_lo": lo, "ci_hi": hi}
log(f"  {key}: median_delta_disp_rmse={med:.4f}  95% CI=[{lo:.4f},{hi:.4f}]")

results["bootstrap_vs_identity"] = bootstrap_results
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

log("DONE")
