"""Functional substitution: the decisive test of whether Model D's CNN
growth-feedback pathway is actually load-bearing, or merely redundant with
information Model A already has.

predictability.py showed h_g is well-predicted (R2_global=0.9511) by
a probe (q_A, a full_X MLP) trained on Model A's own state. Here, that
probe replaces Model D's TRUE CNN feature with a feature predicted ONLINE,
live, from the model's own current x_base at each rollout step -- no
teacher forcing: predict_fn is called on x_base as it exists at that exact
step of THIS rollout, never on a cached/ground-truth value. If Model D with
the predicted feature performs about as well as Model D with its true CNN
feature, the CNN pathway adds little beyond what Model A-style information
already provides; if it performs much worse, the CNN is contributing
information Model A's state alone cannot recover.

Compares, all 185 validation sims:
  - Model A (reused from the already-cached deliverables zarr, no
    new rollout)
  - Model D - true CNN (the identity-intervention baseline from
    intervention_sweep.py)
  - Model D - predicted CNN (q_A, this script)
  - Model D - shuffled CNN (progress-matched, and volume-matched with the
    constrained-companion selection -- both from intervention_sweep.py)
  - Model D - mean CNN (from intervention_sweep.py)
  - Model D - zero CNN (from intervention_sweep.py)

NOTE: must be run with the working directory set to the repository root
(where the FE dataset from data/DATA_AVAILABILITY.md has been placed) --
this script does not chdir. It adds evaluation/ to sys.path relative to its
own file location.
"""
import os
import sys
import json

import numpy as np
import torch
import zarr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")
sys.path.insert(0, EVAL_DIR)
sys.path.insert(0, SCRIPT_DIR)

import eval_model_a as ma  # noqa: E402
from shared_eval_utils import compute_surface_disp_rmse_trajectory  # noqa: E402
from probe_utils import MLP  # noqa: E402
from rollout_cnn_node_intervened import rollout_cnn_node_intervened, make_predicted_intervention  # noqa: E402

R_MODES = 9
A0_CM2 = 0.25
NODES_CSV = "GOH_Nodes_Test_for_Visualization.csv"
ELEMS_CSV = "GOH_Elements_Test_for_Visualization.csv"
D_CKPT = "checkpoints/Model_D_v2_BEST.pt"
D_VAL_ZARR = "deliverables/D_v2_val_rollouts_r9.zarr"
A_VAL_ZARR = "deliverables/A_v2_val_rollouts_r9.zarr"
OUT_DIR = SCRIPT_DIR

val_sims = np.asarray(ma.val_sims, dtype=np.int64)

# --- load q_A probe ---
q = torch.load(os.path.join(OUT_DIR, "q_A_probe.pt"), map_location="cpu", weights_only=False)
q_model = MLP(q["d_in"], hidden=q["hidden"], d_out=q["d_out"])
q_model.load_state_dict(q["model_state_dict"])
q_model.eval()
x_mean, x_std = q["x_mean"], q["x_std"]
y_mean, y_std = q["y_mean"], q["y_std"]


def predict_fn(x_raw_1x20):
    """Apply the q_A probe to one step's raw (un-normalized) Model-A-style
    state vector, returning a predicted h_g in the same physical units the
    true CNN feature would have. Passed into make_predicted_intervention so
    it is called fresh at every rollout step on that step's own state."""
    xn = (x_raw_1x20 - x_mean) / x_std
    with torch.no_grad():
        yn = q_model(torch.tensor(xn, dtype=torch.float32)).numpy()
    return yn * y_std + y_mean


# --- Model A: reuse cached zarr, no new rollout ---
def per_sim_metrics_from_zarr(zarr_path):
    root = zarr.open_group(zarr_path, mode="r")
    sims_grp = root["simulations"]
    disp_rmse, ag_err, sids = [], [], []
    (_, _, sim_index, time_vals, _, _, _) = ma.load_raw_data(R_MODES)
    H, W, _ = ma.get_bottom_grid_cache()
    for sk in sorted(sims_grp.group_keys()):
        g = sims_grp[sk]
        sid = int(g.attrs["sim_id"])
        u_pred = np.asarray(g["u_pred_nodes"])
        u_true = np.asarray(g["u_true_nodes"])
        rmse_t = compute_surface_disp_rmse_trajectory(u_pred, u_true)
        ag_pred = float(g.attrs["area_gain_pred_final_cm2"])
        ag_true = float(ma.compute_true_net_area_gain_final_for_sim(
            sim_id=sid, sim_index=sim_index, time_vals=time_vals,
            H=H, W=W, zarr_sdv_path=ma.ZARR_SDV, folder_x="snaps_SDV1", folder_y="snaps_SDV4", A0_cm2=A0_CM2,
        ))
        disp_rmse.append(np.mean(rmse_t))
        ag_err.append(abs(ag_pred - ag_true))
        sids.append(sid)
    order = np.argsort(sids)
    return np.array(sids)[order], np.array(disp_rmse)[order], np.array(ag_err)[order]


print("=== Model A (from cache) ===")
sids_A, dispA, agA = per_sim_metrics_from_zarr(A_VAL_ZARR)
print(f"  disp_rmse_median={np.median(dispA):.4f}  ag_median={np.median(agA):.4f}")

# --- Model D predicted-CNN (online, no teacher forcing) ---
print("=== Model D - predicted CNN (q_A, online) ===")
intervention_fn = make_predicted_intervention(predict_fn)
disp_pred, ag_pred_err = [], []
(_, _, sim_index, time_vals, _, _, _) = ma.load_raw_data(R_MODES)
H, W, _ = ma.get_bottom_grid_cache()

root = zarr.open_group(D_VAL_ZARR, mode="r")
sims_grp = root["simulations"]
u_true_by_sim = {int(sims_grp[sk].attrs["sim_id"]): np.asarray(sims_grp[sk]["u_true_nodes"]) for sk in sims_grp.group_keys()}
ag_true_by_sim = {}
for sid in val_sims:
    ag_true_by_sim[int(sid)] = float(ma.compute_true_net_area_gain_final_for_sim(
        sim_id=int(sid), sim_index=sim_index, time_vals=time_vals,
        H=H, W=W, zarr_sdv_path=ma.ZARR_SDV, folder_x="snaps_SDV1", folder_y="snaps_SDV4", A0_cm2=A0_CM2,
    ))

for i, sid in enumerate(val_sims):
    roll = rollout_cnn_node_intervened(
        r_modes=R_MODES, sim_id=int(sid), nodes_csv=NODES_CSV, elems_csv=ELEMS_CSV,
        intervention_fn=intervention_fn, ckpt_path_template=D_CKPT,
    )
    rmse_t = compute_surface_disp_rmse_trajectory(roll["u_pred_nodes"], u_true_by_sim[int(sid)])
    disp_pred.append(np.mean(rmse_t))
    ag_pred_err.append(abs(roll["area_gain_pred_final_cm2"] - ag_true_by_sim[int(sid)]))
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(val_sims)} sims done")

disp_pred = np.array(disp_pred)
ag_pred_err = np.array(ag_pred_err)
print(f"  disp_rmse_median={np.median(disp_pred):.4f}  ag_median={np.median(ag_pred_err):.4f}")

# --- assemble comparison with intervention_sweep.py's already-computed conditions ---
with open(os.path.join(OUT_DIR, "intervention_sweep_results.json")) as f:
    intervention_results = json.load(f)

comparison = {
    "Model_A": {"disp_rmse_median": float(np.median(dispA)), "disp_rmse_mean": float(np.mean(dispA)),
                "ag_median_abs_err": float(np.median(agA)), "ag_mean_abs_err": float(np.mean(agA))},
    "Model_D_true_CNN": {"disp_rmse_median": intervention_results["identity"]["disp_rmse_median"],
                          "disp_rmse_mean": intervention_results["identity"]["disp_rmse_mean"],
                          "ag_median_abs_err": intervention_results["identity"]["ag_median_abs_err"],
                          "ag_mean_abs_err": intervention_results["identity"]["ag_mean_abs_err"]},
    "Model_D_predicted_CNN": {"disp_rmse_median": float(np.median(disp_pred)), "disp_rmse_mean": float(np.mean(disp_pred)),
                              "ag_median_abs_err": float(np.median(ag_pred_err)), "ag_mean_abs_err": float(np.mean(ag_pred_err)),
                              "disp_rmse_per_sim": disp_pred.tolist(), "ag_err_per_sim": ag_pred_err.tolist()},
    "Model_D_shuffled_CNN_progress": {"disp_rmse_median": intervention_results["shuffle_progress"]["disp_rmse_median_overall"],
                                       "disp_rmse_mean": intervention_results["shuffle_progress"]["disp_rmse_mean_overall"],
                                       "ag_median_abs_err": intervention_results["shuffle_progress"]["ag_median_abs_err_overall"],
                                       "ag_mean_abs_err": intervention_results["shuffle_progress"]["ag_mean_abs_err_overall"]},
    "Model_D_shuffled_CNN_volume_constrained": {"disp_rmse_median": intervention_results["shuffle_volume_constrained"]["disp_rmse_median_overall"],
                                                 "disp_rmse_mean": intervention_results["shuffle_volume_constrained"]["disp_rmse_mean_overall"],
                                                 "ag_median_abs_err": intervention_results["shuffle_volume_constrained"]["ag_median_abs_err_overall"],
                                                 "ag_mean_abs_err": intervention_results["shuffle_volume_constrained"]["ag_mean_abs_err_overall"]},
    "Model_D_mean_CNN": {"disp_rmse_median": intervention_results["mean"]["disp_rmse_median"],
                          "disp_rmse_mean": intervention_results["mean"]["disp_rmse_mean"],
                          "ag_median_abs_err": intervention_results["mean"]["ag_median_abs_err"],
                          "ag_mean_abs_err": intervention_results["mean"]["ag_mean_abs_err"]},
    "Model_D_zero_CNN": {"disp_rmse_median": intervention_results["zero"]["disp_rmse_median"],
                          "disp_rmse_mean": intervention_results["zero"]["disp_rmse_mean"],
                          "ag_median_abs_err": intervention_results["zero"]["ag_median_abs_err"],
                          "ag_mean_abs_err": intervention_results["zero"]["ag_mean_abs_err"]},
}

# paired bootstrap: predicted-CNN vs true-CNN (the decisive comparison)
identity_per_sim = np.array(intervention_results["identity"]["disp_rmse_per_sim"])
delta = disp_pred - identity_per_sim
rng = np.random.default_rng(42)
n = len(delta)
boot = np.array([np.median(delta[rng.integers(0, n, n)]) for _ in range(5000)])
lo, hi = np.percentile(boot, [2.5, 97.5])
comparison["bootstrap_predicted_vs_true"] = {"median_delta": float(np.median(delta)), "ci_lo": float(lo), "ci_hi": float(hi)}

print()
print("=== FINAL COMPARISON TABLE ===")
for k, v in comparison.items():
    if k == "bootstrap_predicted_vs_true":
        continue
    print(f"  {k:28s} disp_rmse_median={v['disp_rmse_median']:.4f}  ag_median={v['ag_median_abs_err']:.4f}")
print(f"  predicted-CNN vs true-CNN delta = {comparison['bootstrap_predicted_vs_true']['median_delta']:+.4f}  "
      f"95% CI=[{comparison['bootstrap_predicted_vs_true']['ci_lo']:+.4f}, {comparison['bootstrap_predicted_vs_true']['ci_hi']:+.4f}]")

with open(os.path.join(OUT_DIR, "functional_substitution_results.json"), "w") as f:
    json.dump(comparison, f, indent=2)
print("saved functional_substitution_results.json")
print("DONE")
