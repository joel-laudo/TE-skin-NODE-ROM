"""Are Model D's actual learned CNN growth features (h_g) predictable from
information Model A already has, without ever looking at the growth field
itself? If so, the growth-feedback pathway may be recoverable/redundant
given the mechanical-latent trajectory and design parameters alone --
motivating the functional-substitution test in functional_substitution.py.

Builds nested tiers of Model-A-available information -- h(mu,t), h(a),
h(a,mu), h(a,mu,t), and the full existing-state X -- and, for each tier,
fits a probe (linear_probe/mlp_probe from probe_utils.py) to predict h_g
(16-dim, cnn_final_features). Comparing R^2/cosine-similarity across tiers
shows which pieces of information (design parameters mu, POD coefficients
a, elapsed time t) actually carry the growth-relevant signal.

modelA_inputs column layout (see extract_features.py): [z(10, incl. volume
at col 9), e(1), I(1), sp(1), design(7)] -- so:
  alpha (a)  = modelA_inputs[:, 0:9]      (POD coeffs, EXCLUDES volume)
  design(mu) = modelA_inputs[:, 13:20]
  time (t)   = the separate 'time' array
Note there is no intermediate tier isolating e/I/sp (the pressure-tracking
error/integral/setpoint terms) alone -- that omission is intentional, not an
oversight, since those terms are secondary to the a/mu/t partition this
analysis is built around.
"""
import os
import sys
import json

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

from probe_utils import linear_probe, mlp_probe  # noqa: E402

train = np.load("train_features.npz")
val = np.load("val_features.npz")

h_train = train["cnn_final_features"]
h_val = val["cnn_final_features"]


def build_sets(feat):
    modelA_inputs = feat["modelA_inputs"]
    alpha = modelA_inputs[:, 0:9]
    design = modelA_inputs[:, 13:20]
    t = feat["time"].reshape(-1, 1)
    full_X = modelA_inputs
    return {
        "h(mu,t)": np.concatenate([design, t], axis=1),
        "h(a)": alpha,
        "h(a,mu)": np.concatenate([alpha, design], axis=1),
        "h(a,mu,t)": np.concatenate([alpha, design, t], axis=1),
        "full_X": full_X,
    }


sets_train = build_sets(train)
sets_val = build_sets(val)

results = {}
print("=== Predicting Model D's h_g from nested Model A information tiers ===")
for name in ["h(mu,t)", "h(a)", "h(a,mu)", "h(a,mu,t)", "full_X"]:
    Xtr, Xva = sets_train[name], sets_val[name]
    lin = linear_probe(Xtr, h_train, Xva, h_val)
    mlp = mlp_probe(Xtr, h_train, Xva, h_val)

    pred_mlp = mlp["pred_val_denorm"]
    # Cosine similarity per row, averaged: unlike R^2 (which penalizes any
    # magnitude/scale mismatch), cosine similarity only asks whether the
    # predicted h_g points in the same direction as the true h_g. Reporting
    # both separates "gets the direction right" from "gets the full vector
    # (direction + magnitude) right".
    num = np.sum(pred_mlp * h_val, axis=1)
    den = np.linalg.norm(pred_mlp, axis=1) * np.linalg.norm(h_val, axis=1) + 1e-12
    cos_sim = float(np.mean(num / den))

    print(f"  {name:12s}  d_in={Xtr.shape[1]:2d}  linear R2={lin['r2_global']:.4f}  "
          f"MLP R2={mlp['r2_global']:.4f}  cos_sim={cos_sim:.4f}")

    results[name] = {
        "d_in": int(Xtr.shape[1]),
        "linear_r2_global": lin["r2_global"],
        "linear_r2_per_dim": lin["r2_per_dim"],
        "linear_rmse_normalized": lin["rmse_normalized"],
        "mlp_r2_global": mlp["r2_global"],
        "mlp_r2_per_dim": mlp["r2_per_dim"],
        "mlp_rmse_normalized": mlp["rmse_normalized"],
        "cosine_similarity": cos_sim,
    }

with open(os.path.join(SCRIPT_DIR, "predictability_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("saved predictability_results.json")

# ---------------------------------------------------------------------------
# Persist the best-performing probe (full_X MLP) as "q_A", for the online
# functional-substitution test in functional_substitution.py --
# retrain once more (deterministic, same seed as the run above) and save
# state_dict + normalization stats.
# ---------------------------------------------------------------------------
print("=== retraining + saving the full_X MLP probe (q_A) for functional_substitution.py ===")
best_mlp = mlp_probe(sets_train["full_X"], h_train, sets_val["full_X"], h_val)
torch.save({
    "model_state_dict": best_mlp["model"].state_dict(),
    "x_mean": best_mlp["x_mean"], "x_std": best_mlp["x_std"],
    "y_mean": best_mlp["y_mean"], "y_std": best_mlp["y_std"],
    "d_in": sets_train["full_X"].shape[1],
    "d_out": h_train.shape[1],
    "hidden": 128,
    "input_tier": "full_X",
    "val_r2_global": best_mlp["r2_global"],
}, os.path.join(SCRIPT_DIR, "q_A_probe.pt"))
print(f"saved q_A_probe.pt (val R2_global={best_mlp['r2_global']:.4f})")
print("DONE")
