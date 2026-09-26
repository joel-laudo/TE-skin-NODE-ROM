"""Build the per-snapshot CNN-feature dataset used throughout Section 4.2's
conditional-utility analysis of Model D's learned growth feedback.

Model D encodes the current growth field g(x,y) with a small CNN
(`growth_encoder`) into a feature vector h_g (called g_CNN in the
manuscript) that is fed back into the NODE's velocity network at every
rollout step. This script recovers h_g -- along with the matching physical
growth quantities and Model-A-style state inputs -- for every snapshot of
every simulation, so that later scripts (growth_decoding.py,
intervention_sweep.py, predictability.py,
functional_substitution.py, representation_structure.py,
jacobian_sensitivity.py) can all probe the same feature dataset rather than
re-running Model D.

Validation sims (185): feature extraction is POST-HOC only -- reuses the
already-cached deliverables/D_v2_val_rollouts_r9.zarr
(g_pred_grid, lamdag_pred_elem, z_pred, e_hist, I_hist, t already stored per
step for every val sim), and runs only the checkpoint's own growth_encoder
submodule on the cached (already-normalized) g_pred_grid arrays to recover
h_g. No new rollout needed.

Training sims (742 = all sims minus val_sims): no cache exists, so one
fresh rollout pass via eval_model_d.rollout_single_sim_cnn_node_matched
is run per sim, then the same post-hoc growth_encoder pass is applied.

Design choice (disclosed): features are extracted ON-POLICY, from Model D's
own realized rollout trajectory (its own predicted z, hence its own e/I and
growth field) -- not a teacher-forced ground-truth trajectory. This is what
the later probing/intervention/substitution analyses need: the model's own
current predicted state, not a cached ground-truth one.

modelA_inputs = [z(10), e, I, sp, design(7)] is stored ONCE per row (raw,
un-normalized physical units) -- it serves both the "Model A inputs" and
"Model D non-CNN inputs" roles, since x_base for Model D is structurally
identical to Model A's own NODE input.

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
import torch
import zarr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(SCRIPT_DIR, "..", "..", "evaluation")
sys.path.insert(0, EVAL_DIR)
sys.path.insert(0, SCRIPT_DIR)

import shared_eval_utils as md  # noqa: E402
from eval_model_d import rollout_single_sim_cnn_node_matched  # noqa: E402
import eval_model_a as ma  # noqa: E402
from build_growth_helpers import mean_lambda_g, growth_anisotropy, project_onto_growth_pca  # noqa: E402

t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


R_MODES = 9
A0_CM2 = 0.25
NODES_CSV = "GOH_Nodes_Test_for_Visualization.csv"
ELEMS_CSV = "GOH_Elements_Test_for_Visualization.csv"
D_CKPT = "checkpoints/Model_D_v2_BEST.pt"
D_VAL_ZARR = "deliverables/D_v2_val_rollouts_r9.zarr"
GROWTH_PCA_NPZ = "growth_pca_trainonly_H60_W60_k8_val0.20_seed123.npz"

OUT_DIR = SCRIPT_DIR

device = torch.device("cpu")
log("loading Model D checkpoint + growth_encoder...")
ckpt = torch.load(D_CKPT, map_location=device)
model = md.build_cnn_node_from_ckpt(ckpt, device)
model.eval()
growth_encoder = model.growth_encoder

log("loading raw data (sim_index/time_vals/volume_SP/design_all)...")
(U_lat, volume_snap, sim_index, time_vals, volume_SP_all, design_all, n_sims) = md.load_raw_data(R_MODES)

val_sims = np.asarray(ma.val_sims, dtype=np.int64)
all_sims = np.unique(sim_index)
val_set = set(int(s) for s in val_sims)
train_sims = np.asarray([s for s in all_sims if int(s) not in val_set], dtype=np.int64)
log(f"n_val_sims={len(val_sims)}  n_train_sims={len(train_sims)}  n_all_sims={len(all_sims)}")

pca_npz = np.load(GROWTH_PCA_NPZ)
H = int(pca_npz["H"])
W = int(pca_npz["W"])


def sim_context(sim_id):
    """Per-sim (idx_sorted, sp_sim, design_vec) needed to align raw quantities."""
    idx = np.where(sim_index == sim_id)[0]
    t_s = time_vals[idx]
    order = np.argsort(t_s)
    idx_sorted = idx[order]
    sp_sim = volume_SP_all[idx_sorted]
    design_vec = design_all[idx_sorted[0], :]  # constant within a sim
    target_volume = float(sp_sim[-1])
    return idx_sorted, sp_sim, design_vec, target_volume


@torch.no_grad()
def run_growth_encoder_batch(g_pred_grid: np.ndarray) -> np.ndarray:
    """Run Model D's own growth_encoder CNN on a batch of normalized growth
    grids to recover h_g (g_CNN), the compact feature vector the NODE
    conditions on. g_pred_grid: (T,2,H,W) already-normalized CNN input
    (channels = the two in-plane growth stretches lambda_g_x, lambda_g_y on
    the bottom-surface (H,W) grid). Returns (T,dim_gfeat) h_g."""
    x = torch.from_numpy(np.ascontiguousarray(g_pred_grid, dtype=np.float32))
    h = growth_encoder(x)
    return h.cpu().numpy()


def rows_from_rollout(sim_id, t_arr, z_pred, e_hist, I_hist, g_pred_grid, lamdag_pred_elem):
    """Build per-step rows for one sim's rollout arrays (all already numpy)."""
    idx_sorted, sp_sim, design_vec, target_volume = sim_context(sim_id)
    T = t_arr.shape[0]
    if sp_sim.shape[0] != T:
        raise ValueError(f"sim {sim_id}: sp_sim length {sp_sim.shape[0]} != rollout T={T}")

    h_g = run_growth_encoder_batch(g_pred_grid)  # (T, dim_gfeat)

    rows = {
        "cnn_final_features": h_g,
        "z": z_pred.astype(np.float32),
        "modelA_inputs": np.concatenate(
            [
                z_pred.astype(np.float32),
                e_hist.reshape(-1, 1).astype(np.float32),
                I_hist.reshape(-1, 1).astype(np.float32),
                sp_sim.reshape(-1, 1).astype(np.float32),
                np.tile(design_vec.astype(np.float32), (T, 1)),
            ],
            axis=1,
        ),
        "growth_pca": np.zeros((T, 16), dtype=np.float32),
        "Ag": np.zeros((T,), dtype=np.float32),
        "mean_lambda_g1": np.zeros((T,), dtype=np.float32),
        "mean_lambda_g2": np.zeros((T,), dtype=np.float32),
        "growth_anisotropy": np.zeros((T,), dtype=np.float32),
        "time": t_arr.astype(np.float64),
        "volume": z_pred[:, R_MODES].astype(np.float32),
        "target_volume": np.full((T,), target_volume, dtype=np.float32),
        "sim_id": np.full((T,), sim_id, dtype=np.int64),
        "snapshot_id": np.arange(T, dtype=np.int64),
    }
    for k in range(T):
        lamdag_k = lamdag_pred_elem[k]
        rows["growth_pca"][k] = project_onto_growth_pca(lamdag_k, H, W, pca_npz)
        rows["Ag"][k] = ma.compute_net_area_gain_from_lamdag_elem(lamdag_k, A0_cm2=A0_CM2)
        m1, m2 = mean_lambda_g(lamdag_k)
        rows["mean_lambda_g1"][k] = m1
        rows["mean_lambda_g2"][k] = m2
        rows["growth_anisotropy"][k] = growth_anisotropy(lamdag_k)
    return rows


def concat_rows(row_dicts):
    keys = row_dicts[0].keys()
    return {k: np.concatenate([r[k] for r in row_dicts], axis=0) for k in keys}


# ---------------------------------------------------------------------------
# Validation sims: post-hoc only, from the cached zarr
# ---------------------------------------------------------------------------
log("=== Validation sims: extracting from cached zarr (no new rollout) ===")
root = zarr.open_group(D_VAL_ZARR, mode="r")
sims_grp = root["simulations"]
val_rows = []
for i, sk in enumerate(sorted(sims_grp.group_keys())):
    g = sims_grp[sk]
    sim_id = int(g.attrs["sim_id"])
    t_arr = np.asarray(g["t"])
    z_pred = np.asarray(g["z_pred"])
    e_hist = np.asarray(g["e_hist"])
    I_hist = np.asarray(g["I_hist"])
    g_pred_grid = np.asarray(g["g_pred_grid"])
    lamdag_pred_elem = np.asarray(g["lamdag_pred_elem"])
    val_rows.append(rows_from_rollout(sim_id, t_arr, z_pred, e_hist, I_hist, g_pred_grid, lamdag_pred_elem))
    if (i + 1) % 50 == 0:
        log(f"  val: {i+1}/{len(val_sims)} sims processed")

val_data = concat_rows(val_rows)
np.savez_compressed(os.path.join(OUT_DIR, "val_features.npz"), **val_data)
log(f"saved val_features.npz  n_rows={val_data['sim_id'].shape[0]}")

# ---------------------------------------------------------------------------
# Training sims: fresh rollout needed (no cache exists)
# ---------------------------------------------------------------------------
log("=== Training sims: running fresh Model D rollouts (742 sims) ===")
train_rows = []
n_failed = 0
for i, sim_id in enumerate(train_sims):
    try:
        roll = rollout_single_sim_cnn_node_matched(
            r_modes=R_MODES, sim_id=int(sim_id), nodes_csv=NODES_CSV, elems_csv=ELEMS_CSV,
            ckpt_path_template=D_CKPT, make_plots=False,
        )
        train_rows.append(
            rows_from_rollout(
                int(sim_id), np.asarray(roll["t"]), np.asarray(roll["z_pred"]),
                np.asarray(roll["e_hist"]), np.asarray(roll["I_hist"]),
                np.asarray(roll["g_pred_grid"]), np.asarray(roll["lamdag_pred_elem"]),
            )
        )
    except Exception as exc:  # noqa: BLE001 -- one bad sim shouldn't kill a 742-sim run
        n_failed += 1
        log(f"  !! sim {sim_id} FAILED: {exc!r}")
    if (i + 1) % 25 == 0:
        log(f"  train: {i+1}/{len(train_sims)} sims processed ({n_failed} failed so far)")
    if (i + 1) % 100 == 0:
        # incremental checkpointing in case of a long-running crash
        partial = concat_rows(train_rows)
        np.savez_compressed(os.path.join(OUT_DIR, "train_features_partial.npz"), **partial)

train_data = concat_rows(train_rows)
np.savez_compressed(os.path.join(OUT_DIR, "train_features.npz"), **train_data)
partial_path = os.path.join(OUT_DIR, "train_features_partial.npz")
if os.path.exists(partial_path):
    os.remove(partial_path)
log(f"saved train_features.npz  n_rows={train_data['sim_id'].shape[0]}  n_failed_sims={n_failed}")

metadata = {
    "model_D_ckpt": D_CKPT,
    "r_modes": R_MODES,
    "H": H, "W": W,
    "dim_gfeat": int(ckpt["model_state_dict"]["growth_encoder.head.1.bias"].numel()),
    "n_val_sims": int(len(val_sims)),
    "n_train_sims": int(len(train_sims)),
    "n_train_failed": int(n_failed),
    "val_rows": int(val_data["sim_id"].shape[0]),
    "train_rows": int(train_data["sim_id"].shape[0]),
    "modelA_inputs_convention": "[z(10, incl. volume as z[:,9]), e, I, sp, design(7)] -- raw physical units, NOT normalized. Identical structurally to Model A's own NODE input and to Model D's x_base (VelocityNet.forward's first argument).",
    "on_policy": True,
    "on_policy_note": "All features extracted from Model D's own realized rollout trajectory (own predicted z, e, I, growth field), not FE ground truth -- i.e. no teacher forcing, matching how the intervention-sweep and functional-substitution analyses use these features.",
    "growth_pca_source": GROWTH_PCA_NPZ,
    "growth_pca_convention": "Bit-for-bit matches X_pca_all_norm's own convention: per-channel PCA (mean_x/components_x, mean_y/components_y) then normalized by the basis's own train-only pca_feat_mean/pca_feat_std.",
    "val_zarr_source": D_VAL_ZARR,
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}
with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
log("saved metadata.json")
log("DONE")
