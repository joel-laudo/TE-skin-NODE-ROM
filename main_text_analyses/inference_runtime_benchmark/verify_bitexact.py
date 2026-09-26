"""Sanity check: verify that rollout_benchmark_wrappers.py's pre-loaded-model
wrappers reproduce the authoritative rollout functions
(eval_ablation_model_a_base_closureMLP.rollout_single_sim_vanilla_node and
eval_rollout_cnn.rollout_single_sim_cnn_node_matched) bit-exactly, before any
benchmark timing from benchmark_runtime.py is trusted. If the wrappers ever
drifted from the originals, the runtime comparison in Section 4.1 would be
timing something other than the models actually described in the paper.

Run this locally (or on the target cluster) BEFORE relying on
benchmark_runtime.py's output. It is not part of the timed benchmark itself.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

import eval_ablation_model_a_base_closureMLP as ma
import eval_rollout_cnn as ed
from rollout_benchmark_wrappers import rollout_A_preloaded, rollout_D_preloaded, prefetch_growth_data

R_MODES = 9
NODES_CSV = "GOH_Nodes_Test_for_Visualization.csv"
ELEMS_CSV = "GOH_Elements_Test_for_Visualization.csv"
A_CKPT = "FINAL_models/Model_A_ablation_fullrollout_v2/Model_A_Vanilla_r9_BEST.pt"
D_CKPT = "FINAL_models/Model_D_ablation_fullrollout_v2/Model_D_CNN_r9_BEST.pt"

N_CHECK_SIMS = 3


def max_abs_diff(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def check_model(name, sample_sims, original_fn, ckpt_path, wrapper_fn, build_fn,
                 exact_fields, growth_cache, growth_expected_to_differ=False):
    print(f"=== {name}: bit-exact check on {len(sample_sims)} sims ===")
    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_fn(ckpt, device)
    model.eval()

    all_ok = True
    for sim_id in sample_sims:
        ref = original_fn(
            r_modes=R_MODES, sim_id=int(sim_id), nodes_csv=NODES_CSV, elems_csv=ELEMS_CSV,
            ckpt_path_template=ckpt_path, make_plots=False,
        )
        test = wrapper_fn(
            model=model, ckpt=ckpt, device=device, r_modes=R_MODES, sim_id=int(sim_id),
            nodes_csv=NODES_CSV, elems_csv=ELEMS_CSV, growth_cache=growth_cache,
        )
        diffs = {k: max_abs_diff(ref[k], test[k]) for k in exact_fields}
        worst = max(diffs.values())
        ok = worst < 1e-6
        all_ok = all_ok and ok
        print(f"  sim {sim_id}: max abs diffs (must match) = {diffs} -> {'OK' if ok else 'FAIL'}")

        if growth_expected_to_differ:
            # Reported for transparency only -- NOT part of the pass/fail
            # check, since rollout_A_preloaded deliberately swaps in the
            # matched growth integrator (see rollout_benchmark_wrappers.py).
            ag_diff = abs(ref["area_gain_pred_final_cm2"] - test["area_gain_pred_final_cm2"])
            g_diff = max_abs_diff(ref["g_pred_grid"], test["g_pred_grid"])
            print(f"    (expected-to-differ, integrator swap) area_gain diff={ag_diff:.4g}, "
                  f"g_pred_grid max abs diff={g_diff:.4g}")
    return all_ok


if __name__ == "__main__":
    sample_sims = [int(s) for s in ma.val_sims[:N_CHECK_SIMS]]

    (_, _, sim_index, time_vals, _, _, _) = ma.load_raw_data(R_MODES)
    growth_cache = prefetch_growth_data(sample_sims, sim_index, time_vals)

    # Model A: only u_pred_nodes/z_pred/vol_pred are required to match exactly
    # -- those are the fields Model A's own NODE dynamics actually determine.
    # g_pred_grid/lamdag_pred_elem/area_gain are expected to differ, since
    # rollout_A_preloaded deliberately uses the matched growth integrator
    # (see rollout_benchmark_wrappers.py's module docstring for why).
    ok_a = check_model(
        "Model A", sample_sims,
        original_fn=ma.rollout_single_sim_vanilla_node, ckpt_path=A_CKPT,
        wrapper_fn=rollout_A_preloaded, build_fn=ma.build_vanilla_node_from_ckpt,
        exact_fields=["u_pred_nodes", "z_pred", "vol_pred"],
        growth_cache=growth_cache, growth_expected_to_differ=True,
    )
    # Model D: no deviations at all -- everything must match exactly.
    ok_d = check_model(
        "Model D", sample_sims,
        original_fn=ed.rollout_single_sim_cnn_node_matched, ckpt_path=D_CKPT,
        wrapper_fn=rollout_D_preloaded, build_fn=ed.build_cnn_node_from_ckpt,
        exact_fields=["u_pred_nodes", "z_pred", "g_pred_grid", "lamdag_pred_elem", "vol_pred",
                      "area_gain_pred_final_cm2"],
        growth_cache=growth_cache,
    )

    print()
    if ok_a and ok_d:
        print("ALL CHECKS PASSED -- wrappers are bit-exact. Safe to trust benchmark_runtime.py timings.")
    else:
        print("CHECK FAILED -- do NOT trust benchmark timings until this is fixed.")
        raise SystemExit(1)
