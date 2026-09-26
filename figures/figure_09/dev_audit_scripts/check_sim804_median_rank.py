"""Archived development-time diagnostic (see ../README.md, "dev_audit_scripts/"):
is validation sim 804 actually the median-performing simulation for each
model's pre-revision ("original") Figure-11 rollout zarr? Not part of
reproducing any reported figure; read-only, prints to stdout only. The
`FINAL_models/Model_*/` paths below are from the excluded, pre-"v2"
checkpoint family and will not resolve in this repo -- kept only for
provenance."""
import numpy as np
import zarr

ZARRS = {
    "A": "FINAL_models/Model_A/Model_A_vanilla_val_rollouts_r9.zarr",
    "B": "FINAL_models/Model_B/Model_B_Ag_val_rollouts_r9.zarr",
    "C": "FINAL_models/Model_C/Model_C_pca_val_rollouts_r9.zarr",
    "D": "FINAL_models/Model_D/Model_D_cnn_val_rollouts_r9.zarr",
}

for fam, path in ZARRS.items():
    try:
        root = zarr.open_group(path, mode="r")
    except Exception as e:
        print(f"{fam}: FAILED to open {path}: {e!r}")
        continue
    sims_grp = root["simulations"]
    sim_keys = sorted(list(sims_grp.group_keys()))
    per_sim_mean_rmse = {}
    missing_field = []
    for sk in sim_keys:
        g = sims_grp[sk]
        sid = int(g.attrs.get("sim_id", sk))
        if "disp_err_pointwise_rmse" not in g:
            missing_field.append(sid)
            continue
        rmse_t = np.asarray(g["disp_err_pointwise_rmse"], dtype=np.float64)
        per_sim_mean_rmse[sid] = float(np.mean(rmse_t))

    if missing_field:
        print(f"{fam}: {len(missing_field)} sims missing 'disp_err_pointwise_rmse' field "
              f"(e.g. {missing_field[:5]}) -- cannot fully rank")
    if not per_sim_mean_rmse:
        print(f"{fam}: no usable per-sim RMSE found, skipping")
        continue

    sids = np.array(sorted(per_sim_mean_rmse.keys()))
    vals = np.array([per_sim_mean_rmse[s] for s in sids])
    order = np.argsort(vals)
    ranked_sids = sids[order]
    ranked_vals = vals[order]
    n = len(sids)
    median_val = np.median(vals)
    median_idx = np.argmin(np.abs(vals - median_val))
    median_sid = sids[median_idx]

    if 804 in per_sim_mean_rmse:
        rank_of_804 = int(np.where(ranked_sids == 804)[0][0]) + 1  # 1-indexed
        val_804 = per_sim_mean_rmse[804]
        pct_804 = 100.0 * rank_of_804 / n
        print(f"{fam}: n_sims={n} median_val={median_val:.4f} (closest sim={median_sid}) | "
              f"sim804_val={val_804:.4f} sim804_rank={rank_of_804}/{n} (pctile={pct_804:.1f}%)")
    else:
        print(f"{fam}: n_sims={n} median_val={median_val:.4f} (closest sim={median_sid}) | "
              f"sim 804 NOT FOUND in this zarr's validation set")
