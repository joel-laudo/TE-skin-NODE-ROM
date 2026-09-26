"""Archived development-time diagnostic (see ../README.md, "dev_audit_scripts/"):
verify whether a later re-generated copy of the pre-revision rollout zarrs
in FINAL_models/Model_{A,B,C,D}/ reproduces the diagnostic numbers recorded
earlier from the same evaluation pipeline, using the same target-time /
support / interpolation logic as the Figure 11 histogram panels
(infer_support_time_from_global_max_t=True: target_time = max over all 185
sims' own max-t, per model's own zarr).

Not part of reproducing any reported figure; read-only, prints to stdout
only. The `FINAL_models/Model_*/` paths below are from the excluded,
pre-"v2" checkpoint family and will not resolve in this repo -- kept only
for provenance.
"""
import numpy as np
import zarr

ZARRS = {
    "A (vanilla)": "FINAL_models/Model_A/Model_A_vanilla_val_rollouts_r9.zarr",
    "B (ag)": "FINAL_models/Model_B/Model_B_Ag_val_rollouts_r9.zarr",
    "C (pca)": "FINAL_models/Model_C/Model_C_pca_val_rollouts_r9.zarr",
    "D (cnn)": "FINAL_models/Model_D/Model_D_cnn_val_rollouts_r9.zarr",
}

# Exact saved March-2026 notebook output, for comparison
ORIGINAL = {
    "A (vanilla)": dict(n_used=30, n_skipped=155, mean=6.811554, median=7.400182,
                          p10=4.367335, p90=8.597694, vmin=3.365692, vmax=8.958734),
    "B (ag)": dict(n_used=30, n_skipped=155, mean=5.544557, median=5.821345,
                    p10=2.773457, p90=7.848920, vmin=1.735626, vmax=10.721738),
    "C (pca)": dict(n_used=30, n_skipped=155, mean=4.390776, median=3.145210,
                     p10=1.896740, p90=8.846492, vmin=1.705083, vmax=14.214500),
    "D (cnn)": dict(n_used=30, n_skipped=155, mean=1.084154, median=0.836258,
                     p10=0.528646, p90=1.707560, vmin=0.312438, vmax=5.509896),
}

atol = 1e-6

for fam, path in ZARRS.items():
    try:
        root = zarr.open_group(path, mode="r")
    except Exception as e:
        print(f"{fam}: FAILED to open {path}: {e!r}")
        continue
    sims_grp = root["simulations"]
    sim_keys = sorted(list(sims_grp.group_keys()))

    all_t_max = []
    per_sim = {}
    for sk in sim_keys:
        g = sims_grp[sk]
        if "disp_err_pointwise_rmse" not in g or "t" not in g:
            continue
        t = np.asarray(g["t"], dtype=np.float64)
        rmse_t = np.asarray(g["disp_err_pointwise_rmse"], dtype=np.float64)
        n = min(len(t), len(rmse_t))
        t, rmse_t = t[:n], rmse_t[:n]
        order = np.argsort(t)
        t, rmse_t = t[order], rmse_t[order]
        per_sim[sk] = (t, rmse_t)
        all_t_max.append(t.max())

    target_time = float(np.max(all_t_max))

    vals = []
    n_skipped = 0
    for sk, (t, rmse_t) in per_sim.items():
        t_min, t_max = t.min(), t.max()
        if (target_time < t_min - atol) or (target_time > t_max + atol):
            n_skipped += 1
            continue
        t_unique, idx = np.unique(t, return_index=True)
        rmse_unique = rmse_t[idx]
        val = float(np.interp(target_time, t_unique, rmse_unique))
        vals.append(val)

    vals = np.array(vals)
    orig = ORIGINAL[fam]
    print(f"=== {fam} ===")
    print(f"  target_time (global max t) = {target_time:.6f}")
    print(f"  n_used={len(vals)} (orig {orig['n_used']})  n_skipped={n_skipped} (orig {orig['n_skipped']})")
    if len(vals) > 0:
        mean_v, median_v = float(np.mean(vals)), float(np.median(vals))
        p10_v, p90_v = float(np.percentile(vals, 10)), float(np.percentile(vals, 90))
        vmin_v, vmax_v = float(vals.min()), float(vals.max())
        print(f"  mean={mean_v:.6f} (orig {orig['mean']:.6f}, diff={mean_v-orig['mean']:+.6f})")
        print(f"  median={median_v:.6f} (orig {orig['median']:.6f}, diff={median_v-orig['median']:+.6f})")
        print(f"  p10={p10_v:.6f} (orig {orig['p10']:.6f})  p90={p90_v:.6f} (orig {orig['p90']:.6f})")
        print(f"  min={vmin_v:.6f} (orig {orig['vmin']:.6f})  max={vmax_v:.6f} (orig {orig['vmax']:.6f})")
        match = abs(median_v - orig["median"]) < 1e-3 and abs(mean_v - orig["mean"]) < 1e-3
        print(f"  VERDICT: {'MATCHES (July zarr faithfully reproduces March run)' if match else 'DIVERGES -- investigate further'}")
    print()
