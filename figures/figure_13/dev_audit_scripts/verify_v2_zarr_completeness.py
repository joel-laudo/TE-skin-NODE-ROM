"""Archived development-time diagnostic (see ../README.md, "dev_audit_scripts/"):
verify all four v2uncorrected rollout zarrs have complete
disp_err_pointwise_rmse and t fields for all 185 validation sims, and that
all four models' inferred global-max support time agrees, before building
the Figure 11/13 plots. Not part of reproducing either figure; read-only.
The paths below reference the old private-project `phase36_deliverables/`
layout and will not resolve in this repo -- kept only for provenance."""
import numpy as np
import zarr

ZARRS = {
    "A": "phase36_deliverables/A_v2uncorrected_val_rollouts_r9.zarr",
    "B": "phase36_deliverables/B_v2uncorrected_val_rollouts_r9.zarr",
    "C": "phase36_deliverables/C_v2uncorrected_val_rollouts_r9.zarr",
    "D": "phase36_deliverables/D_v2uncorrected_val_rollouts_r9.zarr",
}

all_t_max_by_fam = {}
for fam, path in ZARRS.items():
    root = zarr.open_group(path, mode="r")
    sims_grp = root["simulations"]
    sim_keys = sorted(list(sims_grp.group_keys()))
    n_missing = 0
    t_maxes = []
    for sk in sim_keys:
        g = sims_grp[sk]
        if "disp_err_pointwise_rmse" not in g or "t" not in g:
            n_missing += 1
            continue
        t = np.asarray(g["t"], dtype=np.float64)
        t_maxes.append(t.max())
    target_time = float(np.max(t_maxes)) if t_maxes else None
    all_t_max_by_fam[fam] = target_time
    print(f"{fam}: n_sims={len(sim_keys)} n_missing_fields={n_missing} target_time(global_max_t)={target_time}")

vals = list(all_t_max_by_fam.values())
print(f"\nAll four target_times equal: {len(set(vals)) == 1} -> {vals}")
