"""NODE displacement RMSE for Models A and D on the 185 held-out validation
simulations, using each model's BEST checkpoint (Model_A_Vanilla_r9_BEST.pt
epoch 356, Model_D_CNN_r9_BEST.pt epoch 357). This script reads
pre-computed rollout zarrs; no new rollout is run here.

Reads: deliverables/A_v2_val_rollouts_r9.zarr and
deliverables/D_v2_val_rollouts_r9.zarr -- each has one group per validation
simulation (attrs: sim_id) containing the pointwise displacement RMSE
trajectory (disp_err_pointwise_rmse). These zarrs are produced by running
`python evaluation/generate_deliverables.py` from the repository root
(with the FE dataset in place); run that first if they don't exist yet.

Writes: step8_node_error_val185.csv (per-simulation e_node for Models A and
D on the 185-simulation validation set), used by the ROM-accuracy
comparison in SI Section 2 / Figure 2 and consumed by
step9_master_join.py.
"""
import os
import numpy as np
import pandas as pd
import zarr

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

ZARRS = {
    "A": os.path.join("deliverables", "A_v2_val_rollouts_r9.zarr"),
    "D": os.path.join("deliverables", "D_v2_val_rollouts_r9.zarr"),
}

per_model = {}
for fam, path in ZARRS.items():
    root = zarr.open_group(path, mode="r")
    sims_grp = root["simulations"]
    sim_keys = sorted(list(sims_grp.group_keys()))
    rows = []
    for sk in sim_keys:
        g = sims_grp[sk]
        sid = int(g.attrs["sim_id"])
        rmse_t = np.asarray(g["disp_err_pointwise_rmse"], dtype=np.float64)
        rows.append({"sim_id": sid, f"e_node_{fam}": float(np.mean(rmse_t))})
    per_model[fam] = pd.DataFrame(rows)
    print(f"Model {fam}: n_sims={len(rows)}, median e_node={per_model[fam][f'e_node_{fam}'].median():.4f}")

merged = per_model["A"].merge(per_model["D"], on="sim_id", how="inner")
assert len(merged) == 185, f"expected 185 validation sims, got {len(merged)}"
merged["is_validation"] = True

out_path = os.path.join(OUT_DIR, "step8_node_error_val185.csv")
merged.to_csv(out_path, index=False)
print(f"\nsaved {out_path} (n={len(merged)})")
print("DONE")
