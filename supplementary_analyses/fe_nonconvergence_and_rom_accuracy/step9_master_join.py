"""Build the master per-simulation table, joining job-level failure-risk
scores (p_fail, d_fail) onto sim-level POD and NODE errors.

The join key is the sim_id -> job_id mapping stored in
Design_and_Metadata.zarr's job_index_per_sim, which is indexed directly by
sim_id (job_id = int(job_index_per_sim[sim_id])).

Reads:
  - step6_failure_risk_scores_all_1000.csv (job-level p_fail/d_fail, from
    step6_failure_risk_scores.py)
  - step7_pod_reconstruction_error.csv (sim-level POD basis error, from
    step7_pod_reconstruction_error.py)
  - step8_node_error_val185.csv (sim-level NODE error on the validation
    set, from step8_node_error_val.py)
  - Design_and_Metadata.zarr (the sim_id -> job_id mapping)

Writes: step9_master_per_sim_table.csv, the per-simulation table (927 rows,
185 flagged as validation) consumed by step10_error_vs_risk_analysis.py and
step11_figures.py.

Run from the repository root; Design_and_Metadata.zarr is a plain relative
path resolved against the repository root.
"""
import os
import numpy as np
import pandas as pd
import zarr

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

risk = pd.read_csv(os.path.join(OUT_DIR, "step6_failure_risk_scores_all_1000.csv"))
pod = pd.read_csv(os.path.join(OUT_DIR, "step7_pod_reconstruction_error.csv"))
node = pd.read_csv(os.path.join(OUT_DIR, "step8_node_error_val185.csv"))

root = zarr.open_group("Design_and_Metadata.zarr", mode="r")
job_index_per_sim = np.asarray(root["Mapping_indexes_and_metadata"]["job_index_per_sim"])
sim_ids_927 = np.arange(len(job_index_per_sim))  # sim_id IS the 0-based array index
job_map = pd.DataFrame({"sim_id": sim_ids_927, "job_id": job_index_per_sim.astype(int)})

assert set(job_map["sim_id"]) == set(pod["sim_id"]), "sim_id sets from job_map and POD error don't match"
assert set(job_map["job_id"]) == set(risk.loc[risk["converged"], "job_id"]), \
    "job_id sets from job_map and converged rows of risk table don't match"

master = pod.merge(job_map, on="sim_id", how="left")
master = master.merge(risk[["job_id", "tol", "Tcrit", "Vf", "mu", "kk1", "kk2", "kappa", "k1",
                              "p_fail_oof", "d_fail"]], on="job_id", how="left")
master = master.merge(node, on="sim_id", how="left")
master["is_validation"] = master["is_validation"].fillna(False)

assert master["p_fail_oof"].notna().all(), "some sims failed to join a p_fail score"
assert master["d_fail"].notna().all(), "some sims failed to join a d_fail score"
assert (master["is_validation"].sum()) == 185, "expected exactly 185 validation sims flagged"

out_path = os.path.join(OUT_DIR, "step9_master_per_sim_table.csv")
master.to_csv(out_path, index=False)
print(f"saved {out_path}")
print(f"n_rows={len(master)} (expect 927), n_validation={master['is_validation'].sum()} (expect 185)")
print(master[["sim_id", "job_id", "e_pod", "p_fail_oof", "d_fail", "e_node_A", "e_node_D"]].head())
print("DONE")
