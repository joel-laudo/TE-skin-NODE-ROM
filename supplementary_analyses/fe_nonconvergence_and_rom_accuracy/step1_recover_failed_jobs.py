"""Recover the 73 non-converged FE job ids and their full 8-parameter LHS
design values.

Background: of the 1000 planned Abaqus finite-element simulations (a Latin
hypercube design over 8 material/geometry parameters), 927 converged and
were saved into the project's Zarr archive; the other 73 failed to converge
and were never individually saved anywhere. This script reconstructs their
design parameters by set difference: all 1000 job ids (from the full design
file) minus the 927 job ids that converged (read from
Design_and_Metadata.zarr's job_index_per_sim).

Reads:
  - PGOH_Ideal_tol_Tcrit_V_mu_kk1_kk2_kappa_k1_design.txt (the full 1000x9
    LHS design table: job_id plus 8 parameters)
  - Design_and_Metadata.zarr (job_index_per_sim: the 927 job ids that
    converged)

Writes (consumed by step2_classifier.py and later steps in this pipeline):
  - nonconverged_jobs_full.csv: the 73 non-converged jobs with their full
    design parameters
  - all_1000_jobs_with_convergence_flag.csv: all 1000 jobs, each flagged
    converged/not-converged

This is the first step of the SI Section 2 analysis of FE non-convergence
and ROM accuracy; the Cohen's-d comparison printed at the end reproduces the
converged-vs-non-converged parameter-distribution effect sizes reported
there.

Run from the repository root. DESIGN_PATH/ZARR_PATH are plain relative
paths, resolved against the current working directory, and both input
files are expected at the repository root.
"""
import os
import numpy as np
import pandas as pd
import zarr

DESIGN_PATH = "PGOH_Ideal_tol_Tcrit_V_mu_kk1_kk2_kappa_k1_design.txt"
ZARR_PATH = "Design_and_Metadata.zarr"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

COLS = ["job_id", "tol", "Tcrit", "Vf", "mu", "kk1", "kk2", "kappa", "k1"]

# Full 1000-row design table
design = np.loadtxt(DESIGN_PATH)
assert design.shape == (1000, 9), f"expected (1000,9), got {design.shape}"
df = pd.DataFrame(design, columns=COLS)
df["job_id"] = df["job_id"].astype(int)

# 927 job ids that actually converged
root = zarr.open_group(ZARR_PATH, mode="r")
job_index_per_sim = np.asarray(root["Mapping_indexes_and_metadata"]["job_index_per_sim"])
assert job_index_per_sim.shape == (927,), f"expected (927,), got {job_index_per_sim.shape}"
converged_ids = set(int(j) for j in job_index_per_sim)

all_ids = set(df["job_id"].tolist())
assert all_ids == set(range(1, 1001)), "job ids are not exactly 1..1000"

failed_ids = sorted(all_ids - converged_ids)
assert len(failed_ids) == 73, f"expected 73 failed jobs, got {len(failed_ids)}"

df["converged"] = df["job_id"].isin(converged_ids)
df_failed = df[df["job_id"].isin(failed_ids)].copy()
df_failed = df_failed.sort_values("job_id").reset_index(drop=True)

out_path = os.path.join(OUT_DIR, "nonconverged_jobs_full.csv")
df_failed.to_csv(out_path, index=False)
df.to_csv(os.path.join(OUT_DIR, "all_1000_jobs_with_convergence_flag.csv"), index=False)

print(f"Recovered {len(failed_ids)} failed job ids (verified 1000-927=73).")
print(f"saved {out_path}")
print(f"saved {os.path.join(OUT_DIR, 'all_1000_jobs_with_convergence_flag.csv')}")

# Sanity check: recompute the converged-vs-non-converged Cohen's d for each
# parameter, reproducing the effect sizes reported in SI Section 2.
for param in ["mu", "k1", "tol", "kappa"]:
    conv_mean = df.loc[df["converged"], param].mean()
    fail_mean = df.loc[~df["converged"], param].mean()
    conv_std = df.loc[df["converged"], param].std(ddof=1)
    fail_std = df.loc[~df["converged"], param].std(ddof=1)
    n1, n2 = df["converged"].sum(), (~df["converged"]).sum()
    pooled_std = np.sqrt(((n1 - 1) * conv_std**2 + (n2 - 1) * fail_std**2) / (n1 + n2 - 2))
    cohens_d = (fail_mean - conv_mean) / pooled_std
    print(f"  {param}: converged_mean={conv_mean:.4f} nonconverged_mean={fail_mean:.4f} cohens_d={cohens_d:.3f}")

print("DONE")
