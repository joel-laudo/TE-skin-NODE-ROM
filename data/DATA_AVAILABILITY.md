# Data availability

This repository contains code, trained model checkpoints, and small derived
artifacts, but **not** the full finite-element (FE) dataset the models were
trained and evaluated on. The FE dataset (185 simulations spanning the
7-parameter design space described in the manuscript) totals many gigabytes
of raw nodal displacement fields, internal-variable growth snapshots, and
mesh files, stored as zarr archives during the original study.

## What's included in this repo

The four files below ship at the **repository root** (not inside `data/`)
because every script in this repo resolves them as plain relative paths
against the current working directory, and the convention throughout this
repo is to run scripts with cwd = repository root (see below). `data/`
holds only this availability statement.

- `GOH_Nodes_Test_for_Visualization.csv`, `GOH_Elements_Test_for_Visualization.csv` --
  the bilayer tissue-expansion mesh (node coordinates and element
  connectivity). Small (~1 MB total) and required by every evaluation/
  script to reconstruct the bottom-surface mesh.
- `PGOH_Ideal_tol_Tcrit_V_mu_kk1_kk2_kappa_k1_design.txt` -- the 7-parameter
  design table (one row per simulated FE job), used to look up each
  simulation's growth parameters (theta_crit, k1, k2) and material/BC
  parameters (tol, mu, kk1, kk2, kappa).
- `growth_pca_trainonly_H60_W60_k8_val0.20_seed123.npz` -- the fitted
  growth-field PCA encoder used by Model C's growth-feedback feature (see
  `data_processing/README.md` for its provenance and a caveat about its
  component count).
- `cache/final_ag_per_sim_stats_r9.npz` -- per-simulation final-net-area-gain
  mean/std, used by Model B (`evaluation/eval_model_b.py`) to normalize its
  scalar Ag growth-feedback feature. Small (~13 KB).
- Small, already-computed intermediate/derived artifacts needed for
  specific figures/tables (e.g. `figures/figure_14/spatial_effect_maps.npz`,
  the cached CSV/JSON outputs under `supplementary_analyses/` and
  `main_text_analyses/`) -- each such file is documented in its own
  subfolder's README or in the producing script's header.

## What's NOT included, and how to obtain it

The raw FE simulation outputs (zarr archives of nodal displacements,
element-wise growth stretches, expander volumes, and per-snapshot design
metadata) are not bundled here due to size. These are required to:

- Re-run `training/*.py` (fresh model training).
- Re-run `evaluation/cross_model_comparison.py` and
  `evaluation/generate_deliverables.py` (full validation-set rollouts and
  Ag comparisons).
- Regenerate `figures/figure_14/spatial_effect_maps.npz` from scratch via
  `figures/figure_14/build_spatial_effect_maps.py`.
- Re-run most of `supplementary_analyses/` and `main_text_analyses/`.

**The FE dataset is available upon reasonable request.** Contact the
corresponding author (Adrian Buganza Tepole, Columbia University) to
request access.
Once obtained, place the zarr archives
(`displacements.zarr`, `ip_growth_elem.zarr`, `expd_volumes.zarr`,
`Design_and_Metadata.zarr`) at the repository root -- every script in this
repo resolves data paths as plain relative paths against the current
working directory, so **run all scripts with the repository root as the
working directory** (e.g. `python evaluation/cross_model_comparison.py`
from the repo root, not from inside `evaluation/`).

## What you CAN reproduce without the FE dataset

- Loading and inspecting the trained checkpoints in `checkpoints/`.
- Regenerating the manuscript Figure 14 plot from the included
  `spatial_effect_maps.npz` via
  `python figures/figure_14/plot_figure14_mechanical_influence.py`.
- Re-running the cross-model comparison bootstrap/figure scripts under
  `evaluation/ablation_comparisons/` and `supplementary_analyses/` that
  consume only the already-included cached JSON/zarr outputs (documented
  per-script).
