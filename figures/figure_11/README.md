# Figure 11 -- Validation displacement-RMSE trajectories and final-error histograms

## Entry points

```
python figures/figure_11/build_figure11_trajectories.py
python figures/figure_11/build_figure11_histograms.py
```

Run from the repository root, after `evaluation/generate_deliverables.py`
has produced `deliverables/{A,B,C,D}_v2_val_rollouts_r9.zarr`. Each script
resolves its own output location via `__file__` (writes into
`figures/figure_11/` regardless of caller's cwd), but the input zarr paths
are plain relative paths resolved against the current working directory
(repository root), matching the rest of this repo's convention.

- `build_figure11_trajectories.py` -> `Model_{A,B,C,D}_displacement_trajectory.pdf/.png`
  (median + 10th-90th percentile displacement-RMSE trajectory per model).
- `build_figure11_histograms.py` -> `Model_{A,B,C,D}_final_error_histogram.pdf/.png`
  (final-time surface-displacement-RMSE histogram per model, using each
  model's exact original bin count / axis limits: A=5 bins, B/D=20 bins,
  C=60 bins with no xlim so its ~25.9 max outlier stays visible).

## `dev_audit_scripts/`

Two read-only, one-off development-time diagnostics from the manuscript
revision, archived (not deleted) because they aren't part of reproducing the
published figure. Their internal paths were left untouched (still reference
the old private-project `phase36_deliverables/..._v2uncorrected_...` zarr
convention) since they are not meant to run in this repo -- kept only for
provenance:

- `check_model_name_attrs.py` -- printed each rollout zarr's `model_name`
  attribute as a quick sanity check before building the Figure 11/13 plots.
- `verify_v2_zarr_completeness.py` -- verified all four v2 rollout zarrs have
  complete `disp_err_pointwise_rmse`/`t` fields for all 185 validation sims,
  and that all four models' inferred global-max support time agrees, before
  building the plots.

## Path/import fixes applied to the kept scripts

Both `build_figure11_trajectories.py` and `build_figure11_histograms.py`
originally: (1) wrote output into a private-project-only
`revised_figures_11_13_v2/figure11/` folder computed from `PROJECT_ROOT`,
and (2) read `phase36_deliverables/{fam}_v2uncorrected_val_rollouts_r9.zarr`.
Both were changed to (1) write into this script's own directory via
`BASE = os.path.dirname(os.path.abspath(__file__))`, and (2) read
`deliverables/{fam}_v2_val_rollouts_r9.zarr` (matching
`evaluation/generate_deliverables.py`'s renamed output convention). The
`PROJECT_ROOT` computation itself (3 `os.path.dirname` calls from `__file__`)
was already correct for this repo's `figures/figure_11/` depth and was kept
unchanged, just renamed `BASE`/`PROJECT_ROOT` for clarity.
