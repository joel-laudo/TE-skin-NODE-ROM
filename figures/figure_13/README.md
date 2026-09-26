# Figure 13 -- Ag true-vs-predicted scatter panels

## Entry point

```
python figures/figure_13/build_figure13_scatter.py
```

Run from the repository root, after `evaluation/generate_deliverables.py`
has produced `deliverables/deliverables_results.json`. Output is written
into this script's own directory (`figures/figure_13/`) via `__file__`:
`Model_{A,B,C,D}_panel.pdf/.png`.

## `dev_audit_scripts/`

Two read-only, one-off development-time diagnostics (identical in content
to the same-named scripts archived in `figures/figure_11/dev_audit_scripts/`
-- they were used together while preparing both Figure 11 and Figure 13),
archived here because they aren't part of reproducing the published figure.
Their internal paths were left untouched (still reference the old private
`phase36_deliverables/..._v2uncorrected_...` zarr convention) since they
aren't meant to run in this repo -- kept only for provenance:

- `check_model_name_attrs.py` -- printed each rollout zarr's `model_name`
  attribute as a quick sanity check.
- `verify_v2_zarr_completeness.py` -- verified all four v2 rollout zarrs
  have complete `disp_err_pointwise_rmse`/`t` fields for all 185 validation
  sims, and that all four models' inferred global-max support time agrees,
  before building the Figure 11/13 plots.

## Path fixes applied to `build_figure13_scatter.py`

Originally: (1) wrote output into a private-project-only
`revised_figures_11_13_v2/figure13/` folder computed from `PROJECT_ROOT`,
and (2) read `phase36_deliverables/phase36_results.json` under condition
keys `"{fam}_v2uncorrected"`. Changed to (1) write into this script's own
directory via `BASE = os.path.dirname(os.path.abspath(__file__))`, and (2)
read `deliverables/deliverables_results.json` (a plain relative path,
resolved against the repository-root cwd per this repo's convention) under
the renamed condition keys `"{fam}_v2"` (matching
`evaluation/generate_deliverables.py`'s renamed output). No plotting/math
logic (R^2 definition, tolerance band, styling) was changed.
