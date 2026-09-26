# Figure 12 -- z1-z2 latent phase portraits + 3D displacement panels (scripts/)

This README covers the file classification and path/import fixes made to
adapt these scripts to this repo.

## Entry points

Run from the repository root (checkpoints/ and the FE-derived data files
must be resolvable as plain relative paths, matching every other script in
this repo):

```
python figures/figure_12/scripts/build_figure12_panels.py
python figures/figure_12/scripts/build_figure12_disp3d_panels.py
```

- `build_figure12_panels.py` -> `../panels/fig12_Model{A,D}_sim804_step{10,46,64,68}.{png,pdf}`
  (z1-z2 latent streamplot + trajectory overlay panels).
- `build_figure12_disp3d_panels.py` -> `../panels_disp3d/fig12_disp3d_Model{A,D}_sim804_step{10,46,64,68}.{png,pdf}`
  (companion 3D reconstructed-displacement-magnitude panels).

Both import their plotting logic from `z1z2_plot_functions.py` and (for the
3D panels) `disp3d_plot_function.py`, both of which are genuinely-needed
dependencies (category 2) -- verbatim-extracted plotting functions from the
original notebook, kept as importable modules rather than re-extracted each
run.

## Path/import fixes applied

All four kept files (`build_figure12_panels.py`, `build_figure12_disp3d_panels.py`,
`z1z2_plot_functions.py`, `disp3d_plot_function.py`) originally imported
model-construction/rollout helpers from three private-project-only modules
that don't exist in this repo: `eval_ablation_model_a_base_closureMLP.py`,
`eval_rollout_cnn.py`, `model_d_stabilized_eval_lib.py`. Fixed to import the
equivalent, cleaned functions from this repo's `evaluation/` package
instead:

| Old import | New import |
|---|---|
| `eval_ablation_model_a_base_closureMLP.{build_vanilla_node_from_ckpt,load_raw_data,rollout_single_sim_vanilla_node,load_bottom_surface_mesh_direct}` | `evaluation/eval_model_a.py` (same names) |
| `eval_rollout_cnn.rollout_single_sim_cnn_node_matched` | `evaluation/eval_model_d.py` (same name) |
| `model_d_stabilized_eval_lib.build_cnn_node_from_ckpt` | `evaluation/shared_eval_utils.py` (same name; `eval_model_d.py` itself imports it from there too) |

`z1z2_plot_functions.py` and `disp3d_plot_function.py` each add
`sys.path.insert(0, .../figure_12/scripts/../../../evaluation)` (3 levels up
from `figure_12/scripts/` to the repository root, then into `evaluation/`)
before importing, so they resolve correctly regardless of the caller's
working directory. Both were confirmed to import cleanly (see below).

`build_figure12_panels.py` and `build_figure12_disp3d_panels.py` additionally:
- Dropped `sys.path.insert(0, PROJECT_ROOT)` and `os.chdir(PROJECT_ROOT)`
  (a private-root hack to make the extracted functions' bare-relative data
  paths resolve). No longer needed: `z1z2_plot_functions.py` /
  `disp3d_plot_function.py` now resolve their own `evaluation/` import via
  `__file__`, and this repo's convention (see root README) is simply to run
  every script from the repository root so bare-relative data paths (e.g.
  `GOH_Nodes_Test_for_Visualization.csv`, `checkpoints/...`) resolve on
  their own -- no chdir needed. Note the original `PROJECT_ROOT` computation
  (3 nested `os.path.dirname()` calls from `__file__`) was actually off by
  one directory level for this repo's `figures/figure_12/scripts/` depth
  (it resolved to `figures/`, not the repository root) -- moot now that it's
  removed, but flagged here in case it's informative.
- Changed output directories from a private `revised_figure12_v2/panels/`
  (`panels_disp3d/`) location to this repo's actual `figures/figure_12/panels/`
  and `figures/figure_12/panels_disp3d/` folders (computed via `__file__`,
  matching where the existing panel PNGs/PDFs already live).
- Changed `ckpt_path_template` from `FINAL_models/Model_{A,D}_ablation_fullrollout_v2/Model_{A,D}_{Vanilla,CNN}_r{r_modes}_BEST.pt`
  to `checkpoints/Model_{A,D}_v2_BEST.pt`.

## `dev_audit_scripts/`

Three one-off development-time scripts, archived (not deleted) because
they're not part of reproducing the published panels -- their internal
paths were left untouched since they're not meant to run in this repo:

- `extract_plot_functions.py` and `extract_3d_disp_function.py` -- one-time
  extraction scripts that read `Visualize_Latent_Trajectories_extracted.py`
  (a private, notebook-converted script that does not exist in this repo)
  and wrote out `z1z2_plot_functions.py` / `disp3d_plot_function.py`
  respectively. Those two output modules are now the checked-in, directly
  maintained artifacts (fixed above) -- the extraction scripts that
  originally produced them are no longer re-run.
- `numerics_and_preview.py` -- printed the sim-804 quantitative comparison
  numbers (Section E of the accompanying report) and built
  `preview_combined_grid.png`, an internal-review-only montage explicitly
  *not* a manuscript deliverable (see the report's Section G/H). Depends on
  `campaign_fast_eval.py`, a private-project-only module not present in
  this repo.

## Verification performed

- All four kept `.py` files compile cleanly (`py_compile`).
- `z1z2_plot_functions.py` and `disp3d_plot_function.py` were imported
  directly (`import z1z2_plot_functions`, `import disp3d_plot_function`)
  with cwd set to the repository root and `figures/figure_12/scripts` on
  `sys.path` -- both succeeded, confirming the `evaluation/` imports
  (`eval_model_a`, `eval_model_d`, `shared_eval_utils`) resolve correctly.
- `build_figure12_panels.py` / `build_figure12_disp3d_panels.py` were **not**
  run end-to-end: doing so requires the full FE dataset (for
  `eval_model_a.load_raw_data`/mesh loading) which is not available in this
  verification environment. Only import-time and path-construction
  correctness were confirmed by inspection and by the successful module
  imports above.
