# Data Processing: POD and Growth-POD Basis Construction

- **`Build_POD_Bases.ipynb`** : constructs the r=9
  displacement POD basis (`displacements.zarr/pod_full/{U,mean}`) used by all
  four models, from the 742 training-set displacement snapshots. Retains 9
  modes (99.99% cumulative explained variance threshold), matching Methods
  Section 2 of the manuscript.

- **`Generate_Growth_POD_Components.ipynb`** : constructs the growth-POD
  basis used by Model C (8 modes per growth-stretch channel, 16 total),
  producing `growth_pca_trainonly_H60_W60_k8_val0.20_seed123.npz`.

  **Known provenance caveat**: this notebook's currently-saved `N_COMPONENTS`
  parameter is `4`, which would regenerate an 8-total-feature (`k4`) basis if
  re-run as-is — not the `k8` (16-total-feature) basis actually used
  throughout the reported pipeline. The `k8` artifact on disk was produced by
  an earlier execution of the same cell with `N_COMPONENTS = 8`. **To
  regenerate the basis actually used for the reported results, set
  `N_COMPONENTS = 8` before running this notebook.** This does not affect
  any reported number (the on-disk `k8` artifact is confirmed correct and is
  what every eval/training script actually reads), it only affects whether
  *re-running this notebook from scratch* reproduces the k8 or k4 basis.
