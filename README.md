# Stable Long-Horizon Neural ODE Reduced-Order Models for Biological Growth and Remodeling

Code, trained model checkpoints, and analysis scripts accompanying the
manuscript "Stable Long-Horizon Neural ODE Reduced-Order Models for
Biological Growth and Remodeling" (Joel Laudo & Adrian Buganza Tepole,
Columbia University).

This repository reproduces the methods, quantitative results, tables, and
figures reported in the manuscript and its Supplementary Information (SI).
See [data/DATA_AVAILABILITY.md](data/DATA_AVAILABILITY.md) for what data is
and isn't bundled here.

## Background

The manuscript builds a reduced-order model (ROM) of skin tissue expansion
(a growth-and-remodeling process) by combining:

1. **POD** (proper orthogonal decomposition) of finite-element (FE)
   displacement fields into a low-dimensional latent basis (r = 9 modes).
2. **Neural ODEs** trained to evolve that latent state forward in time,
   given the applied volumetric loading protocol and design parameters.
3. Four model variants (**A/B/C/D**) that differ in how much growth-state
   feedback is fed back into the latent dynamics: A is open-loop (no growth
   feedback), B uses a scalar net-area-gain feature, C uses a PCA-compressed
   growth-field feature, and D uses a learned CNN encoding of the full
   spatial growth field.

Growth (the tissue's own internal remodeling response) is integrated with
an implicit-Newton scheme matched exactly to the FE UMAT's own
time-discretization, both during training (differentiably) and at
evaluation time.

## Repository structure

```
campaign_fast_eval.py   Fast validation-set displacement-RMSE evaluator used
                         by every training/*.py script for BEST-checkpoint
                         selection (invoked as a subprocess -- see its
                         module docstring for why).
checkpoints/            Trained model weights (Models A-D, r=9) + SI-cited
                         ablation checkpoints, plus each checkpoint's
                         training/eval logs.
training/                Training scripts for the 4 reported models.
  ablations/              SI-cited training ablations (Model C Ag-loss-
                          throughout, Models A/D rollout-only, Models A/D
                          delayed-LR-decay 560-epoch).
evaluation/              Rollout/evaluation code for all 4 models, the
                         cross-model Ag comparison, and deliverable
                         (figure/table) generation.
  ablation_comparisons/   Scripts reproducing the SI's training-ablation
                          comparisons (Figure 4, Table 3).
data_processing/         Notebooks that build the POD displacement basis
                         and the growth-field PCA encoder from raw FE data.
figures/                 Manuscript figure regeneration (figures 9, 11-14).
main_text_analyses/      Section 4.1 (inference runtime benchmark) and
                         Section 4.2 (conditional utility of learned
                         growth feedback) analyses.
supplementary_analyses/  SI Figures 1-3 (area-growth tolerance derivation,
                         FE non-convergence effects, Model C outliers).
data/                    Small shared data files (mesh, design table,
                         growth-PCA encoder) + DATA_AVAILABILITY.md.
```

Each subfolder listed above has its own README (or a header docstring in
its main script) documenting exactly which manuscript/SI figure or table it
reproduces.

## Setup

```
python -m venv .venv
.venv\Scripts\activate            # or `source .venv/bin/activate` on Linux/Mac
pip install -r requirements.txt
```

`requirements.txt`'s pinned versions (torch, numpy, zarr, numba, matplotlib)
reflect the environment the checkpoints and logs were produced in. The zarr
pin matters: this codebase's zarr I/O helpers use an API removed in newer
zarr 3.x releases.

**Run every script from the repository root** (e.g.
`python evaluation/cross_model_comparison.py`, not from inside
`evaluation/`) -- all data, checkpoint, and output paths are plain relative
paths resolved against the current working directory.

## Reproducing results

Most evaluation and analysis scripts require the full FE dataset (zarr
archives of displacement/growth fields), which is not bundled in this repo
due to size -- see [data/DATA_AVAILABILITY.md](data/DATA_AVAILABILITY.md).
A few things can be reproduced without it:

- **Figure 14** (mechanical influence of growth): the intermediate
  regression output is already included, so
  `python figures/figure_14/plot_figure14_mechanical_influence.py`
  regenerates the published figure directly.
- **Loading trained checkpoints**: every file in `checkpoints/` is a plain
  `torch.load`-able state dict + metadata, usable independent of the FE
  dataset for architecture inspection.

With the FE dataset in place at the repository root, the typical pipeline
for the main cross-model comparison is:

```
python evaluation/generate_deliverables.py       # per-model validation rollouts + figures
python evaluation/cross_model_comparison.py      # cross-model Ag-error comparison
python evaluation/ablation_comparisons/paired_A_vs_D_bootstrap.py
python evaluation/ablation_comparisons/ablation_supplementary_figure.py
python evaluation/ablation_comparisons/delayeddecay_eval_and_comparison.py
```

Note: several `training/*.py` and `evaluation/eval_model_*.py` functions have
a `ckpt_path_template` default pointing at a `FINAL_models/...` path that
isn't part of this repo's layout (a training-run output convention, not a
path meant to be read from directly). The scripts above always pass the
real `checkpoints/...` path explicitly, so this only matters if you call
those functions directly with their defaults.

## License

MIT -- see [LICENSE](LICENSE).
