# Figure 9 -- Training diagnostics (Models A-D)

## Entry point

```
python figures/figure_09/build_figure9_v2_final.py
```

Produces `figure9_v2_final.pdf` / `.png` (the published Figure 9: validation
one-step loss, median validation displacement RMSE, mean pre-cap rollout
objective, mean capped rollout objective, all log-scaled, one 2x2 panel per
model family A-D) and `figure9_v2_final_data.csv` (the plotted data).

Reads `checkpoints/Model_{A,B,C,D}_v2_loss_data.txt` and
`checkpoints/Model_{A,B,C,D}_v2_disp_rmse_eval_log.txt`, resolved relative to
this script's own directory via `__file__` -- it can be run from anywhere,
though the repo convention is to run from the repository root.

## Supplementary table

```
python figures/figure_09/build_360_vs_560_best_table.py
```

Builds the consistent 360-vs-560-epoch (delayed-LR-decay) BEST-checkpoint
supplement table + paired A-vs-D bootstrap (`table_360_vs_560_best_results.json`),
combining each run's own disp-RMSE-selected BEST checkpoint for both
displacement and Ag metrics. This is a genuine data-aggregation step (not a
diagnostic) but it is **not self-contained**: it reads already-computed
outputs and must be run from the repository root, after:

```
python evaluation/generate_deliverables.py
python evaluation/ablation_comparisons/delayeddecay_eval_and_comparison.py
```

which produce `deliverables/A_v2_val_rollouts_r9.zarr`,
`deliverables/D_v2_val_rollouts_r9.zarr`, `deliverables/deliverables_results.json`
(keys `"A_v2"`, `"D_v2"`), and `deliverables/delayeddecay560_results.json`
(keys `"A_delayed"`, `"D_delayed"`).

## `dev_audit_scripts/`

Seven scripts from the manuscript-revision process, archived here (not
deleted) because they are one-off development-time audits/diagnostics, not
part of reproducing the published figure. None of their internal paths have
been fixed/updated -- they reference private-project-only locations
(`FINAL_models/...`, `revised_figure9_v2/...`, `phase36_deliverables/...`,
`campaign_fast_eval.py`) that don't exist in this repo, and are kept purely
for provenance/context:

- `audit_sparse_eval_delayeddecay.py` -- exhaustive retrospective checkpoint
  sweep (epochs 301-500) for the 560-epoch delayed-decay runs, to check the
  training-time-selected BEST checkpoint was truly the global minimum.
  Depends on the private project's `campaign_fast_eval.py` and per-epoch
  checkpoint archive, neither of which is in this repo.
- `find_true_global_best.py` -- combines `audit_sparse_eval_delayeddecay.py`'s
  stdout logs with the training-time `disp_rmse_eval_log.txt` to compute the
  true global-minimum epoch. Depends on that same excluded checkpoint
  archive and on stdout log files not present in this repo.
- `verify_checkpoint_identity.py` -- one-off tensor-equality check that each
  reported `*_BEST.pt` is bit-identical to the corresponding `epoch_NNN.pt`
  snapshot. Depends on the private project's `epoch_checkpoints/` archives.
- `build_revised_figure9.py` -- an earlier draft of the Figure 9 panel
  composition (disp-RMSE / pre-cap rollout / a derived "cap attenuation"
  panel / capped rollout, in linear- and log-axis variants), superseded by
  `build_figure9_v2_final.py`'s different, final 4-panel composition
  (val-loss / disp-RMSE / pre-cap rollout / capped rollout, all log-scaled).
  Its outputs remain in this folder for provenance
  (`figure9_v2_linear.*`, `figure9_v2_log_panelA.*`, `figure9_v2_log_all_panels.*`,
  `figure9_v2_data.csv`) but are **not** the published figure.
- `check_log_feasibility.py` -- diagnostic that inspected
  `figure9_v2_data.csv` (produced by the superseded `build_revised_figure9.py`,
  which it imports directly) for zero/negative values before deciding whether
  a log y-axis was feasible for each panel. Archived alongside the draft it
  depends on.
- `check_sim804_median_rank.py` and `verify_fig11_zarr_provenance.py` --
  read-only checks against `FINAL_models/Model_{A,B,C,D}/Model_*_val_rollouts_r9.zarr`,
  i.e. the **pre-revision, pre-"v2"** checkpoint family excluded entirely
  from this repo (no "_ablation_fullrollout_v2" suffix -- see the root
  README's note on excluded superseded checkpoints). They are diagnostics
  from the manuscript-revision process confirming that the (excluded, older)
  rollout zarrs matched earlier notebook-computed numbers; not part of
  reproducing any reported figure.

## Note on `checkpoints/Model_*_v2_loss_data.txt`

`build_figure9_v2_final.py` needs each model's full per-epoch loss log
(`epoch,train_loss,val_loss,rollout_raw,rollout_plus_ag_pre_cap,rollout_capped`),
which is a different file from `checkpoints/Model_*_v2_training_log_r9.txt`
(the raw stdout log, with an unrelated column schema:
`epoch,train_loss,val_loss,best_val,roll_report_avg,...` -- no rollout/cap
columns). The four `checkpoints/Model_{A,B,C,D}_v2_loss_data.txt` files were
copied into this repo's `checkpoints/` directory (from each model's private
`FINAL_models/Model_*_ablation_fullrollout_v2/Model_*_r9_loss_data.txt`) as
part of this fix, using the same `_v2_` naming convention as the other
already-copied per-checkpoint logs.
