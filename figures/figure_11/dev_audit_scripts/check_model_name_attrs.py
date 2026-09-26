"""Archived development-time diagnostic (see ../README.md, "dev_audit_scripts/"):
print each rollout zarr's `model_name` attribute as a quick sanity check
before building the Figure 11/13 plots. Not part of reproducing either
figure. The paths below reference the old private-project
`phase36_deliverables/` layout and will not resolve in this repo -- kept
only for provenance."""
import zarr
for fam, path in {
    "A": "phase36_deliverables/A_v2uncorrected_val_rollouts_r9.zarr",
    "B": "phase36_deliverables/B_v2uncorrected_val_rollouts_r9.zarr",
    "C": "phase36_deliverables/C_v2uncorrected_val_rollouts_r9.zarr",
    "D": "phase36_deliverables/D_v2uncorrected_val_rollouts_r9.zarr",
}.items():
    root = zarr.open_group(path, mode="r")
    print(fam, dict(root.attrs))
