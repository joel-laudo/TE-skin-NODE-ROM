"""Archived development-time diagnostic (see ../README.md, "dev_audit_scripts/"):
inspect the CSV produced by the superseded draft `build_revised_figure9.py`
for zero/negative/non-finite values (including after moving-average
smoothing) to decide whether a log-scaled y-axis was feasible for each
rollout-loss panel. Not part of reproducing any reported figure. Archived
alongside the draft it depends on -- its input path won't resolve in this
repo since `revised_figure9_v2/` is not part of it."""
import csv
import numpy as np

rows = list(csv.DictReader(open("revised_figure9_v2/figure9_v2_data.csv")))
cols = ["rollout_raw", "rollout_plus_ag_pre_cap", "rollout_capped", "cap_attenuation"]
for col in cols:
    print("---", col, "---")
    for fam in "ABCD":
        vals = np.array([float(r[col]) for r in rows if r["family"] == fam and r[col] not in ("", "nan")])
        n_zero = int(np.sum(vals == 0.0))
        n_neg = int(np.sum(vals < 0.0))
        n_nonfinite = int(np.sum(~np.isfinite(vals)))
        print(f"  {fam}: n={len(vals)} min={vals.min():.6e} n_zero={n_zero} n_neg={n_neg} n_nonfinite={n_nonfinite}")

# Check the moving-average (window=11, NaN-aware) of cap_attenuation for zero/near-zero stretches
import sys
sys.path.insert(0, "revised_figure9_v2")
from build_revised_figure9 import moving_average_nan  # noqa: E402

print()
print("=== smoothed cap_attenuation (window=11) zero/near-zero check ===")
for fam in "ABCD":
    frows = [r for r in rows if r["family"] == fam]
    frows.sort(key=lambda r: int(r["epoch"]))
    y = np.array([float(r["cap_attenuation"]) for r in frows])
    ys = moving_average_nan(y, window=11)
    n_zero_smoothed = int(np.sum(ys == 0.0))
    n_nonpos_smoothed = int(np.sum(ys <= 0.0))
    print(f"  {fam}: n_zero(smoothed)={n_zero_smoothed} n_nonpositive(smoothed)={n_nonpos_smoothed} min(smoothed)={np.nanmin(ys):.6e}")
