"""Investigate why the random forest's permutation importance ranks kappa
above k1/mu, when the univariate Cohen's-d/Welch-t analysis (step1) and the
logistic regression (step2) both rank k1/mu highest and kappa did not
survive Bonferroni correction.

This script runs several diagnostic checks on that discrepancy: pairwise
correlations among the 8 raw design parameters, kappa's own bin-wise
failure rate, stability of the random forest's permutation importance
across random seeds, and held-out (cross-validated) permutation
importance -- to determine whether kappa's top ranking reflects a genuine
nonlinear signal or an artifact of a single model fit.

Reads: all_1000_jobs_with_convergence_flag.csv (written by
step1_recover_failed_jobs.py).

Run from the repository root; all paths resolve relative to this script.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(OUT_DIR, "all_1000_jobs_with_convergence_flag.csv"))
FEATURES = ["tol", "Tcrit", "Vf", "mu", "kk1", "kk2", "kappa", "k1"]
y = (~df["converged"]).astype(int).values

# 1. Check pairwise correlations among the 8 raw LHS parameters -- should be
#    near-zero if LHS did its job; a real correlation could explain kappa
#    "absorbing" k1/mu's signal.
corr = df[FEATURES].corr()
print("Pairwise correlation matrix (raw parameters):")
print(corr.round(3))
print("\nMax off-diagonal |correlation|:", corr.values[~np.eye(8, dtype=bool)].__abs__().max())

# 2. Kappa's own bin-wise failure probability (simple, direct check --
#    does kappa actually show a real marginal pattern, just one Cohen's-d
#    (a linear/location-shift statistic) doesn't capture well?)
kappa_bins = pd.qcut(df["kappa"], q=8, duplicates="drop")
bin_stats = df.groupby(kappa_bins, observed=True).apply(
    lambda g: pd.Series({"n": len(g), "n_failed": (~g["converged"]).sum(),
                           "fail_rate": (~g["converged"]).mean()})
)
print("\nKappa bin-wise failure rate (8 quantile bins):")
print(bin_stats)

# 3. Stability check: re-run RF permutation importance across several seeds
#    to see if kappa's #1 ranking is stable or an artifact of one particular
#    train/model fit.
Xdf = df[FEATURES].copy()
Xdf["k1_x_mu"] = (Xdf["k1"] - Xdf["k1"].mean()) / Xdf["k1"].std() * (Xdf["mu"] - Xdf["mu"].mean()) / Xdf["mu"].std()
feat_names = list(Xdf.columns)
X = Xdf.values

print("\nPermutation importance stability across 5 random seeds (RF, PR-AUC scoring):")
all_importances = []
for seed in [1, 2, 3, 4, 5]:
    rf = RandomForestClassifier(n_estimators=500, max_depth=4, class_weight="balanced",
                                  random_state=seed, n_jobs=-1)
    rf.fit(X, y)
    perm = permutation_importance(rf, X, y, n_repeats=30, random_state=seed,
                                    scoring="average_precision", n_jobs=-1)
    s = pd.Series(perm.importances_mean, index=feat_names)
    all_importances.append(s)
    top3 = s.sort_values(ascending=False).head(3)
    print(f"  seed={seed}: top3 = {list(top3.index)} ({[f'{v:.3f}' for v in top3.values]})")

avg_imp = pd.concat(all_importances, axis=1).mean(axis=1).sort_values(ascending=False)
print("\nAverage permutation importance across 5 seeds:")
for name, imp in avg_imp.items():
    print(f"  {name:14s} {imp:+.4f}")

# 4. Held-out (train/test split) permutation importance, to reduce in-sample
#    overfitting risk with only 73 positive examples.
print("\nHeld-out (5-fold CV) permutation importance (fit on train fold, score on held-out fold):")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
fold_importances = []
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    rf = RandomForestClassifier(n_estimators=500, max_depth=4, class_weight="balanced",
                                  random_state=123, n_jobs=-1)
    rf.fit(X[train_idx], y[train_idx])
    perm = permutation_importance(rf, X[test_idx], y[test_idx], n_repeats=30, random_state=123,
                                    scoring="average_precision", n_jobs=-1)
    fold_importances.append(pd.Series(perm.importances_mean, index=feat_names))
heldout_avg = pd.concat(fold_importances, axis=1).mean(axis=1).sort_values(ascending=False)
print("Average held-out permutation importance across 5 folds:")
for name, imp in heldout_avg.items():
    print(f"  {name:14s} {imp:+.4f}")

print("\nDONE")
