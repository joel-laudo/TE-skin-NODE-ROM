"""Multivariate classifier for FE non-convergence.

step1_recover_failed_jobs.py's per-parameter Cohen's-d comparison looks at
one design parameter at a time. This script asks whether combining all 8
design parameters (plus two theoretically-motivated pairwise interactions)
predicts non-convergence better than any single parameter alone, using two
classifiers evaluated on the same data: class-weighted logistic regression
(with the interaction terms) and a class-weighted random forest.

Because the non-converged class is rare (73/1000 = 7.3% prevalence), PR-AUC
(area under the precision-recall curve) is used as the primary metric
rather than ROC-AUC -- under this much class imbalance, ROC-AUC can look
deceptively good even for a model that isn't separating the classes well.

Reads: all_1000_jobs_with_convergence_flag.csv (written by
step1_recover_failed_jobs.py).

Writes:
  - step2_classifier_feature_results.csv (per-feature coefficients /
    importances for both models)
  - step2_classifier_summary.csv (cross-validated PR-AUC/ROC-AUC per model)
These are the numbers SI Section 2 reports for the multivariate
non-convergence classifier.

Run from the repository root; all paths resolve relative to this script.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(OUT_DIR, "all_1000_jobs_with_convergence_flag.csv"))
FEATURES = ["tol", "Tcrit", "Vf", "mu", "kk1", "kk2", "kappa", "k1"]
X_raw = df[FEATURES].values
y = (~df["converged"]).astype(int).values  # 1 = failed/non-converged
print(f"n={len(y)}, n_failed={y.sum()}, prevalence={y.mean():.3%}")

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
Xdf = pd.DataFrame(X, columns=FEATURES)

# Interaction terms motivated by step1's per-parameter Cohen's-d ranking:
# k1*mu (the strongest single-parameter pair) and tol*kappa (the next
# strongest marginal signals), added to check whether joint effects carry
# predictive power beyond the two parameters' individual (additive) terms.
Xdf["k1_x_mu"] = Xdf["k1"] * Xdf["mu"]
Xdf["tol_x_kappa"] = Xdf["tol"] * Xdf["kappa"]
X_full = Xdf.values
feat_names = list(Xdf.columns)

# 5-fold stratified CV keeps the ~7.3% non-converged prevalence roughly
# constant across folds despite there being only 73 positive examples.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

# --- Logistic regression (class-weighted) ---
# class_weight="balanced" upweights the minority (non-converged) class so
# the fit isn't dominated by the 92.7% majority class.
logreg = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=123)
# cross_val_predict returns out-of-fold predictions: each job's predicted
# probability comes from a model that was NOT trained on that job (it was
# held out in that fold), so the PR-AUC/ROC-AUC computed below reflect
# honest generalization performance rather than in-sample fit.
logreg_proba = cross_val_predict(logreg, X_full, y, cv=cv, method="predict_proba")[:, 1]
logreg_pr_auc = average_precision_score(y, logreg_proba)
logreg_roc_auc = roc_auc_score(y, logreg_proba)
print(f"\nLogistic regression (class-weighted, 5-fold stratified CV):")
print(f"  PR-AUC  = {logreg_pr_auc:.4f}  (baseline/prevalence = {y.mean():.4f})")
print(f"  ROC-AUC = {logreg_roc_auc:.4f}  (baseline = 0.5)")

# Fit on full data for interpretable coefficients (standardized features already)
logreg_full = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=123)
logreg_full.fit(X_full, y)
coefs = pd.Series(logreg_full.coef_[0], index=feat_names).sort_values(key=np.abs, ascending=False)
print("\nStandardized logistic coefficients (sorted by |coef|):")
for name, c in coefs.items():
    print(f"  {name:14s} {c:+.4f}")

# --- Random forest (class-weighted) ---
# Shallow trees (max_depth=4) with many estimators (500) limit overfitting
# given only 73 positive examples; class_weight="balanced" again corrects
# for the class imbalance.
rf = RandomForestClassifier(n_estimators=500, max_depth=4, class_weight="balanced",
                              random_state=123, n_jobs=-1)
# Same out-of-fold logic as the logistic regression above.
rf_proba = cross_val_predict(rf, X_full, y, cv=cv, method="predict_proba")[:, 1]
rf_pr_auc = average_precision_score(y, rf_proba)
rf_roc_auc = roc_auc_score(y, rf_proba)
print(f"\nRandom forest (class-weighted, 5-fold stratified CV):")
print(f"  PR-AUC  = {rf_pr_auc:.4f}")
print(f"  ROC-AUC = {rf_roc_auc:.4f}")

rf_full = RandomForestClassifier(n_estimators=500, max_depth=4, class_weight="balanced",
                                   random_state=123, n_jobs=-1)
rf_full.fit(X_full, y)
# Permutation importance (rather than the forest's built-in impurity-based
# importance) avoids the bias impurity-based measures have toward
# high-cardinality continuous features; scoring="average_precision" keeps
# the importance ranking consistent with the PR-AUC metric used above.
perm = permutation_importance(rf_full, X_full, y, n_repeats=50, random_state=123,
                                scoring="average_precision", n_jobs=-1)
perm_series = pd.Series(perm.importances_mean, index=feat_names).sort_values(ascending=False)
print("\nPermutation importance (random forest, mean over 50 repeats, scoring=PR-AUC):")
for name, imp in perm_series.items():
    print(f"  {name:14s} {imp:+.4f}")

# Save results
results = pd.DataFrame({
    "feature": feat_names,
    "logreg_std_coef": [coefs.get(f, np.nan) for f in feat_names],
    "rf_permutation_importance": [perm_series.get(f, np.nan) for f in feat_names],
})
results.to_csv(os.path.join(OUT_DIR, "step2_classifier_feature_results.csv"), index=False)

summary = pd.DataFrame([
    {"model": "logistic_regression", "cv_pr_auc": logreg_pr_auc, "cv_roc_auc": logreg_roc_auc},
    {"model": "random_forest", "cv_pr_auc": rf_pr_auc, "cv_roc_auc": rf_roc_auc},
    {"model": "baseline_prevalence", "cv_pr_auc": y.mean(), "cv_roc_auc": 0.5},
])
summary.to_csv(os.path.join(OUT_DIR, "step2_classifier_summary.csv"), index=False)
print(f"\nsaved step2_classifier_feature_results.csv, step2_classifier_summary.csv")
print("DONE")
