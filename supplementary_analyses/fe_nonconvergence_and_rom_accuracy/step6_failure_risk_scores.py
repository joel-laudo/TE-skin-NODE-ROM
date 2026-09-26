"""Compute out-of-fold FE-failure-risk scores (p_fail) and nearest-neighbor
distance-to-failure (d_fail) for all 1000 LHS jobs.

Reuses the classifier setup from step2_classifier.py (class-weighted
random forest, 5-fold stratified CV, seed=123), with one difference: the
canonical model here uses only the 8 raw sampled parameters, with no
hand-added interaction terms, since random-forest trees can already
capture feature interactions from the raw inputs without needing them
added explicitly. step2's interaction-augmented RF is also refit here, but
purely as a consistency check against the canonical (raw-feature) score --
not as the canonical score itself.

cross_val_predict(..., method="predict_proba") returns out-of-fold
predictions by construction: each fold's predicted probabilities come from
a model that never saw that fold during fitting, so they are legitimate
held-out risk scores rather than in-sample fits. step2_classifier.py
computed these correctly but only saved the aggregate PR-AUC/ROC-AUC, not
the per-job probabilities; this script saves the per-job p_fail values
needed for the risk-score analysis in the later steps of this pipeline.

Reads: all_1000_jobs_with_convergence_flag.csv (written by
step1_recover_failed_jobs.py).

Writes (consumed by later steps in this pipeline):
  - step6_failure_risk_scores_all_1000.csv: p_fail_oof (out-of-fold
    canonical risk score), p_fail_oof_interact_check (consistency-check
    score), and d_fail (nearest-neighbor distance to the failed-job set,
    using the same normalization convention as
    step5_nearest_neighbor_holes.py) for all 1000 jobs
  - step6_classifier_consistency_check.csv: PR-AUC/ROC-AUC for both RF
    variants plus their correlation, confirming the two variants agree

Run from the repository root; all paths resolve relative to this script.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.spatial import cKDTree

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["tol", "Tcrit", "Vf", "mu", "kk1", "kk2", "kappa", "k1"]

df = pd.read_csv(os.path.join(OUT_DIR, "all_1000_jobs_with_convergence_flag.csv"))
y = (~df["converged"]).astype(int).values
X_raw = df[FEATURES].values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

# --- Canonical: RF on the 8 raw sampled parameters only, unscaled (RF is scale-invariant) ---
rf_canonical = RandomForestClassifier(n_estimators=500, max_depth=4, class_weight="balanced",
                                        random_state=123, n_jobs=-1)
p_fail_oof = cross_val_predict(rf_canonical, X_raw, y, cv=cv, method="predict_proba")[:, 1]
pr_auc = average_precision_score(y, p_fail_oof)
roc_auc = roc_auc_score(y, p_fail_oof)
print(f"Canonical RF (8 raw features, out-of-fold): PR-AUC={pr_auc:.4f} (baseline={y.mean():.4f}), "
      f"ROC-AUC={roc_auc:.4f}")

# --- Consistency check: step2's interaction-augmented version, same CV setup ---
Xdf = df[FEATURES].copy()
Xdf["k1_x_mu"] = ((Xdf["k1"] - Xdf["k1"].mean()) / Xdf["k1"].std()) * \
                  ((Xdf["mu"] - Xdf["mu"].mean()) / Xdf["mu"].std())
Xdf["tol_x_kappa"] = ((Xdf["tol"] - Xdf["tol"].mean()) / Xdf["tol"].std()) * \
                       ((Xdf["kappa"] - Xdf["kappa"].mean()) / Xdf["kappa"].std())
rf_interact = RandomForestClassifier(n_estimators=500, max_depth=4, class_weight="balanced",
                                       random_state=123, n_jobs=-1)
p_fail_oof_interact = cross_val_predict(rf_interact, Xdf.values, y, cv=cv, method="predict_proba")[:, 1]
pr_auc_i = average_precision_score(y, p_fail_oof_interact)
roc_auc_i = roc_auc_score(y, p_fail_oof_interact)
print(f"Interaction-augmented RF (consistency check): PR-AUC={pr_auc_i:.4f}, ROC-AUC={roc_auc_i:.4f}")
corr_between = np.corrcoef(p_fail_oof, p_fail_oof_interact)[0, 1]
print(f"Pearson correlation between the two RF variants' out-of-fold probabilities: {corr_between:.4f}")
if abs(pr_auc - pr_auc_i) > 0.15 or corr_between < 0.7:
    print("  [FLAG] canonical and interaction-augmented RF differ substantially -- investigate before proceeding.")
else:
    print("  Consistent -- proceeding with the canonical (8-raw-feature) RF as p_fail.")

df["p_fail_oof"] = p_fail_oof
df["p_fail_oof_interact_check"] = p_fail_oof_interact

# --- Nearest-neighbor distance-to-failure (d_fail); same normalization
#     convention as step5_nearest_neighbor_holes.py ---
mins, maxs = X_raw.min(axis=0), X_raw.max(axis=0)
X_norm = (X_raw - mins) / (maxs - mins)
fail_mask = (~df["converged"]).values
X_fail_norm = X_norm[fail_mask]
tree_fail = cKDTree(X_fail_norm)
d_fail_all, _ = tree_fail.query(X_norm, k=1)
# For the 73 failed points themselves, distance to nearest OTHER failed point (k=2, since k=1 is self)
d_fail_all_k2, _ = tree_fail.query(X_norm, k=2)
d_fail_all[fail_mask] = d_fail_all_k2[fail_mask, 1]
df["d_fail"] = d_fail_all

out_path = os.path.join(OUT_DIR, "step6_failure_risk_scores_all_1000.csv")
df.to_csv(out_path, index=False)
print(f"\nsaved {out_path}")

pd.DataFrame([
    {"model": "canonical_rf_8raw", "cv_pr_auc": pr_auc, "cv_roc_auc": roc_auc},
    {"model": "interaction_augmented_rf", "cv_pr_auc": pr_auc_i, "cv_roc_auc": roc_auc_i},
    {"model": "pearson_corr_between_variants", "cv_pr_auc": corr_between, "cv_roc_auc": np.nan},
]).to_csv(os.path.join(OUT_DIR, "step6_classifier_consistency_check.csv"), index=False)
print("saved step6_classifier_consistency_check.csv")
print("DONE")
