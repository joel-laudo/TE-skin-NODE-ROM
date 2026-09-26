"""Small helpers for summarizing a raw element-wise growth-stretch field
(lambda_g_x, lambda_g_y) into simple scalar/physical quantities. These are
used throughout the Section 4.2 conditional-utility analysis (feature
extraction, decoding probes, structure/correlation plots) as the
ground-truth physical targets that h_g (the CNN's learned growth feature)
is checked against.
"""
import numpy as np


def mean_lambda_g(lamdag_elem: np.ndarray) -> tuple[float, float]:
    """lamdag_elem: (Ne,2) raw [lambda_g_x, lambda_g_y]. Returns (mean_lam1, mean_lam2),
    the spatial average in-plane growth stretch along each of the two
    growth directions."""
    return float(np.mean(lamdag_elem[:, 0])), float(np.mean(lamdag_elem[:, 1]))


def growth_anisotropy(lamdag_elem: np.ndarray) -> float:
    """Simple anisotropy measure: mean(lambda_g_x) - mean(lambda_g_y).
    Positive values mean the tissue is growing more along the x-direction
    than the y-direction; zero means isotropic (equal) growth."""
    m1, m2 = mean_lambda_g(lamdag_elem)
    return float(m1 - m2)


def project_onto_growth_pca(lamdag_elem: np.ndarray, H: int, W: int, pca_npz) -> np.ndarray:
    """Project a raw (Ne=H*W, 2) growth-stretch snapshot onto the existing
    train-only growth-PCA basis, reproducing X_pca_all_norm's own convention
    bit-for-bit: per-channel PCA (mean_x/components_x, mean_y/components_y),
    concatenated, then normalized by the basis's own train-only
    pca_feat_mean/pca_feat_std.

    pca_npz: an already-loaded np.lib.npyio.NpzFile (growth_pca_trainonly_...npz).
    Returns: (2k,) normalized coefficient vector, matching X_pca_all_norm's row format.
    """
    gx = lamdag_elem[:, 0].reshape(H * W)
    gy = lamdag_elem[:, 1].reshape(H * W)

    mean_x = pca_npz["mean_x"]      # (H*W,)
    comp_x = pca_npz["components_x"]  # (k, H*W)
    mean_y = pca_npz["mean_y"]
    comp_y = pca_npz["components_y"]

    cx = (gx - mean_x) @ comp_x.T   # (k,)
    cy = (gy - mean_y) @ comp_y.T   # (k,)
    coeffs = np.concatenate([cx, cy])  # (2k,)

    feat_mean = pca_npz["pca_feat_mean"].reshape(-1)  # (2k,)
    feat_std = pca_npz["pca_feat_std"].reshape(-1)
    return (coeffs - feat_mean) / feat_std
