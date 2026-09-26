"""Shared probing utilities: given some input features X and a target Y,
how well can X predict Y? Used by growth_decoding.py (does h_g encode
physical growth quantities?) and predictability.py (is h_g itself
predictable from Model A's own state?) to fit two kinds of probes on the
same train/val split:

  - linear_probe: closed-form ridge regression.
  - mlp_probe: a small 2-hidden-layer (width 128), GELU MLP, trained with
    Adam (lr=1e-3, 200 epochs, batch size 1024).

Both standardize X and Y using train-only mean/std, and report R^2 in that
normalized-target space (so scores are comparable across targets with very
different physical units/scales).
"""
import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d_in, hidden=128, d_out=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def _standardize_fit(X):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    return mean, std


def r2_per_dim(pred, true):
    """Coefficient of determination, computed independently for each output
    dimension: 1.0 means the probe's predictions perfectly track that
    dimension's variance across the validation set, 0.0 means it does no
    better than always predicting the (train-set) mean."""
    ss_res = np.sum((pred - true) ** 2, axis=0)
    ss_tot = np.sum((true - true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def linear_probe(X_train, Y_train, X_val, Y_val):
    """Closed-form ridge regression (small L2 for stability) on standardized
    (train-fit) features/targets. Returns dict with predictions + metrics."""
    x_mean, x_std = _standardize_fit(X_train)
    y_mean, y_std = _standardize_fit(Y_train)
    Xtr = (X_train - x_mean) / x_std
    Xva = (X_val - x_mean) / x_std
    Ytr = (Y_train - y_mean) / y_std
    Yva = (Y_val - y_mean) / y_std

    Xtr1 = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1))], axis=1)
    Xva1 = np.concatenate([Xva, np.ones((Xva.shape[0], 1))], axis=1)
    lam = 1e-4
    A = Xtr1.T @ Xtr1 + lam * np.eye(Xtr1.shape[1])
    B = Xtr1.T @ Ytr
    W = np.linalg.solve(A, B)

    pred_va = Xva1 @ W
    r2_dim = r2_per_dim(pred_va, Yva)
    r2_global = 1.0 - np.mean((pred_va - Yva) ** 2)
    rmse_norm = float(np.sqrt(np.mean((pred_va - Yva) ** 2)))
    return {
        "r2_global": float(r2_global),
        "r2_per_dim": r2_dim.tolist(),
        "rmse_normalized": rmse_norm,
        "pred_val_denorm": pred_va * y_std + y_mean,
    }


def mlp_probe(X_train, Y_train, X_val, Y_val, hidden=128, epochs=200, batch=1024, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    x_mean, x_std = _standardize_fit(X_train)
    y_mean, y_std = _standardize_fit(Y_train)
    Xtr = torch.tensor((X_train - x_mean) / x_std, dtype=torch.float32)
    Ytr = torch.tensor((Y_train - y_mean) / y_std, dtype=torch.float32)
    Xva = torch.tensor((X_val - x_mean) / x_std, dtype=torch.float32)
    Yva = torch.tensor((Y_val - y_mean) / y_std, dtype=torch.float32)

    d_in = X_train.shape[1]
    d_out = Y_train.shape[1]
    model = MLP(d_in, hidden=hidden, d_out=d_out)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.shape[0]

    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            loss = ((model(xb) - yb) ** 2).mean()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred_va = model(Xva).numpy()
    Yva_np = Yva.numpy()
    r2_dim = r2_per_dim(pred_va, Yva_np)
    r2_global = 1.0 - np.mean((pred_va - Yva_np) ** 2)
    rmse_norm = float(np.sqrt(np.mean((pred_va - Yva_np) ** 2)))
    return {
        "r2_global": float(r2_global),
        "r2_per_dim": r2_dim.tolist(),
        "rmse_normalized": rmse_norm,
        "pred_val_denorm": pred_va * y_std + y_mean,
        "model": model,
        "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std,
    }
