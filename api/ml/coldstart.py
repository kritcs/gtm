"""Cold start for a product the graph has never seen.

1. Description prior: a ridge map g_phi from text embeddings of tool
   descriptions to their LightGCN item embeddings gives e0 = g_phi(description).
2. Fold-in: optimize only the new item's embedding on its few known customers
   with BPR, everything else frozen. With three customers an unregularized
   fit memorizes them, so this is MAP estimation: an L2 pull toward e0.

       e* = argmin_e  sum_(c+, c-) -log sigma(<E_c+, e> - <E_c-, e>) + mu * ||e - e0||^2
"""
from __future__ import annotations

import numpy as np


def fit_ridge(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    Xb = np.hstack([X, np.ones((len(X), 1), X.dtype)])
    A = Xb.T @ Xb + alpha * np.eye(Xb.shape[1], dtype=X.dtype)
    return np.linalg.solve(A, Xb.T @ Y)


def apply_ridge(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.append(x, 1.0).astype(W.dtype) @ W


def fold_in(user_emb: np.ndarray, e0: np.ndarray, customers: list[int], mu: float = 0.5,
            steps: int = 60, lr: float = 0.05, n_neg: int = 64, seed: int = 0) -> tuple[np.ndarray, list[float]]:
    if not customers:
        return e0.copy(), []
    rng = np.random.default_rng(seed)
    e = e0.astype(np.float64).copy()
    cust = np.array(customers)
    mask = np.ones(len(user_emb), bool)
    mask[cust] = False
    pool = np.flatnonzero(mask)
    curve = []
    for _ in range(steps):
        neg = rng.choice(pool, n_neg)
        P = user_emb[np.repeat(cust, n_neg // len(cust) + 1)[:n_neg]]
        Nn = user_emb[neg]
        x = (P - Nn) @ e
        curve.append(float(np.mean(np.logaddexp(0, -x)) + mu * np.sum((e - e0) ** 2)))
        g = -((1 / (1 + np.exp(x)))[:, None] * (P - Nn)).mean(0) + 2 * mu * (e - e0)
        e -= lr * g
    return e.astype(np.float32), curve
