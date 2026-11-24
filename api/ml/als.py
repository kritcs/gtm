"""Implicit-feedback ALS (Hu, Koren, Volinsky 2008), in NumPy.

Confidence c_ui = 1 + alpha * r_ui; preference p_ui = 1 if r_ui > 0.
Alternate closed-form ridge solves for users and items. The baseline the
plan keeps behind LightGCN: same interface, trains in well under a second here.
"""
from __future__ import annotations

import numpy as np


class ALS:
    def __init__(self, n_users: int, n_items: int, dim: int = 32, reg: float = 0.1, alpha: float = 20.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.U = rng.normal(0, 0.1, (n_users, dim))
        self.V = rng.normal(0, 0.1, (n_items, dim))
        self.reg, self.alpha = reg, alpha
        self.loss_curve: list[float] = []

    def _loss(self, R):
        C = 1 + self.alpha * R
        err = (R > 0).astype(float) - self.U @ self.V.T
        return float((C * err**2).sum() / R.size + self.reg * ((self.U**2).sum() + (self.V**2).sum()) / R.size)

    @staticmethod
    def _solve(X, R, alpha, reg):
        """Rows of the other side fixed in X; solve each row of the target factor."""
        d = X.shape[1]
        XtX = X.T @ X
        out = np.empty((R.shape[0], d))
        for u in range(R.shape[0]):
            c = 1 + alpha * R[u]
            nz = R[u] > 0
            A = XtX + (X[nz].T * (c[nz] - 1)) @ X[nz] + reg * np.eye(d)
            b = (X[nz].T * c[nz]).sum(1)
            out[u] = np.linalg.solve(A, b)
        return out

    def fit(self, pairs: np.ndarray, iters: int = 12, callback=None) -> "ALS":
        R = np.zeros((len(self.U), len(self.V)))
        R[pairs[:, 0], pairs[:, 1]] = 1
        for it in range(1, iters + 1):
            self.U = self._solve(self.V, R, self.alpha, self.reg)
            self.V = self._solve(self.U, R.T, self.alpha, self.reg)
            self.loss_curve.append(self._loss(R))
            if callback:
                callback(it, self.U, self.V)
        return self
