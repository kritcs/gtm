"""LightGCN with BPR loss, in NumPy.

LightGCN has no weights or nonlinearities in its propagation, so the final
embeddings are a *linear* function of the layer-0 embeddings:

    E = M E0,   M = sum_{k=0..K} a_k * A_hat^k,   A_hat = D^-1/2 A D^-1/2

That makes the backward pass exact and cheap: dL/dE0 = M^T dL/dE, and since
A_hat is symmetric, M^T = M. With ~1k nodes, M is a dense 1k x 1k matrix and
a full training run takes a few seconds on a laptop CPU, with no torch needed.

Users are companies and items are tools. Training pairs are reviewed
(company, ADOPTED, tool) triples from the as-of snapshot.
"""
from __future__ import annotations

import numpy as np


class LightGCN:
    def __init__(self, n_users: int, n_items: int, dim: int = 48, layers: int = 3,
                 lr: float = 0.03, reg: float = 1e-4, seed: int = 0):
        self.U, self.I, self.d, self.K = n_users, n_items, dim, layers
        self.lr, self.reg = lr, reg
        self.rng = np.random.default_rng(seed)
        self.E0 = (self.rng.normal(0, 0.1, (n_users + n_items, dim))).astype(np.float32)
        self.loss_curve: list[float] = []

    def _propagation(self, pairs: np.ndarray) -> np.ndarray:
        N = self.U + self.I
        A = np.zeros((N, N), np.float32)
        A[pairs[:, 0], self.U + pairs[:, 1]] = 1
        A[self.U + pairs[:, 1], pairs[:, 0]] = 1
        deg = A.sum(1)
        dinv = np.where(deg > 0, 1 / np.sqrt(np.maximum(deg, 1e-9)), 0).astype(np.float32)
        Ahat = dinv[:, None] * A * dinv[None, :]
        M = np.eye(N, dtype=np.float32)
        P = np.eye(N, dtype=np.float32)
        for _ in range(self.K):
            P = P @ Ahat
            M += P
        return M / (self.K + 1)

    def fit(self, pairs: np.ndarray, epochs: int = 160, batch: int = 4096) -> "LightGCN":
        pairs = np.asarray(pairs, np.int64)
        self.M = self._propagation(pairs)
        pos = {(int(u), int(i)) for u, i in pairs}
        m = np.zeros_like(self.E0)
        v = np.zeros_like(self.E0)
        b1, b2 = 0.9, 0.999
        for ep in range(1, epochs + 1):
            idx = self.rng.integers(0, len(pairs), batch)
            u, ip = pairs[idx, 0], pairs[idx, 1]
            ineg = self.rng.integers(0, self.I, batch)
            clash = np.array([(int(a), int(b)) in pos for a, b in zip(u, ineg)])
            ineg[clash] = self.rng.integers(0, self.I, clash.sum())
            E = self.M @ self.E0
            eu, ep_, en = E[u], E[self.U + ip], E[self.U + ineg]
            x = (eu * (ep_ - en)).sum(1)
            self.loss_curve.append(float(np.mean(np.logaddexp(0, -x))))
            g = (-1 / (1 + np.exp(x)))[:, None].astype(np.float32) / batch
            G = np.zeros_like(E)
            np.add.at(G, u, g * (ep_ - en))
            np.add.at(G, self.U + ip, g * eu)
            np.add.at(G, self.U + ineg, -g * eu)
            grad = self.M @ G + self.reg * self.E0
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad * grad
            self.E0 -= self.lr * (m / (1 - b1**ep)) / (np.sqrt(v / (1 - b2**ep)) + 1e-8)
        E = self.M @ self.E0
        self.user_emb, self.item_emb = E[: self.U], E[self.U:]
        return self

    def scores(self, item_vec: np.ndarray) -> np.ndarray:
        """y_hat(u, item) for every user, given an item embedding in final space."""
        return self.user_emb @ item_vec
