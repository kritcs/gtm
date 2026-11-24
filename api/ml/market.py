"""The market model: text embeddings + LightGCN + cold start, built from one graph snapshot.

The founder's product is treated as new: its edges are held out of training,
and its embedding comes only from its description plus whichever customers
the founder names (cold start + fold-in). The backtest uses exactly the same
path with the product's pre-cutoff customers standing in for the founder.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..market.graph import MarketGraph
from .coldstart import apply_ridge, fit_ridge, fold_in
from .lightgcn import LightGCN
from .text import LSA, free_text


@dataclass
class MarketModel:
    graph: MarketGraph
    dim: int = 48
    epochs: int = 160
    seed: int = 0
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        g, w = self.graph, self.graph.world
        self.C, self.T = len(w.companies), len(w.tools)
        self.product = w.product_tool_id

        # text ---------------------------------------------------------------
        post_list = list(g.posts.values())
        free = [free_text(p) for p in post_list]
        self.lsa = LSA(dim=64).fit([f for f in free if f] + [t.description for t in w.tools]
                                   + [c.description for c in w.companies])
        self.post_ids = np.array([p.id for p in post_list])
        # pain vectors: only what people wrote; full vectors: everything (for company similarity)
        self.post_vecs = self.lsa.transform_hybrid(free)
        self.post_vecs[[not f for f in free]] = 0
        full = self.lsa.transform([p.text for p in post_list])
        self.post_row = {pid: i for i, pid in enumerate(self.post_ids)}
        cv = np.zeros((self.C, 64), np.float32)
        for cid, ps in g.posts_by_company.items():
            cv[cid] = full[[self.post_row[p.id] for p in ps]].mean(0)
        desc = self.lsa.transform([c.description for c in w.companies])
        self.company_text = cv + 0.3 * desc
        self.company_text /= np.linalg.norm(self.company_text, axis=1, keepdims=True) + 1e-9
        self.tool_text = self.lsa.transform([t.description for t in w.tools])

        # graph model --------------------------------------------------------
        pairs = np.array([(c, t) for (c, t) in g.adopted if t != self.product], np.int64)
        self.gcn = LightGCN(self.C, self.T, dim=self.dim, seed=self.seed).fit(pairs, epochs=self.epochs)
        self.n_pairs = len(pairs)
        U = self.gcn.user_emb
        self.company_graph = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-9)

        # description -> item embedding map (fit on every tool but the product)
        others = [t for t in range(self.T) if t != self.product]
        self.ridge = fit_ridge(self.tool_text[others], self.gcn.item_emb[others], alpha=0.5)
        self.info.update(pairs=int(len(pairs)), loss_start=self.gcn.loss_curve[0], loss_end=self.gcn.loss_curve[-1],
                         posts=len(post_list))

    # ------------------------------------------------------------------ product
    def product_vector(self, description: str, customers: list[int], mu: float = 0.5) -> tuple[np.ndarray, dict]:
        x = self.lsa.transform([description])[0]
        e0 = apply_ridge(self.ridge, x)
        e, curve = fold_in(self.gcn.user_emb, e0, customers, mu=mu, seed=self.seed)
        return e, {"e0_norm": float(np.linalg.norm(e0)), "shift": float(np.linalg.norm(e - e0)),
                   "fold_in_steps": len(curve), "customers": len(customers)}

    def market_scores(self, description: str, customers: list[int]) -> tuple[np.ndarray, dict]:
        e, meta = self.product_vector(description, customers)
        return self.gcn.scores(e), meta

    def text_vec(self, text: str) -> np.ndarray:
        """Founder pain vector, in the same hybrid space as post_vecs."""
        return self.lsa.transform_hybrid([text])[0]

    def layout_3d(self) -> np.ndarray:
        """PCA of [graph embedding, text embedding] to 3D, scaled to roughly [-1, 1]: the market map."""
        X = np.hstack([self.company_graph, self.company_text]).astype(np.float64)
        X -= X.mean(0)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        P = X @ Vt[:3].T
        P /= np.abs(P).max(0) + 1e-9
        return P.astype(np.float32)

    def experience_distance(self, E: list[int]) -> np.ndarray:
        """d(c, E): min cosine distance in graph space to the founder's experience set."""
        if not E:
            return np.ones(self.C, np.float32)
        S = self.company_graph @ self.company_graph[E].T
        T = self.company_text @ self.company_text[E].T
        sim = 0.5 * S + 0.5 * T
        return np.clip(1 - sim.max(1), 0, 2)
