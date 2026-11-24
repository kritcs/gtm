"""Ranking: market plus founder, weighted by distance from the founder's experience.

    s(c) = pct(y_hat(c)) + lambda(c) * pct(s_founder(c)),   lambda(c) = lambda0 * exp(-d(c, E) / tau)

Both terms are converted to rank percentiles first: y_hat is a BPR inner
product and s_founder is a cosine, so adding them raw would be meaningless.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..market.simulate import SIZE_ORDER
from ..ml.market import MarketModel
from .stats import pct_rank


@dataclass
class Constraints:
    max_size: str | None = None
    min_size: str | None = None
    exclude_industries: tuple[str, ...] = ()

    def allows(self, company) -> bool:
        si = SIZE_ORDER.index(company.size)
        if self.max_size and si > SIZE_ORDER.index(self.max_size):
            return False
        if self.min_size and si < SIZE_ORDER.index(self.min_size):
            return False
        return company.industry not in self.exclude_industries


@dataclass
class Ranking:
    score: np.ndarray
    y_hat: np.ndarray
    s_founder: np.ndarray
    lam: np.ndarray
    dist: np.ndarray
    pain: np.ndarray
    pain_post: np.ndarray
    candidates: np.ndarray  # bool mask
    meta: dict


def pain_match(model: MarketModel, founder_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pain(c) = max over c's posts of cos(v_founder, v_post); also returns the argmax post id."""
    sims = model.post_vecs @ founder_vec
    best = np.full(model.C, -1.0, np.float32)
    best_post = np.full(model.C, -1, np.int64)
    for cid, ps in model.graph.posts_by_company.items():
        rows = [model.post_row[p.id] for p in ps]
        j = int(np.argmax(sims[rows]))
        best[cid], best_post[cid] = sims[rows][j], ps[j].id
    return best, best_post


def rank(model: MarketModel, *, description: str, customers: list[int], experience: list[int],
         founder_text: str, exclude: set[int], constraints: Constraints,
         lambda0: float = 1.0, tau: float = 0.25) -> Ranking:
    w = model.graph.world
    y_hat, meta = model.market_scores(description, customers)
    fvec = model.text_vec(founder_text)
    pain, pain_post = pain_match(model, fvec)
    s_founder = 0.7 * pain + 0.3 * (model.company_text @ model.lsa.transform([founder_text])[0])
    dist = model.experience_distance(experience)
    lam = lambda0 * np.exp(-dist / tau)
    score = pct_rank(y_hat) + lam * pct_rank(s_founder)
    cand = np.array([c.id not in exclude and constraints.allows(c) and c.id in model.graph.posts_by_company
                     for c in w.companies])
    meta |= {"lambda0": lambda0, "tau": tau, "candidates": int(cand.sum())}
    return Ranking(score, y_hat, s_founder, lam, dist, pain, pain_post, cand, meta)
