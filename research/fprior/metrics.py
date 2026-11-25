"""Ranking metrics on the accounts the founder has not contacted yet.

Both metrics use true buying probabilities rather than one realized draw, which removes
outcome noise from the comparison (only the simulator can do this).
"""

from __future__ import annotations

import numpy as np


def _top_k(scores: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    # random tie-breaking, so constant scores (labels-only at n = 0) score as random
    order = np.lexsort((rng.random(len(scores)), -scores))
    return order[:k]


def lift_at_k(scores: np.ndarray, p_true: np.ndarray, k: int, rng: np.random.Generator) -> float:
    """Expected hit rate of the top k over that of a random k (1.0 = random)."""
    return float(p_true[_top_k(scores, k, rng)].mean() / p_true.mean())


def expected_auc(scores: np.ndarray, p_true: np.ndarray) -> float:
    """E[P(buyer outranks non-buyer)] over i != j, ties counted as 1/2."""
    order = np.argsort(scores, kind="stable")
    s, p = scores[order], p_true[order]
    q = 1 - p
    num, neg_below, i, n = 0.0, 0.0, 0, len(s)
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        pg, qg = p[i:j + 1], q[i:j + 1]
        num += pg.sum() * neg_below + 0.5 * (pg.sum() * qg.sum() - (pg * qg).sum())
        neg_below += qg.sum()
        i = j + 1
    den = p.sum() * q.sum() - (p * q).sum()
    return float(num / den)
