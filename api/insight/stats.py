"""Statistics that keep the cards honest."""
from __future__ import annotations

import numpy as np


def poisson_binomial_tail(ps: list[float], k: int) -> float:
    """P(X >= k) where X = sum of independent Bernoulli(p_i).

    Each p_i is the feature's base rate in customer i's *matched* comparison
    group, so the independence assumed here is conditional on the matching
    (size, stage, calendar window), not across the raw market.
    """
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1.0
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= 1 - p
    return float(dist[k:].sum())


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """BH q-values (step-up), same order as input."""
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    q = np.empty(m)
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, pvals[i] * m / rank)
        q[i] = prev
    return q.tolist()


def shrunk_lift(k: int, n: int, base: float, prior_strength: float = 4.0) -> float:
    """Lift with a Beta prior centred on the base rate, so 3-of-3 doesn't read as infinite."""
    base = max(base, 1e-4)
    post_mean = (k + prior_strength * base) / (n + prior_strength)
    return post_mean / base


def bootstrap_recall_ci(ranked_hits: np.ndarray, n_pos: int, k: int, n_boot: int = 1000, seed: int = 0):
    """95% CI for Recall@k by resampling the positives."""
    rng = np.random.default_rng(seed)
    hits = np.asarray(ranked_hits, bool)  # one flag per positive: is it in the top k?
    if n_pos == 0:
        return 0.0, 0.0
    samples = [hits[rng.integers(0, n_pos, n_pos)].mean() for _ in range(n_boot)]
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def pct_rank(x: np.ndarray) -> np.ndarray:
    """Rank percentiles in [0, 1]; puts scores on different scales onto one."""
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r / max(len(x) - 1, 1)
