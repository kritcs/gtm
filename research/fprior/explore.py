"""Sequential outreach with CPR-LP: greedy vs Thompson sampling vs stratified exploration (H4).

Tests the identifiability claim (research-map.md, "Math: bandits"): when the founder's prior steers
outreach, a confidently wrong belief about a segment the founder avoids is never corrected without
exploration; with a minimum exploration rate eta it is, and the prior "washes out" at a finite time.

Posterior: the CPR-LP model (cprlp.py) with omega fixed (no grid), so per-account Laplace predictive
means and variances are available for Thompson draws. Draws are independent per account (marginal
Thompson sampling): it ignores posterior correlation between accounts, which is a simplification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit

from .cprlp import CPRConfig, laplace_bernoulli_evidence
from .sim import Founder, FounderConfig, Market


def posterior(market: Market, founder_segment_logit: np.ndarray, contacted: np.ndarray,
              outcome: np.ndarray, cfg: CPRConfig, omega: float):
    """Per-account Laplace posterior mean and variance of the logit, and posterior mean trust (a, b)."""
    seg = market.segment
    N = market.n
    ell = founder_segment_logit[seg]
    lbar = ell.mean()
    H = np.column_stack([np.ones(N), ell - lbar])
    mu = np.asarray(cfg.trust_mean, dtype=float)
    S = np.diag(np.square(cfg.trust_sd))
    m = lbar + H @ mu
    # prior marginal variance of every account
    prior_var = (omega ** 2 + cfg.graph_sd ** 2 * np.diag(market.graph_cov)
                 + np.einsum("ij,jk,ik->i", H, S, H) + cfg.nugget_sd ** 2)
    if len(contacted) == 0:
        return m, prior_var, mu
    c = contacted
    K_all_c = (omega ** 2 * (seg[:, None] == seg[c][None, :]) + cfg.graph_sd ** 2 * market.graph_cov[:, c]
               + H @ S @ H[c].T)
    K_all_c[c, np.arange(len(c))] += cfg.nugget_sd ** 2
    K_cc = K_all_c[c]
    grad, _ = laplace_bernoulli_evidence(K_cc, m[c], outcome)
    mean = m + K_all_c @ grad
    f_c = mean[c]
    W = expit(f_c) * (1 - expit(f_c))
    # Var = k_vv - k_v^T (K_cc + W^-1)^-1 k_v
    fac = cho_factor(K_cc + np.diag(1.0 / np.maximum(W, 1e-9)))
    var = prior_var - np.einsum("ij,ji->i", K_all_c, cho_solve(fac, K_all_c.T))
    trust = mu + S @ H[c].T @ grad
    return mean, np.maximum(var, 1e-9), trust


def pessimistic_founder(market: Market, n_wrong: int = 1, shift: float = -3.0) -> Founder:
    """Oracle founder except the truly best `n_wrong` segments, believed `shift` logits worse."""
    belief = market.segment_logit.copy()
    sizes = np.bincount(market.segment, minlength=market.n_segments)
    order = np.argsort(-np.where(sizes >= 20, market.segment_logit, -np.inf))
    belief[order[:n_wrong]] += shift
    return Founder(FounderConfig(reliability=1.0), belief)


@dataclass(frozen=True)
class LoopConfig:
    rounds: int = 40
    batch: int = 25
    omega: float = 0.5


def run_policy(market: Market, founder: Founder, policy: str, eta: float, cfg: LoopConfig,
               rng: np.random.Generator, cpr: CPRConfig = CPRConfig()):
    """policy in {greedy, thompson, founder, oracle}; eta = share of each batch spent on stratified
    exploration (uniform segment, then uniform account within it)."""
    hidden = int(np.argmax(founder.segment_logit - market.segment_logit < -1.0))   # the wrongly believed segment
    remaining = np.ones(market.n, dtype=bool)
    contacted, outcome, log = [], [], []
    for t in range(cfg.rounds):
        c = np.array(contacted, dtype=int)
        o = np.array(outcome, dtype=float)
        if policy in ("greedy", "thompson"):
            mean, var, trust = posterior(market, founder.segment_logit, c, o, cpr, cfg.omega)
        else:
            mean, var, trust = founder.segment_logit[market.segment].astype(float), None, np.array([np.nan, np.nan])
        # how the model currently sees the hidden segment
        seg_means = np.array([mean[market.segment == k].mean() if (market.segment == k).any() else -np.inf
                              for k in range(market.n_segments)])
        rank_hidden = int((seg_means > seg_means[hidden]).sum())
        err_hidden = float(seg_means[hidden] - market.segment_logit[hidden])
        log.append(dict(round=t, contacts=len(c), customers=float(o.sum()) if len(o) else 0.0,
                        hidden_contacts=int((market.segment[c] == hidden).sum()) if len(c) else 0,
                        hidden_rank=rank_hidden, hidden_err=err_hidden, trust_b=float(trust[1])))
        # choose the next batch
        idx = np.flatnonzero(remaining)
        n_explore = rng.binomial(cfg.batch, eta) if eta > 0 else 0
        picks = []
        segs_avail = np.unique(market.segment[idx])
        for _ in range(n_explore):
            k = rng.choice(segs_avail)
            cand = idx[(market.segment[idx] == k) & ~np.isin(idx, picks)]
            if len(cand):
                picks.append(int(rng.choice(cand)))
        left = np.setdiff1d(idx, picks)
        if policy == "oracle":
            score = market.true_logit[left]
        elif policy == "founder":
            score = founder.segment_logit[market.segment[left]] + 1e-6 * rng.random(len(left))
        elif policy == "greedy":
            score = mean[left]
        else:  # thompson, marginal draws
            score = mean[left] + np.sqrt(var[left]) * rng.standard_normal(len(left))
        picks += left[np.argsort(-score)[: cfg.batch - len(picks)]].tolist()
        for v in picks:
            remaining[v] = False
            contacted.append(v)
            outcome.append(market.y[v])
    c = np.array(contacted, dtype=int)
    log.append(dict(round=cfg.rounds, contacts=len(c), customers=float(np.sum(outcome)),
                    hidden_contacts=int((market.segment[c] == hidden).sum()), hidden_rank=-1,
                    hidden_err=float("nan"), trust_b=float("nan")))
    return log, hidden
