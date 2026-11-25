"""Simulator v0: market, founder and outreach (PLAN.md sections 2a-2c, reduced).

v0 keeps only what the sanity ladder needs:

- Market: accounts with industry x size segments, an account graph from a
  degree-corrected SBM, and a true buying logit
      logit p_v = c + sd * (sqrt(w_seg) s_seg(v) + sqrt(w_graph) r_v + sqrt(w_latent) z_v)
  where r is graph-smooth (covariance (I + gamma L)^-2) and z is unobservable need.
- Founder: per-segment beliefs whose correlation with the true segment effects is
  the reliability rho. rho = 1, overconfidence = 1, bias = 0 is the oracle founder:
  their beliefs equal the true segment logits exactly.
- Outreach: the founder picks whom to contact, driven by their own beliefs
  (exposure alpha) plus a minimum exploration share eta, until n customers are found.
  Uncontacted accounts are unlabeled, not negative.

Not yet in v0: products / adoption bipartite graph, soft membership, temporal signals,
co-founders, funnel stages, calibration to real data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq
from scipy.special import expit


# ---------------------------------------------------------------- market


@dataclass(frozen=True)
class MarketConfig:
    n_accounts: int = 3000
    n_industries: int = 8
    n_sizes: int = 4
    avg_degree: float = 10.0
    p_within: float = 0.8          # share of edges inside a community (homophily knob)
    community_is_industry: float = 0.7  # prob. an account's community equals its industry
    degree_sigma: float = 0.8      # lognormal spread of degree propensities
    smoothness: float = 5.0        # gamma in (I + gamma L)^-1
    logit_sd: float = 1.5          # total sd of the non-intercept logit
    w_segment: float = 0.5         # variance shares (normalized to sum to 1)
    w_graph: float = 0.3
    w_latent: float = 0.2          # "emergence": share no public signal can see
    base_rate: float = 0.08


@dataclass
class Market:
    cfg: MarketConfig
    industry: np.ndarray
    size: np.ndarray
    segment: np.ndarray            # (N,) int in [0, K)
    n_segments: int
    A: sp.csr_matrix               # account adjacency
    L: sp.csr_matrix               # normalized Laplacian
    graph_cov: np.ndarray          # (N, N) covariance of r, unit mean variance
    intercept: float
    segment_logit: np.ndarray      # (K,) true mean logit per segment, incl. intercept
    true_logit: np.ndarray
    p: np.ndarray
    y: np.ndarray                  # realized customers (would buy if contacted)

    @property
    def n(self) -> int:
        return len(self.segment)

    @property
    def degree(self) -> np.ndarray:
        return np.asarray(self.A.sum(axis=1)).ravel()

    def onehot(self) -> np.ndarray:
        Phi = np.zeros((self.n, self.n_segments))
        Phi[np.arange(self.n), self.segment] = 1.0
        return Phi


def _dcsbm(rng, community, n_comm, theta, n_edges, p_within):
    n = len(community)
    w_all = theta / theta.sum()
    members = [np.flatnonzero(community == c) for c in range(n_comm)]
    comm_mass = np.array([theta[m].sum() for m in members])
    n_in = rng.binomial(n_edges, p_within)
    rows, cols = [], []
    # within-community edges
    cs = rng.choice(n_comm, size=n_in, p=comm_mass / comm_mass.sum())
    for c in range(n_comm):
        k = int((cs == c).sum())
        if k == 0 or len(members[c]) < 2:
            continue
        w = theta[members[c]] / theta[members[c]].sum()
        rows.append(rng.choice(members[c], size=k, p=w))
        cols.append(rng.choice(members[c], size=k, p=w))
    # background edges
    k = n_edges - n_in
    rows.append(rng.choice(n, size=k, p=w_all))
    cols.append(rng.choice(n, size=k, p=w_all))
    r, c = np.concatenate(rows), np.concatenate(cols)
    keep = r != c
    A = sp.coo_matrix((np.ones(keep.sum()), (r[keep], c[keep])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(float)
    return A.tocsr()


def normalized_laplacian(A: sp.csr_matrix) -> sp.csr_matrix:
    d = np.asarray(A.sum(axis=1)).ravel()
    inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
    Dm = sp.diags(inv_sqrt)
    return (sp.identity(A.shape[0]) - Dm @ A @ Dm).tocsr()


def make_market(cfg: MarketConfig, rng: np.random.Generator) -> Market:
    N, I, S = cfg.n_accounts, cfg.n_industries, cfg.n_sizes
    industry = rng.integers(0, I, size=N)
    size = rng.choice(S, size=N, p=np.array([0.4, 0.3, 0.2, 0.1])[:S] / np.array([0.4, 0.3, 0.2, 0.1])[:S].sum())
    segment = industry * S + size
    K = I * S

    # graph
    community = np.where(rng.random(N) < cfg.community_is_industry, industry, rng.integers(0, I, size=N))
    theta = rng.lognormal(0.0, cfg.degree_sigma, size=N)
    A = _dcsbm(rng, community, I, theta, int(N * cfg.avg_degree / 2), cfg.p_within)
    L = normalized_laplacian(A)
    G = np.linalg.inv(np.eye(N) + cfg.smoothness * L.toarray())
    C = G @ G
    scale = np.mean(np.diag(C))
    graph_cov = C / scale
    r = (G @ rng.standard_normal(N)) / np.sqrt(scale)

    # segment effects: industry and size main effects plus an interaction
    eff = (0.7 * rng.standard_normal(I)[:, None] + 0.5 * rng.standard_normal(S)[None, :]
           + 0.5 * rng.standard_normal((I, S))).ravel()
    acc_eff = eff[segment]
    mu, sd = acc_eff.mean(), acc_eff.std()
    seg_std = (eff - mu) / sd

    w = np.array([cfg.w_segment, cfg.w_graph, cfg.w_latent], dtype=float)
    w = w / w.sum()
    seg_part = cfg.logit_sd * np.sqrt(w[0]) * seg_std
    rest = cfg.logit_sd * (np.sqrt(w[1]) * r + np.sqrt(w[2]) * rng.standard_normal(N))
    no_int = seg_part[segment] + rest
    c = brentq(lambda c: expit(c + no_int).mean() - cfg.base_rate, -20, 20)
    true_logit = c + no_int
    p = expit(true_logit)
    y = (rng.random(N) < p).astype(float)
    return Market(cfg, industry, size, segment, K, A, L, graph_cov, c, c + seg_part,
                  true_logit, p, y)


# ---------------------------------------------------------------- founder


@dataclass(frozen=True)
class FounderConfig:
    reliability: float = 1.0       # rho: corr(belief, true segment effect), in [-1, 1]
    overconfidence: float = 1.0    # scales the spread of beliefs (>1 = too extreme)
    bias: float = 0.0              # logit shift: optimism (>0) or pessimism (<0)


@dataclass
class Founder:
    cfg: FounderConfig
    segment_logit: np.ndarray      # (K,) stated belief, logit of "P(buy | contacted)"

    @property
    def reports(self) -> np.ndarray:
        return expit(self.segment_logit)


def make_founder(market: Market, cfg: FounderConfig, rng: np.random.Generator) -> Founder:
    """rho = 1, overconfidence = 1, bias = 0 returns the true segment logits (oracle)."""
    true = market.segment_logit - market.intercept
    s = true.std()
    rho = float(np.clip(cfg.reliability, -1.0, 1.0))
    noise = rng.standard_normal(len(true))
    noise = (noise - noise.mean()) / noise.std() * s
    belief = rho * true + np.sqrt(max(0.0, 1 - rho**2)) * noise
    return Founder(cfg, market.intercept + cfg.bias + cfg.overconfidence * belief)


# ---------------------------------------------------------------- outreach


@dataclass(frozen=True)
class OutreachConfig:
    n_customers: int = 10          # stop once this many customers are found
    exposure: float = 2.0          # alpha: how strongly beliefs steer contacts
    explore: float = 0.1           # eta: minimum uniform share of contact probability
    batch: int = 10
    max_contacts: int = 1500


@dataclass
class Outreach:
    contacted: np.ndarray          # account indices, in contact order
    outcome: np.ndarray            # 1 = became a customer, 0 = contacted, did not buy
    propensity: np.ndarray         # selection weight at the time of contact

    @property
    def n_customers(self) -> int:
        return int(self.outcome.sum())


def run_outreach(market: Market, founder: Founder, cfg: OutreachConfig,
                 rng: np.random.Generator) -> Outreach:
    score = founder.segment_logit[market.segment]
    remaining = np.ones(market.n, dtype=bool)
    contacted, outcome, prop = [], [], []
    found = 0
    while found < cfg.n_customers and len(contacted) < cfg.max_contacts and remaining.any():
        idx = np.flatnonzero(remaining)
        z = cfg.exposure * score[idx]
        soft = np.exp(z - z.max())
        soft /= soft.sum()
        pi = (1 - cfg.explore) * soft + cfg.explore / len(idx)
        k = min(cfg.batch, len(idx))
        pick = rng.choice(len(idx), size=k, replace=False, p=pi)
        for j in pick:
            v = idx[j]
            remaining[v] = False
            contacted.append(v)
            outcome.append(market.y[v])
            prop.append(pi[j])
            found += int(market.y[v])
            if found >= cfg.n_customers:
                break
    return Outreach(np.array(contacted, dtype=int), np.array(outcome, dtype=float),
                    np.array(prop, dtype=float))
