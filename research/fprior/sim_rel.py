"""Simulator v1: a relational market (PLAN.md 2a; RDL direction, see PLAN.md section 3).

Tables
- accounts(industry, size)                       -> segment = industry x size
- products(category, popularity)                 -> ~200 existing products
- adoptions(account, product)                    -> who already uses what
- seller_similarity(product, score)              -> public, noisy similarity of the
                                                    seller's new product to each product
Hidden
- account taste a_v in R^d (partly set by industry), product embedding u_p, seller u_s.

Buying logit for account v:
    c + sd * ( sqrt(w_seg) seg(v) + sqrt(w_rel) std(a_v . u_s) + sqrt(w_lat) z_v )

Novelty knob e in [0, 1]: u_s = sqrt(1-e) u_anchor + sqrt(e) u_fresh, where u_anchor is an
existing product and u_fresh is a product in a brand-new category. e = 0: the seller is a
near-clone of an existing product, so that product's adopters reveal who will buy. e = 1:
a new category; adopters of existing products say little ("similar-product path is empty").

Founders and outreach from fprior.sim work unchanged (same segment / segment_logit fields).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq
from scipy.special import expit

from .sim import normalized_laplacian


@dataclass(frozen=True)
class RelMarketConfig:
    n_accounts: int = 3000
    n_industries: int = 8
    n_sizes: int = 4
    n_products: int = 200
    n_categories: int = 20
    dim: int = 8
    adoptions_per_account: float = 12.0
    new_dims: int = 2               # taste dims no existing product covers (a new category lives here)
    industry_taste: float = 0.5     # share of taste variance set by industry
    taste_strength: float = 5.0     # how much taste drives adoption vs popularity
    novelty: float = 0.3            # e: 0 = clone of an existing product, 1 = new category
    sim_noise: float = 0.2          # noise on the public seller-product similarity
    logit_sd: float = 1.5
    w_segment: float = 0.35
    w_relational: float = 0.45
    w_latent: float = 0.2
    base_rate: float = 0.08
    graph_smoothness: float = 5.0   # for the co-adoption GMRF covariance (CPR-LP baseline)


@dataclass
class RelMarket:
    cfg: RelMarketConfig
    industry: np.ndarray
    size: np.ndarray
    segment: np.ndarray
    n_segments: int
    product_category: np.ndarray    # (P,)
    adoption: sp.csr_matrix         # (N, P) 0/1
    seller_similarity: np.ndarray   # (P,) observed
    anchor: int                     # the existing product the seller resembles (hidden)
    A: sp.csr_matrix                # account co-adoption graph
    graph_cov: np.ndarray           # GMRF covariance on A, unit mean variance
    intercept: float
    segment_logit: np.ndarray       # (K,) true mean logit per segment (what an oracle founder knows)
    relational: np.ndarray          # (N,) standardized a_v . u_s (hidden)
    true_logit: np.ndarray
    p: np.ndarray
    y: np.ndarray

    @property
    def n(self) -> int:
        return len(self.segment)

    @property
    def degree(self) -> np.ndarray:
        return np.asarray(self.adoption.sum(axis=1)).ravel()


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def make_rel_market(cfg: RelMarketConfig, rng: np.random.Generator) -> RelMarket:
    N, I, S, P, C, d = (cfg.n_accounts, cfg.n_industries, cfg.n_sizes, cfg.n_products,
                        cfg.n_categories, cfg.dim)
    industry = rng.integers(0, I, size=N)
    sp_ = np.array([0.4, 0.3, 0.2, 0.1])[:S]
    size = rng.choice(S, size=N, p=sp_ / sp_.sum())
    segment = industry * S + size
    K = I * S

    # hidden tastes and product embeddings
    t = cfg.industry_taste
    taste = np.sqrt(t) * rng.standard_normal((I, d))[industry] + np.sqrt(1 - t) * rng.standard_normal((N, d))
    k = d - cfg.new_dims             # existing products only span the first k taste dims
    cat_center = np.zeros((C, d))
    cat_center[:, :k] = rng.standard_normal((C, k))
    category = rng.integers(0, C, size=P)
    noise = np.zeros((P, d))
    noise[:, :k] = rng.standard_normal((P, k))
    u = _unit(0.8 * cat_center[category] + 0.6 * noise)
    pop = rng.standard_normal(P)

    # adoptions: logit = b0 + pop_p + strength * a_v . u_p / sqrt(d)
    fit = cfg.taste_strength * (taste @ u.T) / np.sqrt(d) + pop[None, :]
    b0 = brentq(lambda b: expit(b + fit).sum(1).mean() - cfg.adoptions_per_account, -30, 10)
    adoption = sp.csr_matrix((rng.random((N, P)) < expit(b0 + fit)).astype(float))

    # seller: mix of an existing anchor product and a product from a new category
    anchor = int(rng.integers(0, P))
    fresh = np.zeros(d)                # a brand-new category: a need no existing product serves
    fresh[k:] = rng.standard_normal(cfg.new_dims)
    fresh = _unit(fresh)
    e = float(np.clip(cfg.novelty, 0.0, 1.0))
    u_s = _unit(np.sqrt(1 - e) * u[anchor] + np.sqrt(e) * fresh)
    seller_similarity = u @ u_s + cfg.sim_noise * rng.standard_normal(P)

    rel = taste @ u_s
    rel = (rel - rel.mean()) / rel.std()

    # segment effects as in sim.make_market
    eff = (0.7 * rng.standard_normal(I)[:, None] + 0.5 * rng.standard_normal(S)[None, :]
           + 0.5 * rng.standard_normal((I, S))).ravel()
    acc_eff = eff[segment]
    seg_std = (eff - acc_eff.mean()) / acc_eff.std()

    w = np.array([cfg.w_segment, cfg.w_relational, cfg.w_latent], dtype=float)
    w = w / w.sum()
    seg_part = cfg.logit_sd * np.sqrt(w[0]) * seg_std
    no_int = (seg_part[segment] + cfg.logit_sd * (np.sqrt(w[1]) * rel
              + np.sqrt(w[2]) * rng.standard_normal(N)))
    c = brentq(lambda c: expit(c + no_int).mean() - cfg.base_rate, -20, 20)
    true_logit = c + no_int
    p = expit(true_logit)
    y = (rng.random(N) < p).astype(float)

    # co-adoption account graph (for graph baselines): weight = shared products, top-10 neighbours
    co = (adoption @ adoption.T).tocsr()
    co.setdiag(0)
    co.eliminate_zeros()
    rows, cols, vals = [], [], []
    for v in range(N):
        s, e_ = co.indptr[v], co.indptr[v + 1]
        if e_ > s:
            idx = co.indices[s:e_]
            val = co.data[s:e_]
            top = np.argsort(-val)[:10]
            rows += [v] * len(top)
            cols += idx[top].tolist()
            vals += val[top].tolist()
    A = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    A = A.maximum(A.T).tocsr()
    L = normalized_laplacian(A)
    G = np.linalg.inv(np.eye(N) + cfg.graph_smoothness * L.toarray())
    Cg = G @ G
    graph_cov = Cg / np.mean(np.diag(Cg))

    return RelMarket(cfg, industry, size, segment, K, category, adoption, seller_similarity, anchor,
                     A, graph_cov, c, c + seg_part, rel, true_logit, p, y)


def similar_adopter_score(market: RelMarket) -> np.ndarray:
    """The 'similar-product adopters' heuristic: mean public similarity to the seller of the
    products each account already uses, centred over products (accounts with none score 0)."""
    w = market.seller_similarity - market.seller_similarity.mean()
    deg = np.maximum(market.degree, 1.0)
    return np.asarray(market.adoption @ w).ravel() / deg
