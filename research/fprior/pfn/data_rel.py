"""Batched relational prior for PC-RFM v1 (mirrors fprior.sim_rel with every knob randomized).

v1 = v0 tokens + precomputed relational features per account (PLAN.md section 3, stage v1),
all computed from public tables only (adoptions, seller-product similarity) plus the
context labels:
  0 sim_mean   mean centred similarity-to-seller of the products the account uses
  1 sim_max    max similarity-to-seller among them
  2 log_deg    log(1 + number of products used)
  3 ra_pos     resource allocation through products to known customers
               sum_p A_vp * (# context positives using p) / deg(p)
  4 ra_neg     same, to contacted non-buyers
  5 any_pos    1 if the account shares any product with a known customer
Continuous features are z-scored within each market, so train and eval match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .data import (N_FEAT, N_IND, N_SIZE, SIZE_P, Batch, PriorConfig, _solve_intercept, _u,
                   founder_and_outreach, make_tokens)

N_REL = 6
N_FEAT_REL = N_FEAT + N_REL


@dataclass(frozen=True)
class RelPriorConfig(PriorConfig):
    n_products: int = 200
    n_categories: int = 20
    dim: int = 8
    new_dims: int = 2
    novelty_range: tuple[float, float] = (0.0, 1.0)
    industry_taste_range: tuple[float, float] = (0.2, 0.8)
    taste_strength_range: tuple[float, float] = (2.0, 7.0)
    adoptions_range: tuple[float, float] = (4.0, 20.0)
    sim_noise_range: tuple[float, float] = (0.05, 0.4)


def _z(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp_min(1e-6)


def rel_features(adoption: torch.Tensor, sim: torch.Tensor, context: torch.Tensor,
                 outcome: torch.Tensor) -> torch.Tensor:
    """adoption (B, N, P) 0/1; sim (B, P); context (B, N) bool; outcome (B, N) -> (B, N, N_REL)."""
    deg = adoption.sum(2)                                            # (B, N)
    simc = sim - sim.mean(1, keepdim=True)
    sim_mean = (adoption * simc[:, None, :]).sum(2) / deg.clamp_min(1)
    sim_max = torch.where(adoption > 0, sim[:, None, :], torch.full_like(adoption, -3.0)).amax(2)
    sim_max = torch.where(deg > 0, sim_max, sim.min(1, keepdim=True).values)
    pdeg = adoption.sum(1).clamp_min(1)                              # (B, P)
    pos = (context & (outcome > 0.5)).float()
    neg = (context & (outcome < 0.5)).float()
    pos_p = torch.einsum("bn,bnp->bp", pos, adoption)
    neg_p = torch.einsum("bn,bnp->bp", neg, adoption)
    ra_pos = torch.einsum("bnp,bp->bn", adoption, pos_p / pdeg)
    ra_neg = torch.einsum("bnp,bp->bn", adoption, neg_p / pdeg)
    any_pos = (torch.einsum("bnp,bp->bn", adoption, pos_p) > 0).float()
    return torch.stack([_z(sim_mean), _z(sim_max), _z(torch.log1p(deg)),
                        torch.log1p(ra_pos), torch.log1p(ra_neg), any_pos], -1)


def _unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)


@torch.no_grad()
def sample_rel_batch(cfg: RelPriorConfig, B: int, g: torch.Generator, dev="cpu") -> Batch:
    N, I, S, K = cfg.n_accounts, N_IND, N_SIZE, N_IND * N_SIZE
    P, C, d, k = cfg.n_products, cfg.n_categories, cfg.dim, cfg.dim - cfg.new_dims
    rn = lambda *shape: torch.randn(*shape, generator=g, device=dev)  # noqa: E731
    industry = torch.randint(0, I, (B, N), generator=g, device=dev)
    size = torch.multinomial(SIZE_P.to(dev).expand(B, -1), N, replacement=True, generator=g)
    segment = industry * S + size

    # tastes, products, adoptions (as in sim_rel.make_rel_market)
    t = _u(*cfg.industry_taste_range, (B,), g, dev)[:, None, None]
    ind_taste = rn(B, I, d)
    taste = t.sqrt() * ind_taste.gather(1, industry[..., None].expand(-1, -1, d)) + (1 - t).sqrt() * rn(B, N, d)
    cat = torch.zeros(B, C, d, device=dev)
    cat[..., :k] = rn(B, C, k)
    category = torch.randint(0, C, (B, P), generator=g, device=dev)
    noise = torch.zeros(B, P, d, device=dev)
    noise[..., :k] = rn(B, P, k)
    u = _unit(0.8 * cat.gather(1, category[..., None].expand(-1, -1, d)) + 0.6 * noise)
    pop = rn(B, P)
    strength = _u(*cfg.taste_strength_range, (B,), g, dev)
    fit = strength[:, None, None] * torch.einsum("bnd,bpd->bnp", taste, u) / math.sqrt(d) + pop[:, None, :]
    target = _u(*cfg.adoptions_range, (B,), g, dev)
    lo, hi = torch.full((B,), -30.0, device=dev), torch.full((B,), 10.0, device=dev)
    for _ in range(40):
        mid = (lo + hi) / 2
        over = torch.sigmoid(mid[:, None, None] + fit).sum(2).mean(1) > target
        hi, lo = torch.where(over, mid, hi), torch.where(over, lo, mid)
    adoption = (torch.rand(B, N, P, generator=g, device=dev) < torch.sigmoid(((lo + hi) / 2)[:, None, None] + fit)).float()

    # seller between an existing anchor product and a brand-new category
    e = _u(*cfg.novelty_range, (B,), g, dev)[:, None]
    anchor = torch.randint(0, P, (B,), generator=g, device=dev)
    fresh = torch.zeros(B, d, device=dev)
    fresh[:, k:] = rn(B, d - k)
    u_s = _unit((1 - e).sqrt() * u[torch.arange(B, device=dev), anchor] + e.sqrt() * _unit(fresh))
    sim_noise = _u(*cfg.sim_noise_range, (B,), g, dev)[:, None]
    sim = torch.einsum("bpd,bd->bp", u, u_s) + sim_noise * rn(B, P)
    rel = _z(torch.einsum("bnd,bd->bn", taste, u_s))

    # segment effects + variance shares (Dirichlet(1,1,1) over segment / relational / latent)
    eff = (0.7 * rn(B, I, 1) + 0.5 * rn(B, 1, S) + 0.5 * rn(B, I, S)).reshape(B, K)
    acc = eff.gather(1, segment)
    seg_std = (eff - acc.mean(1, keepdim=True)) / acc.std(1, keepdim=True)
    w = -torch.log(torch.rand(B, 3, generator=g, device=dev).clamp_min(1e-12))
    w = w / w.sum(1, keepdim=True)
    logit_sd = _u(*cfg.logit_sd_range, (B,), g, dev)
    seg_part = (logit_sd * w[:, 0].sqrt())[:, None] * seg_std
    z = (seg_part.gather(1, segment) + (logit_sd * w[:, 1].sqrt())[:, None] * rel
         + (logit_sd * w[:, 2].sqrt())[:, None] * rn(B, N))
    lb = math.log(cfg.base_rate_range[0]), math.log(cfg.base_rate_range[1])
    base = _u(*lb, (B,), g, dev).exp()
    c = _solve_intercept(z, base)
    p = torch.sigmoid(c[:, None] + z)
    y = (torch.rand(B, N, generator=g, device=dev) < p).float()

    f_acc, has_founder, context, finfo = founder_and_outreach(cfg, seg_part, c, segment, y, g, dev)
    x = torch.cat([make_tokens(industry, size, f_acc, has_founder, context, y),
                   rel_features(adoption, sim, context, y)], -1)
    info = dict(novelty=e.squeeze(1), w_seg=w[:, 0], w_rel=w[:, 1], base=base,
                adoption=adoption, sim=sim, **finfo)
    return Batch(x, context, p, y, info)
