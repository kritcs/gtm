"""Batched prior sampler for PC-RFM v0 (PLAN.md section 3, stage v0: no graph).

Each sample is one task: a market of N accounts, a founder of random reliability,
and founder-steered outreach until n customers are found. It mirrors fprior.sim
(same segment structure, founder model and outreach rule), but on the GPU and with
every knob drawn at random per task, so the network has to infer reliability in context.

v0 has no graph: the simulator's graph-smooth part is folded into unobserved latent
need. Outreach uses Plackett-Luce order with fixed weights, which matches
fprior.sim.run_outreach up to how the exploration share is renormalized.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

N_IND, N_SIZE = 8, 4
SIZE_P = torch.tensor([0.4, 0.3, 0.2, 0.1])
# token features: industry one-hot, size one-hot, founder logit (raw, centred), founder flag,
# label state one-hot (unlabeled, contacted & no, contacted & yes)
N_FEAT = N_IND + N_SIZE + 3 + 3


@dataclass(frozen=True)
class PriorConfig:
    n_accounts: int = 1024
    rho_range: tuple[float, float] = (-0.5, 1.0)
    p_oracle: float = 0.1              # share of tasks with an exactly-oracle founder
    overconf_range: tuple[float, float] = (0.5, 2.0)   # log-uniform
    bias_sd: float = 0.5
    n_max: int = 50
    alpha_range: tuple[float, float] = (0.0, 4.0)
    eta_range: tuple[float, float] = (0.02, 0.3)
    w_seg_range: tuple[float, float] = (0.05, 0.8)
    logit_sd_range: tuple[float, float] = (0.8, 2.5)
    base_rate_range: tuple[float, float] = (0.02, 0.2)  # log-uniform
    p_no_founder: float = 0.0          # 1.0 = ablation: founder features always zeroed
    random_seg_structure: bool = False # True: per-task Dirichlet mix of industry / size / interaction effects


@dataclass
class Batch:
    x: torch.Tensor          # (B, N, N_FEAT)
    context: torch.Tensor    # (B, N) bool, contacted accounts (labels visible)
    target_p: torch.Tensor   # (B, N) true buying probability
    y: torch.Tensor          # (B, N) realized outcome
    info: dict               # per-task knobs, for diagnostics


def _u(lo, hi, shape, g, dev):
    return lo + (hi - lo) * torch.rand(shape, generator=g, device=dev)


def _solve_intercept(z: torch.Tensor, base: torch.Tensor, iters: int = 40) -> torch.Tensor:
    lo = torch.full_like(base, -20.0)
    hi = torch.full_like(base, 20.0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        too_high = torch.sigmoid(mid[:, None] + z).mean(1) > base
        hi = torch.where(too_high, mid, hi)
        lo = torch.where(too_high, lo, mid)
    return (lo + hi) / 2


def founder_features(segment_logit_acc: torch.Tensor, has_founder: torch.Tensor) -> torch.Tensor:
    """(B, N) founder logit for each account's segment -> (B, N, 3) features."""
    centred = segment_logit_acc - segment_logit_acc.mean(1, keepdim=True)
    f = torch.stack([segment_logit_acc, centred, torch.ones_like(centred)], -1)
    return f * has_founder[:, None, None]


def make_tokens(industry, size, founder_logit_acc, has_founder, context, outcome) -> torch.Tensor:
    B, N = industry.shape
    dev = industry.device
    ind = torch.nn.functional.one_hot(industry, N_IND).float()
    siz = torch.nn.functional.one_hot(size, N_SIZE).float()
    state = torch.zeros(B, N, 3, device=dev)
    state[..., 0] = (~context).float()
    state[..., 1] = (context & (outcome < 0.5)).float()
    state[..., 2] = (context & (outcome > 0.5)).float()
    return torch.cat([ind, siz, founder_features(founder_logit_acc, has_founder), state], -1)


def segment_effects(cfg, B, I, S, g, dev):
    """Segment effects (B, I*S). Default recipe 0.7 industry + 0.5 size + 0.5 interaction (as in
    sim.make_market); with random_seg_structure the three variance shares are Dirichlet(1,1,1) per
    task, so some markets are pure main effects and others pure interactions (see stress_sim S2)."""
    gi = torch.randn(B, I, 1, generator=g, device=dev)
    gs = torch.randn(B, 1, S, generator=g, device=dev)
    gis = torch.randn(B, I, S, generator=g, device=dev)
    if not cfg.random_seg_structure:
        return (0.7 * gi + 0.5 * gs + 0.5 * gis).reshape(B, I * S)
    w = -torch.log(torch.rand(B, 3, generator=g, device=dev).clamp_min(1e-12))
    w = (w / w.sum(1, keepdim=True)).sqrt()
    return (w[:, 0, None, None] * gi + w[:, 1, None, None] * gs + w[:, 2, None, None] * gis).reshape(B, I * S)


def founder_and_outreach(cfg, seg_part, c, segment, y, g, dev):
    """Founder beliefs (as in sim.make_founder) and founder-steered outreach (as in sim.run_outreach).

    seg_part: (B, K) true segment effects without intercept; c: (B,) intercept.
    Returns founder logit per account, has_founder flag, context mask, and knob info.
    """
    B, K = seg_part.shape
    N = segment.shape[1]
    rho = _u(*cfg.rho_range, (B,), g, dev)
    rho = torch.where(torch.rand(B, generator=g, device=dev) < cfg.p_oracle, torch.ones_like(rho), rho)
    lo, hi = math.log(cfg.overconf_range[0]), math.log(cfg.overconf_range[1])
    kappa = _u(lo, hi, (B,), g, dev).exp()
    bias = cfg.bias_sd * torch.randn(B, generator=g, device=dev)
    oracle = rho >= 1.0
    kappa = torch.where(oracle, torch.ones_like(kappa), kappa)
    bias = torch.where(oracle, torch.zeros_like(bias), bias)
    true = seg_part
    s = true.std(1, keepdim=True)
    e = torch.randn(B, K, generator=g, device=dev)
    e = (e - e.mean(1, keepdim=True)) / e.std(1, keepdim=True) * s
    belief = rho[:, None] * true + (1 - rho ** 2).clamp(min=0).sqrt()[:, None] * e
    f_logit = c[:, None] + bias[:, None] + kappa[:, None] * belief                # (B, K)
    f_acc = f_logit.gather(1, segment)
    has_founder = (torch.rand(B, generator=g, device=dev) >= cfg.p_no_founder).float()

    # outreach: Plackett-Luce order on (1 - eta) softmax(alpha * l) + eta / N, stop at n customers
    alpha = _u(*cfg.alpha_range, (B,), g, dev)
    eta = _u(*cfg.eta_range, (B,), g, dev)
    n = (torch.exp(_u(0.0, math.log(cfg.n_max + 2), (B,), g, dev)) - 1).floor().clamp(0, cfg.n_max)
    w = torch.softmax(alpha[:, None] * f_acc, 1)
    pi = (1 - eta)[:, None] * w + (eta / N)[:, None]
    gumbel = -torch.log(-torch.log(torch.rand(B, N, generator=g, device=dev).clamp_min(1e-12)))
    order = torch.argsort(pi.log() + gumbel, 1, descending=True)
    y_ord = y.gather(1, order)
    found_before = torch.cumsum(y_ord, 1) - y_ord
    take = (found_before < n[:, None]) & (torch.arange(N, device=dev)[None] < N // 2)
    context = torch.zeros(B, N, dtype=torch.bool, device=dev).scatter(1, order, take)

    return f_acc, has_founder, context, dict(rho=rho, n=n, alpha=alpha, eta=eta, has_founder=has_founder)


@torch.no_grad()
def sample_batch(cfg: PriorConfig, B: int, g: torch.Generator, dev="cpu") -> Batch:
    N, I, S, K = cfg.n_accounts, N_IND, N_SIZE, N_IND * N_SIZE
    industry = torch.randint(0, I, (B, N), generator=g, device=dev)
    size = torch.multinomial(SIZE_P.to(dev).expand(B, -1), N, replacement=True, generator=g)
    segment = industry * S + size

    # segment effects: industry + size main effects + interaction (as in sim.make_market)
    eff = segment_effects(cfg, B, I, S, g, dev)
    acc = eff.gather(1, segment)
    seg_std = (eff - acc.mean(1, keepdim=True)) / acc.std(1, keepdim=True)

    logit_sd = _u(*cfg.logit_sd_range, (B,), g, dev)
    w_seg = _u(*cfg.w_seg_range, (B,), g, dev)
    seg_part = (logit_sd * w_seg.sqrt())[:, None] * seg_std                       # (B, K)
    latent = (logit_sd * (1 - w_seg).sqrt())[:, None] * torch.randn(B, N, generator=g, device=dev)
    z = seg_part.gather(1, segment) + latent
    lb = torch.tensor(cfg.base_rate_range, device=dev).log()
    base = _u(lb[0].item(), lb[1].item(), (B,), g, dev).exp()
    c = _solve_intercept(z, base)
    p = torch.sigmoid(c[:, None] + z)
    y = (torch.rand(B, N, generator=g, device=dev) < p).float()

    f_acc, has_founder, context, finfo = founder_and_outreach(cfg, seg_part, c, segment, y, g, dev)

    x = make_tokens(industry, size, f_acc, has_founder, context, y)
    info = dict(w_seg=w_seg, base=base, **finfo)
    return Batch(x, context, p, y, info)
