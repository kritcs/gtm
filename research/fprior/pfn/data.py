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
    founder_tokens: bool = False       # True: founder as per-segment tokens with stated confidence / "no idea"
    stated_calib_range: tuple[float, float] = (0.3, 3.0)  # log-uniform: stated sd / actual error scale
    p_noidea_max: float = 0.3          # per task, share of segments answered "no idea" ~ U(0, max)
    graph: bool = False                # v0g: account graph with a graph-smooth logit part + neighbour-label features
    avg_degree_range: tuple[float, float] = (4.0, 16.0)
    p_within_range: tuple[float, float] = (0.5, 0.95)
    community_is_industry_range: tuple[float, float] = (0.3, 1.0)
    smoothness_range: tuple[float, float] = (1.0, 10.0)
    w_graph_range: tuple[float, float] = (0.0, 0.8)   # graph share of the non-segment variance


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


N_GFEAT = 4


def graph_features(A: torch.Tensor, context: torch.Tensor, outcome: torch.Tensor) -> torch.Tensor:
    """Per-account neighbour-label features from a dense adjacency A (B, N, N):
    log(1 + degree) / 3, log(1 + #neighbours contacted & bought), log(1 + #contacted & not bought),
    smoothed share bought among contacted neighbours. Only visible labels (context) are used."""
    yes = (context & (outcome > 0.5)).float()
    no = (context & (outcome < 0.5)).float()
    deg = A.sum(-1)
    ny = torch.einsum("bij,bj->bi", A, yes)
    nn_ = torch.einsum("bij,bj->bi", A, no)
    return torch.stack([torch.log1p(deg) / 3, torch.log1p(ny), torch.log1p(nn_), (ny + 0.1) / (ny + nn_ + 1.0)], -1)


def sample_graph(cfg, industry, g, dev):
    """Degree-corrected SBM (as in sim._dcsbm, but Bernoulli per pair) and a graph-smooth unit-sd field
    r = (I + gamma L)^-1 eps, standardized per task. Returns dense A (B, N, N) and r (B, N)."""
    B, N = industry.shape
    q = _u(*cfg.community_is_industry_range, (B, 1), g, dev)
    comm = torch.where(torch.rand(B, N, generator=g, device=dev) < q, industry,
                       torch.randint(0, N_IND, (B, N), generator=g, device=dev))
    theta = torch.exp(_u(0.3, 1.2, (B, 1), g, dev) * torch.randn(B, N, generator=g, device=dev))
    pw = _u(*cfg.p_within_range, (B, 1, 1), g, dev)
    same = (comm[:, :, None] == comm[:, None, :]).float()
    frac_same = same.mean((1, 2), keepdim=True).clamp_min(1e-3)
    W = same * pw / frac_same + (1 - same) * (1 - pw) / (1 - frac_same).clamp_min(1e-3)
    M = theta[:, :, None] * theta[:, None, :] * W
    deg = _u(*cfg.avg_degree_range, (B, 1, 1), g, dev)
    P = (M * deg * N / M.sum((1, 2), keepdim=True)).clamp(max=1.0)
    U = torch.rand(B, N, N, generator=g, device=dev)
    A = torch.triu((U < P).float(), 1)
    A = A + A.transpose(1, 2)
    d = A.sum(-1)
    dm = torch.where(d > 0, d.clamp_min(1e-12).rsqrt(), torch.zeros_like(d))
    L = torch.eye(N, device=dev)[None] - dm[:, :, None] * A * dm[:, None, :]
    gam = _u(*cfg.smoothness_range, (B, 1, 1), g, dev)
    eps = torch.randn(B, N, 1, generator=g, device=dev)
    r = torch.linalg.solve(torch.eye(N, device=dev)[None] + gam * L, eps).squeeze(-1)
    r = (r - r.mean(1, keepdim=True)) / r.std(1, keepdim=True).clamp_min(1e-6)
    return A, r


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
    extra = {}
    if cfg.founder_tokens:
        # stated uncertainty: founder-level calibration x mix of overall and segment-specific error
        err = f_logit - (c[:, None] + true)
        rms = err.pow(2).mean(1, keepdim=True).sqrt().clamp_min(0.05)
        lo_c, hi_c = math.log(cfg.stated_calib_range[0]), math.log(cfg.stated_calib_range[1])
        calib = _u(lo_c, hi_c, (B,), g, dev).exp()[:, None]
        awareness = torch.rand(B, 1, generator=g, device=dev)        # how much they know where they're unsure
        stated_sd = calib * ((1 - awareness) * rms + awareness * err.abs().clamp_min(0.05)) \
            * torch.exp(0.3 * torch.randn(B, K, generator=g, device=dev))
        noidea = torch.rand(B, K, generator=g, device=dev) < cfg.p_noidea_max * torch.rand(B, 1, generator=g, device=dev)
        f_logit = torch.where(noidea, c[:, None] + bias[:, None], f_logit)       # "no idea" -> their overall level
        extra = dict(f_logit=f_logit, stated_sd=stated_sd, noidea=noidea, calib=calib.squeeze(1))
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

    return f_acc, has_founder, context, dict(rho=rho, n=n, alpha=alpha, eta=eta, has_founder=has_founder, **extra)


N_FTOK = N_IND + N_SIZE + 5


def founder_token_features(f_logit, stated_sd, noidea, has_founder):
    """(B, K) per-segment founder statements -> (B, K, N_FTOK) tokens and (B, K) validity mask.
    Features: industry one-hot, size one-hot, belief logit, belief centred over answered segments,
    log stated sd, "no idea" flag, constant."""
    B, K = f_logit.shape
    dev = f_logit.device
    k = torch.arange(K, device=dev)
    ind = torch.nn.functional.one_hot(k // N_SIZE, N_IND).float().expand(B, -1, -1)
    siz = torch.nn.functional.one_hot(k % N_SIZE, N_SIZE).float().expand(B, -1, -1)
    known = (~noidea).float()
    mean_known = (f_logit * known).sum(1, keepdim=True) / known.sum(1, keepdim=True).clamp_min(1)
    belief = torch.where(noidea, torch.zeros_like(f_logit), f_logit)
    centred = torch.where(noidea, torch.zeros_like(f_logit), f_logit - mean_known)
    lsd = torch.where(noidea, torch.zeros_like(f_logit), stated_sd.clamp_min(1e-3).log())
    tok = torch.cat([ind, siz, torch.stack([belief, centred, lsd, noidea.float(), torch.ones_like(belief)], -1)], -1)
    mask = has_founder[:, None].bool().expand(-1, K)
    return tok, mask


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
    eps = torch.randn(B, N, generator=g, device=dev)
    if cfg.graph:   # non-segment variance split into a graph-smooth part and unobservable need
        A, r = sample_graph(cfg, industry, g, dev)
        wg = _u(*cfg.w_graph_range, (B, 1), g, dev)
        eps = wg.sqrt() * r + (1 - wg).sqrt() * eps
    latent = (logit_sd * (1 - w_seg).sqrt())[:, None] * eps
    z = seg_part.gather(1, segment) + latent
    lb = torch.tensor(cfg.base_rate_range, device=dev).log()
    base = _u(lb[0].item(), lb[1].item(), (B,), g, dev).exp()
    c = _solve_intercept(z, base)
    p = torch.sigmoid(c[:, None] + z)
    y = (torch.rand(B, N, generator=g, device=dev) < p).float()

    f_acc, has_founder, context, finfo = founder_and_outreach(cfg, seg_part, c, segment, y, g, dev)

    if cfg.founder_tokens:   # the belief lives only in founder tokens, not on accounts
        x = make_tokens(industry, size, f_acc, torch.zeros_like(has_founder), context, y)
        finfo["f_tok"], finfo["f_mask"] = founder_token_features(finfo["f_logit"], finfo["stated_sd"],
                                                                 finfo["noidea"], has_founder)
    else:
        x = make_tokens(industry, size, f_acc, has_founder, context, y)
    if cfg.graph:
        x = torch.cat([x, graph_features(A, context, y)], -1)
        finfo["w_graph"] = wg.squeeze(1)
    info = dict(w_seg=w_seg, base=base, **finfo)
    return Batch(x, context, p, y, info)
