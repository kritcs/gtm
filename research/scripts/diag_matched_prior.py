"""Derive CPR-LP's trust prior implied by PC-RFM v0's training founder population.

Replicates fprior/pfn/data.py (PriorConfig defaults) for segment effects, intercept and founder beliefs,
then for every simulated founder fits, by account-weighted least squares on the 32 segments,
    true_seg_logit_k - lbar = a + b (l_k - lbar),   lbar = account mean of l
exactly CPR-LP's parameterization (fprior/cprlp.py), and records the residual sd (implied omega).
Writes results/matched_prior_population.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fprior.pfn.data import N_IND, N_SIZE, SIZE_P, PriorConfig, _solve_intercept, _u, segment_effects  # noqa: E402


def population(B: int = 20000, N: int = 1024, seed: int = 0, cfg: PriorConfig = PriorConfig()):
    g = torch.Generator().manual_seed(seed)
    dev = "cpu"
    I, S, K = N_IND, N_SIZE, N_IND * N_SIZE
    industry = torch.randint(0, I, (B, N), generator=g)
    size = torch.multinomial(SIZE_P.expand(B, -1), N, replacement=True, generator=g)
    segment = industry * S + size
    eff = segment_effects(cfg, B, I, S, g, dev)
    acc = eff.gather(1, segment)
    seg_std = (eff - acc.mean(1, keepdim=True)) / acc.std(1, keepdim=True)
    logit_sd = _u(*cfg.logit_sd_range, (B,), g, dev)
    w_seg = _u(*cfg.w_seg_range, (B,), g, dev)
    seg_part = (logit_sd * w_seg.sqrt())[:, None] * seg_std
    latent = (logit_sd * (1 - w_seg).sqrt())[:, None] * torch.randn(B, N, generator=g)
    z = seg_part.gather(1, segment) + latent
    lb = math.log(cfg.base_rate_range[0]), math.log(cfg.base_rate_range[1])
    base = _u(*lb, (B,), g, dev).exp()
    c = _solve_intercept(z, base)
    # founder, as in founder_and_outreach
    rho = _u(*cfg.rho_range, (B,), g, dev)
    rho = torch.where(torch.rand(B, generator=g) < cfg.p_oracle, torch.ones_like(rho), rho)
    lo, hi = math.log(cfg.overconf_range[0]), math.log(cfg.overconf_range[1])
    kappa = _u(lo, hi, (B,), g, dev).exp()
    bias = cfg.bias_sd * torch.randn(B, generator=g)
    oracle = rho >= 1.0
    kappa = torch.where(oracle, torch.ones_like(kappa), kappa)
    bias = torch.where(oracle, torch.zeros_like(bias), bias)
    s = seg_part.std(1, keepdim=True)
    e = torch.randn(B, K, generator=g)
    e = (e - e.mean(1, keepdim=True)) / e.std(1, keepdim=True) * s
    belief = rho[:, None] * seg_part + (1 - rho ** 2).clamp(min=0).sqrt()[:, None] * e
    f_logit = c[:, None] + bias[:, None] + kappa[:, None] * belief
    true = c[:, None] + seg_part
    # account-weighted least squares per founder
    cnt = torch.nn.functional.one_hot(segment, K).sum(1).double()          # (B, K)
    wts = cnt / cnt.sum(1, keepdim=True)
    l, t = f_logit.double(), true.double()
    lbar = (wts * l).sum(1, keepdim=True)
    xc = l - lbar
    yc = t - lbar
    xm = (wts * xc).sum(1, keepdim=True)
    ym = (wts * yc).sum(1, keepdim=True)
    var_x = (wts * (xc - xm) ** 2).sum(1)
    cov = (wts * (xc - xm) * (yc - ym)).sum(1)
    b = cov / var_x.clamp_min(1e-12)
    a = ym.squeeze(1) - b * xm.squeeze(1)
    resid = yc - (a[:, None] + b[:, None] * xc)
    rsd = (wts * resid ** 2).sum(1).sqrt()
    return dict(a=a.numpy(), b=b.numpy(), resid_sd=rsd.numpy(), rho=rho.numpy(), oracle=oracle.numpy())


def summary(pop):
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    out = {}
    for k in ("a", "b", "resid_sd"):
        v = pop[k]
        out[k] = dict(mean=float(v.mean()), sd=float(v.std()), quantiles=dict(zip(map(str, q), np.quantile(v, q).round(3).tolist())))
    out["corr_ab"] = float(np.corrcoef(pop["a"], pop["b"])[0, 1])
    o = pop["oracle"]
    out["non_oracle"] = dict(b_mean=float(pop["b"][~o].mean()), b_sd=float(pop["b"][~o].std()),
                             a_mean=float(pop["a"][~o].mean()), a_sd=float(pop["a"][~o].std()),
                             resid_sd_mean=float(pop["resid_sd"][~o].mean()), share=float((~o).mean()))
    out["oracle"] = dict(b_mean=float(pop["b"][o].mean()), b_sd=float(pop["b"][o].std()),
                         a_mean=float(pop["a"][o].mean()), a_sd=float(pop["a"][o].std()),
                         resid_sd_mean=float(pop["resid_sd"][o].mean()), share=float(o.mean()))
    return out


if __name__ == "__main__":
    pop = population()
    s = summary(pop)
    print(json.dumps(s, indent=1))
    (ROOT / "results/matched_prior_population.json").write_text(json.dumps(s, indent=1))
