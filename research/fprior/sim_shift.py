"""Shifted worlds for the sim-to-sim stress test (PLAN.md section 5, ladder rung 1).

Each shift changes ONE thing relative to fprior.sim, keeping the Market / Founder / Outreach
interfaces, so CPR-LP and PC-RFM run unchanged. Shifts are derived from a default market
(same accounts and graph) by re-composing its buying logit:

    true_logit = intercept + seg_part[segment] + rest       (rest = graph-smooth + latent part)

S0 default            control
S1 heavy_founder      founder as usual, plus 2-3 segments off by 2.5-4 logits (random sign)
S2 interaction        segment effects are i.i.d. per industry x size cell (no main effects)
S3 nonlinear          logit = c + h(z), h an asymmetric squared threshold of the linear latent z,
                      rescaled to the same sd; the oracle founder knows true segment mean rates
S4 skewed_segments    Zipf-like segment sizes (a few huge segments, many tiny)
S5 rank_outreach      founder contacts strictly by belief rank, plus 5% random walk-ins
S6a base_rate_low     base rate 0.01 (below the PC-RFM training range 0.02-0.2)
S6b base_rate_high    base rate 0.30 (above it)
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit

from .sim import Founder, Market, Outreach, OutreachConfig, run_outreach

SHIFTS = ["S0_default", "S1_heavy_founder", "S2_interaction", "S3_nonlinear", "S4_skewed_segments",
          "S5_rank_outreach", "S6a_base_rate_low", "S6b_base_rate_high"]


def _intercept(no_int: np.ndarray, base: float) -> float:
    return brentq(lambda c: expit(c + no_int).mean() - base, -30, 30)


def _finish(m: Market, segment: np.ndarray, no_int: np.ndarray, base: float,
            rng: np.random.Generator, seg_part: np.ndarray | None = None) -> Market:
    """Recompute intercept, p, y and the oracle segment logits for a re-composed logit."""
    c = _intercept(no_int, base)
    true_logit = c + no_int
    p = expit(true_logit)
    y = (rng.random(len(p)) < p).astype(float)
    K = m.n_segments
    if seg_part is not None:
        seg_logit = c + seg_part
    else:  # oracle founder knows each segment's true mean rate
        cnt = np.bincount(segment, minlength=K)
        rate = np.bincount(segment, weights=p, minlength=K) / np.maximum(cnt, 1)
        rate = np.where(cnt > 0, rate, p.mean())
        seg_logit = logit(np.clip(rate, 1e-6, 1 - 1e-6))
    return replace(m, segment=segment, industry=segment // m.cfg.n_sizes, size=segment % m.cfg.n_sizes,
                   intercept=float(c), segment_logit=seg_logit, true_logit=true_logit, p=p, y=y)


def shift_market(m: Market, shift: str, rng: np.random.Generator) -> Market:
    seg_part = m.segment_logit - m.intercept          # (K,) true segment effects
    rest = m.true_logit - m.segment_logit[m.segment]  # graph + latent part per account
    base = m.cfg.base_rate
    if shift in ("S0_default", "S1_heavy_founder", "S5_rank_outreach"):
        return m
    if shift == "S2_interaction":
        acc_sd = seg_part[m.segment].std()
        e = rng.standard_normal(m.n_segments)
        acc = e[m.segment]
        new = (e - acc.mean()) / acc.std() * acc_sd
        return _finish(m, m.segment, new[m.segment] + rest, base, rng, seg_part=new)
    if shift == "S3_nonlinear":
        z = seg_part[m.segment] + rest
        zs = (z - z.mean()) / z.std()
        h = np.where(zs > 0, zs ** 2, -0.3 * np.abs(zs))   # buyers concentrate in the upper tail
        h = (h - h.mean()) / h.std() * z.std()
        return _finish(m, m.segment, h, base, rng)
    if shift == "S4_skewed_segments":
        K = m.n_segments
        w = 1.0 / np.arange(1, K + 1) ** 1.1
        w = w[rng.permutation(K)]
        seg = rng.choice(K, size=m.n, p=w / w.sum())
        return _finish(m, seg, seg_part[seg] + rest, base, rng, seg_part=seg_part)
    if shift == "S6a_base_rate_low":
        return _finish(m, m.segment, seg_part[m.segment] + rest, 0.01, rng, seg_part=seg_part)
    if shift == "S6b_base_rate_high":
        return _finish(m, m.segment, seg_part[m.segment] + rest, 0.30, rng, seg_part=seg_part)
    raise ValueError(shift)


def heavy_tail_founder(market: Market, founder: Founder, rng: np.random.Generator) -> Founder:
    """S1: add 2-3 large errors (2.5-4 logits, random sign) on segments with >= 20 accounts."""
    cnt = np.bincount(market.segment, minlength=market.n_segments)
    eligible = np.flatnonzero(cnt >= 20)
    k = int(rng.integers(2, 4))
    segs = rng.choice(eligible, size=min(k, len(eligible)), replace=False)
    belief = founder.segment_logit.copy()
    belief[segs] += rng.choice([-1.0, 1.0], size=len(segs)) * rng.uniform(2.5, 4.0, size=len(segs))
    return Founder(founder.cfg, belief)


def rank_outreach(market: Market, founder: Founder, cfg: OutreachConfig,
                  rng: np.random.Generator, walk_in: float = 0.05) -> Outreach:
    """S5: contact accounts strictly in order of believed segment rate (random within ties),
    except that each contact is a random walk-in with probability `walk_in`."""
    score = founder.segment_logit[market.segment]
    order = list(np.lexsort((rng.random(market.n), -score)))
    remaining = np.ones(market.n, dtype=bool)
    contacted, outcome, prop = [], [], []
    found, pos = 0, 0
    while found < cfg.n_customers and len(contacted) < cfg.max_contacts and remaining.any():
        if rng.random() < walk_in:
            v = int(rng.choice(np.flatnonzero(remaining)))
            pr = walk_in / remaining.sum()
        else:
            while not remaining[order[pos]]:
                pos += 1
            v = int(order[pos])
            pr = 1.0 - walk_in
        remaining[v] = False
        contacted.append(v)
        outcome.append(market.y[v])
        prop.append(pr)
        found += int(market.y[v])
    return Outreach(np.array(contacted, dtype=int), np.array(outcome, dtype=float),
                    np.array(prop, dtype=float))


def outreach_for(shift: str, market: Market, founder: Founder, cfg: OutreachConfig,
                 rng: np.random.Generator) -> Outreach:
    if shift == "S5_rank_outreach":
        return rank_outreach(market, founder, cfg, rng)
    return run_outreach(market, founder, cfg, rng)
