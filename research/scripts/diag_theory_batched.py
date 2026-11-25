"""Batched P2a check (docs/theory-h4.md, P2a-batched) against results/explore_h4_seg.md.

For each seed, replay segment-level Thompson sampling (policy "thompsonseg", eta = 0, Gaussian prior) with the
hidden segment withheld, i.e. along the "not yet contacted" counterfactual path, and compute every round:

  p_exact   P(some hidden account enters the top-B batch), Monte Carlo over the segment draws z ~ N(0, I_K),
            using the actual posterior means, segment-level sds, remaining accounts and the top-B rule.
  p_prod    the batched formula: p = ∫ φ(z) Π_{j ∈ A_t} Φ((μ_h + s_h z − μ_j)/s_j) dz, A_t = competitor segments
            with >= B remaining accounts (the hidden segment must out-draw every such segment), plus the
            exact handling of partially depleted segments in p_exact.
  p_single  P2a as first stated: Φ(−z), z = (μ_best − μ_h)/sqrt(2) ω (single selection vs the best competitor,
            prior sd ω only).

Hazard → P(first contact within 40 rounds) = 1 − Π(1 − p_t) and median first-contact round, vs observed.
The replay withholds the hidden segment, so the path is the counterfactual "not yet contacted" path; it uses
the same per-seed RNG stream as explore_h4_seg.py for the market and founder (draws differ, which is fine:
p_t is a probability, not a realization).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import integrate, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fprior.cprlp import CPRConfig  # noqa: E402
from fprior.explore import LoopConfig, pessimistic_founder, posterior  # noqa: E402
from fprior.sim import MarketConfig, make_market  # noqa: E402

ROUNDS, B, MC = 40, 25, 4000


def round_probs(mean, segsd, seg, remaining, hidden, K, rng):
    """Return p_exact, p_prod, p_single for one round."""
    rem = np.flatnonzero(remaining)
    # each segment's best remaining account (accounts within a segment are ranked by mean)
    top_mean = np.full(K, -np.inf)
    sd_k = np.zeros(K)
    cnt = np.zeros(K, dtype=int)
    order = rem[np.argsort(-mean[rem])]
    for v in order:
        k = seg[v]
        if cnt[k] == 0:
            top_mean[k], sd_k[k] = mean[v], segsd[v]
        cnt[k] += 1
    hid_acc = rem[seg[rem] == hidden]
    other = rem[seg[rem] != hidden]
    # exact MC: hidden contacted iff its best account's score beats the B-th best other score
    z = rng.standard_normal((MC, K))
    hid_best = (mean[hid_acc][None, :] + segsd[hid_acc][None, :] * z[:, [hidden]]).max(1)
    sc = mean[other][None, :] + segsd[other][None, :] * z[:, seg[other]]
    thr = np.partition(sc, -B, axis=1)[:, -B]
    p_exact = float(np.mean(hid_best > thr))
    # batched product formula over competitors with >= B remaining accounts
    A = [k for k in range(K) if k != hidden and cnt[k] >= B]
    mh, sh = top_mean[hidden], sd_k[hidden]
    f = lambda x: stats.norm.pdf(x) * np.prod([stats.norm.cdf((mh + sh * x - top_mean[j]) / sd_k[j]) for j in A])  # noqa: E731
    p_prod = integrate.quad(f, -10, 10, limit=200)[0]
    # single-selection P2a with prior sd omega only, vs the best available competitor
    best = max(top_mean[k] for k in range(K) if k != hidden and cnt[k] > 0)
    p_single = float(stats.norm.sf((best - mh) / (np.sqrt(2) * 0.5)))
    n_above = sum(1 for k in range(K) if k != hidden and cnt[k] > 0 and top_mean[k] > mh)
    acc_above = sum(cnt[k] for k in range(K) if k != hidden and top_mean[k] > mh)
    return p_exact, p_prod, p_single, sh, n_above, acc_above


def main():
    cpr = CPRConfig()
    obs = json.load(open(ROOT / "results/explore_h4_seg.json"))
    rows = []
    for seed in range(10):
        market = make_market(MarketConfig(), np.random.default_rng(seed))
        founder = pessimistic_founder(market)
        hidden = int(np.argmax(founder.segment_logit - market.segment_logit < -1.0))
        seg, K = market.segment, market.n_segments
        rng = np.random.default_rng([seed, 99])
        mc_rng = np.random.default_rng([seed, 1234])
        remaining = np.ones(market.n, dtype=bool)
        contacted, outcome = [], []
        per = []
        for t in range(ROUNDS):
            c = np.array(contacted, dtype=int)
            o = np.array(outcome, dtype=float)
            mean, var, _ = posterior(market, founder.segment_logit, c, o, cpr, LoopConfig().omega)
            segvar = np.maximum(var - cpr.nugget_sd ** 2 - cpr.graph_sd ** 2 * np.diag(market.graph_cov), 1e-6)
            segsd = np.sqrt(segvar)
            pe, pp, ps, sh, nab, aab = round_probs(mean, segsd, seg, remaining, hidden, K, mc_rng)
            per.append(dict(t=t, p_exact=pe, p_prod=pp, p_single=ps, s_h=sh, seg_above=nab, acc_above=aab))
            # counterfactual step: thompsonseg with the hidden segment withheld
            idx = np.flatnonzero(remaining & (seg != hidden))
            zz = rng.standard_normal(K)
            score = mean[idx] + segsd[idx] * zz[seg[idx]]
            picks = idx[np.argsort(-score)[:B]]
            for v in picks:
                remaining[v] = False
                contacted.append(v)
                outcome.append(market.y[v])
        pe = np.array([r["p_exact"] for r in per]); pp = np.array([r["p_prod"] for r in per])
        ps = np.array([r["p_single"] for r in per])
        surv = np.cumprod(1 - pe)
        med = next((i + 1 for i, s in enumerate(surv) if s <= 0.5), None)
        o = [r for r in obs if r["seed"] == seed and r["policy"] == "thompsonseg" and r["eta"] == 0][0]
        first = next((l["round"] for l in o["log"] if l["hidden_contacts"] > 0), None)
        rows.append(dict(seed=seed, P_exact=1 - surv[-1], P_prod=1 - np.prod(1 - pp), P_single=1 - np.prod(1 - ps),
                         median_pred=med, first_obs=first, s_h0=per[0]["s_h"], acc_above0=per[0]["acc_above"],
                         seg_above0=per[0]["seg_above"], p_by_decile=[float(pe[i:i + 10].mean()) for i in range(0, 40, 10)]))
        print(f"seed {seed}: P_exact {rows[-1]['P_exact']:.2f} P_prod {rows[-1]['P_prod']:.2f} "
              f"P_single {rows[-1]['P_single']:.2e} first_obs {first} s_h0 {per[0]['s_h']:.2f} "
              f"acc_above0 {per[0]['acc_above']} p by 10-round block {np.round(rows[-1]['p_by_decile'], 3)}", flush=True)
    (ROOT / "results/diag_theory_batched.json").write_text(json.dumps(rows, default=float))


if __name__ == "__main__":
    main()
