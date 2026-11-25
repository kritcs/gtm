"""Why does segment-level Thompson's error stay near −1.9 after contact? (theory-h4 P4 applied to explore_h4_seg)

Per seed: hidden contacts n by round 39, first-contact round, observed error, trust slope b — vs the Gaussian and
exact Student-t washout predictions at that n, with each seed's Δ_eff (prior error from the greedy run in
results/explore_h4.json) and σ² = 1/(p(1−p)) + graph² + nugget².
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import integrate, stats
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fprior.cprlp import CPRConfig  # noqa: E402
from fprior.explore import pessimistic_founder  # noqa: E402
from fprior.sim import MarketConfig, make_market  # noqa: E402


def gauss(delta, n, s2, w2=0.25):
    return -delta * s2 / (s2 + n * w2)


def t_exact(delta, n, s2, omega=0.5, nu=3):
    if n == 0:
        return -delta
    s = np.sqrt(s2 / n)
    f = lambda th: stats.t.pdf((th + delta) / omega, nu) * stats.norm.pdf(0.0, th, s)  # noqa: E731
    lo, hi = -delta - 15, 15
    z = integrate.quad(f, lo, hi, points=[-delta, 0], limit=400)[0]
    return integrate.quad(lambda th: th * f(th), lo, hi, points=[-delta, 0], limit=400)[0] / z


cfg = CPRConfig()
seg = json.load(open(ROOT / "results/explore_h4_seg.json"))
base = json.load(open(ROOT / "results/explore_h4.json"))
rob = json.load(open(ROOT / "results/explore_h4_robust.json"))
lines = ["| policy | seed | first contact | n by r39 | observed error | Gaussian at n | Student-t at n | trust b r39 |",
         "|---|---|---|---|---|---|---|---|"]
summ = {}
for pol, src in [("thompsonseg", seg), ("thompsonseg-robust", seg), ("thompson-robust", rob)]:
    acc = []
    for s in range(10):
        r = [x for x in src if x["seed"] == s and x["policy"] == pol and x["eta"] == 0][0]
        market = make_market(MarketConfig(), np.random.default_rng(s))
        founder = pessimistic_founder(market)
        h = int(np.argmax(founder.segment_logit - market.segment_logit < -1.0))
        ph = expit(market.segment_logit[h])
        s2 = 1 / (ph * (1 - ph)) + cfg.graph_sd ** 2 + cfg.nugget_sd ** 2
        g = [x for x in base if x["seed"] == s and x["policy"] == "greedy" and x["eta"] == 0][0]
        d = -g["log"][-2]["hidden_err"]
        n = r["log"][-2]["hidden_contacts"]
        first = next((l["round"] for l in r["log"] if l["hidden_contacts"] > 0), None)
        e = r["log"][-2]["hidden_err"]
        pg, pt = gauss(d, n, s2), t_exact(d, n, s2)
        acc.append((n, e, pg, pt))
        lines.append(f"| {pol} | {s} | {first} | {n} | {e:+.2f} | {pg:+.2f} | {pt:+.2f} | {r['log'][-2]['trust_b']:.2f} |")
    a = np.array(acc)
    summ[pol] = a
    lines.append(f"| **{pol} mean** | | | {a[:, 0].mean():.1f} (median {np.median(a[:, 0]):.0f}) | {a[:, 1].mean():+.2f} | "
                 f"{a[:, 2].mean():+.2f} | {a[:, 3].mean():+.2f} | |")
lines.append("")
for pol, a in summ.items():
    big = a[:, 0] >= 20
    lines.append(f"- {pol}: seeds with ≥ 20 contacts: {big.sum()}/10; their mean observed error "
                 f"{a[big, 1].mean() if big.any() else float('nan'):+.2f}; seeds with < 20 contacts: mean observed "
                 f"{a[~big, 1].mean():+.2f} (Gaussian formula {a[~big, 2].mean():+.2f}, Student-t {a[~big, 3].mean():+.2f}).")
print("\n".join(lines))
(ROOT / "results/diag_theory_postcontact.md").write_text("\n".join(lines) + "\n")
