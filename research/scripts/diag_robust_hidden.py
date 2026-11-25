"""Diagnostic: posterior error on a segment the founder believes 3 logits too low, after k contacts in it
(plus 200 random contacts elsewhere), Gaussian vs Student-t segment deviations. 10 markets."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fprior.cprlp import CPRConfig  # noqa: E402
from fprior.explore import pessimistic_founder, posterior  # noqa: E402
from fprior.sim import MarketConfig, make_market  # noqa: E402

cpr = CPRConfig()
print("k contacts in hidden seg | Gaussian err | t(nu=3) err | t(nu=1) err | t(nu=10) err   (mean over 10 markets; founder starts at -3)")
for k in (0, 5, 10, 30):
    res = {None: [], 3.0: [], 1.0: [], 10.0: []}
    for seed in range(10):
        m = make_market(MarketConfig(), np.random.default_rng(seed))
        f = pessimistic_founder(m)
        hid = int(np.argmax(f.segment_logit - m.segment_logit < -1.0))
        rng = np.random.default_rng([seed, k])
        in_h = np.flatnonzero(m.segment == hid)
        other = np.flatnonzero(m.segment != hid)
        c = np.concatenate([rng.choice(in_h, size=min(k, len(in_h)), replace=False),
                            rng.choice(other, size=200, replace=False)])
        for nu in res:
            mean, _, _ = posterior(m, f.segment_logit, c, m.y[c], cpr, 0.5, nu)
            res[nu].append(mean[m.segment == hid].mean() - m.segment_logit[hid])
    print(f"{k:24d} | " + " | ".join(f"{np.mean(v):+.2f} ± {np.std(v, ddof=1) / np.sqrt(10):.2f}" for v in res.values()))
