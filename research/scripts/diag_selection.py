"""Diagnostic (hypothesis 4): with an oracle founder, does CPR-LP over-correct the segments
the founder contacted most? Reports, at rho = 1, how the top-100 composition and the
per-segment posterior shift relate to contact counts and realized hit rates."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fprior.cprlp import CPRConfig, fit_predict  # noqa: E402
from fprior.sim import FounderConfig, MarketConfig, OutreachConfig, make_founder, make_market, run_outreach  # noqa: E402

cfg = replace(CPRConfig(), omega_average=True)
for n in (10, 20):
    rows = []
    for seed in range(10):
        mk = make_market(MarketConfig(), np.random.default_rng(seed))
        rng = np.random.default_rng([seed, 1100, n])
        f = make_founder(mk, FounderConfig(reliability=1.0), rng)
        out = run_outreach(mk, f, OutreachConfig(n_customers=n), rng)
        U = np.setdiff1d(np.arange(mk.n), out.contacted)
        r = fit_predict(mk.segment, mk.n_segments, mk.graph_cov, f.segment_logit, out.contacted, out.outcome, cfg)
        top = U[np.argsort(-r.logit[U])[:100]]
        best = np.argmax(mk.segment_logit)
        seg_c = mk.segment[out.contacted]
        nb = (seg_c == best).sum()
        hit_best = out.outcome[seg_c == best].mean() if nb else np.nan
        # mean posterior shift (vs prior mean ordering) inside the best segment, within-segment spread
        in_best = U[mk.segment[U] == best]
        shift = r.logit[in_best] - f.segment_logit[best]
        rows.append(dict(contacts=len(out.contacted), in_best=nb, true_p_best=float(mk.p[mk.segment == best].mean()),
                         hit_best=hit_best, top_from_best=float((mk.segment[top] == best).mean()),
                         best_size=len(in_best), omega=r.omega, b=r.trust[1],
                         shift_best=float(shift.mean()), sd_within=float(shift.std())))
    print(f"n={n}")
    for k in rows[0]:
        print(f"  {k:14s}", " ".join(f"{row[k]:6.2f}" for row in rows))
