"""Does the Student-t segment prior break the good cases? M0 sanity-ladder markets (fprior.sim),
same founders/outreach RNG as scripts/sanity_ladder.py, 10 seeds. Compares current CPR-LP
(Gaussian, omega averaged over the grid) with CPR-LP robust (same, plus Student-t(nu) EM per grid
value). Writes results/robust_ladder.md/.json.
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fprior.cprlp import CPRConfig, fit_predict  # noqa: E402
from fprior.metrics import lift_at_k  # noqa: E402
from fprior.sim import FounderConfig, MarketConfig, OutreachConfig, make_founder, make_market, run_outreach  # noqa: E402

RHOS = [1.0, 0.5, 0.0, -0.5]
NS = [0, 1, 5, 10, 20, 50]
NU = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0

rows = []
for seed in range(10):
    market = make_market(MarketConfig(), np.random.default_rng(seed))
    for rho in RHOS:
        for n in NS:
            rng = np.random.default_rng([seed, int(round(rho * 100)) + 1000, n])
            founder = make_founder(market, FounderConfig(reliability=rho), rng)
            out = run_outreach(market, founder, OutreachConfig(n_customers=n), rng)
            U = np.setdiff1d(np.arange(market.n), out.contacted)
            args = (market.segment, market.n_segments, market.graph_cov, founder.segment_logit,
                    out.contacted, out.outcome)
            sc = {"prior_only": founder.segment_logit[market.segment],
                  "labels_only": fit_predict(*args, replace(CPRConfig(), use_founder=False)).score,
                  "cpr_lp": fit_predict(*args, CPRConfig()).score,
                  "cpr_lp_robust": fit_predict(*args, CPRConfig(robust_nu=NU)).score}
            er = np.random.default_rng([seed, 7])
            row = {"seed": seed, "rho": rho, "n": n}
            for k, v in sc.items():
                row[f"lift_{k}"] = lift_at_k(v[U], market.p[U], 100, er)
            rows.append(row)
    print(f"seed {seed} done", flush=True)

ms = lambda v: f"{np.mean(v):.2f} ± {np.std(v, ddof=1) / np.sqrt(len(v)):.2f}"  # noqa: E731
lines = [__doc__.strip(), "", f"nu = {NU}. Expected lift@100 on uncontacted accounts, mean ± SE over 10 seeds.", "",
         "| ρ | n | prior-only | labels-only | CPR-LP | CPR-LP robust | robust − CPR-LP |", "|---|---|---|---|---|---|---|"]
for rho in RHOS:
    for n in NS:
        cell = [r for r in rows if r["rho"] == rho and r["n"] == n]
        lines.append(f"| {rho} | {n} | " + " | ".join(ms([r[f"lift_{k}"] for r in cell]) for k in
                     ("prior_only", "labels_only", "cpr_lp", "cpr_lp_robust"))
                     + f" | {ms([r['lift_cpr_lp_robust'] - r['lift_cpr_lp'] for r in cell])} |")
md = "\n".join(lines)
print(md)
(ROOT / "results/robust_ladder.md").write_text(md)
(ROOT / "results/robust_ladder.json").write_text(json.dumps(rows))
