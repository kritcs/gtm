"""Exploratory: does CPR-LP's graph term help in the simulator? Same markets/founders/outreach as eval_v0g.py.
cpr = default; cpr_nog = graph_sd ~ 0; cpr_perm = node-permuted graph covariance. Lift@100, seed-level mean ± SE."""
import sys, numpy as np
from dataclasses import replace
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from eval_v0g import RHOS, NS, SH
from fprior.cprlp import CPRConfig, fit_predict
from fprior.metrics import lift_at_k
from fprior.sim import FounderConfig, MarketConfig, OutreachConfig, make_founder, make_market
from fprior.sim_shift import SHIFTS, outreach_for, shift_market
rows = []
for seed in range(20):
    base = make_market(MarketConfig(), np.random.default_rng(seed))
    for sh in SH:
        m = shift_market(base, sh, np.random.default_rng([seed, 500 + SHIFTS.index(sh)]))
        perm = np.random.default_rng([seed, 900]).permutation(m.n)
        Cp = m.graph_cov[perm][:, perm]
        for rho in RHOS:
            for n in NS:
                rng = np.random.default_rng([seed, int(round(rho * 100)) + 1000, n])
                f = make_founder(m, FounderConfig(reliability=rho), rng)
                out = outreach_for(sh, m, f, OutreachConfig(n_customers=n), rng)
                U = np.setdiff1d(np.arange(m.n), out.contacted)
                fp = lambda C, cfg: fit_predict(m.segment, m.n_segments, C, f.segment_logit, out.contacted, out.outcome, cfg).score
                sc = {"cpr": fp(m.graph_cov, CPRConfig()), "nog": fp(m.graph_cov, CPRConfig(graph_sd=1e-3)),
                      "perm": fp(Cp, CPRConfig())}
                er = np.random.default_rng([seed, 7])
                rows.append(dict(sh=sh, seed=seed, rho=rho, n=n, **{k: lift_at_k(v[U], m.p[U], 100, er) for k, v in sc.items()}))
    print("seed", seed, flush=True)
def d(sh, x, y, ns):
    c = {}
    for r in rows:
        if r["sh"] == sh and r["n"] in ns: c.setdefault(r["seed"], []).append(r[x] - r[y])
    v = np.array([np.mean(z) for z in c.values()]); return f"{v.mean():+.3f} ± {v.std(ddof=1)/np.sqrt(len(v)):.3f}"
for sh in SH:
    for ns in ([0], [5, 10, 20]):
        print(f"{sh} n∈{ns}: cpr − cpr(no graph) {d(sh,'cpr','nog',ns)} | cpr − cpr(permuted graph) {d(sh,'cpr','perm',ns)}")
