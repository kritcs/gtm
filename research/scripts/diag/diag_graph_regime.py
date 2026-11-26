"""Exploratory: in which simulator regime does CPR-LP's (oracle) graph term help? Sweep graph variance share,
smoothness and degree; founder rho = 0.5; S0 outreach. Metric: lift@100 of CPR-LP − CPR-LP without graph."""
import sys, itertools, json, numpy as np
sys.path.insert(0, ".")
from fprior.cprlp import CPRConfig, fit_predict
from fprior.metrics import lift_at_k
from fprior.sim import FounderConfig, MarketConfig, OutreachConfig, make_founder, make_market, run_outreach
SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
res = []
for wg, gam, deg in itertools.product([0.3, 0.6], [5.0, 50.0], [10.0, 30.0]):
    cfg = MarketConfig(w_segment=0.5 * (1 - wg) / 0.7, w_graph=wg, w_latent=0.2 * (1 - wg) / 0.7,
                       smoothness=gam, avg_degree=deg)
    for seed in range(SEEDS):
        m = make_market(cfg, np.random.default_rng(seed))
        for n in [5, 20, 50]:
            rng = np.random.default_rng([seed, 1050, n])
            f = make_founder(m, FounderConfig(reliability=0.5), rng)
            out = run_outreach(m, f, OutreachConfig(n_customers=n), rng)
            U = np.setdiff1d(np.arange(m.n), out.contacted)
            er = np.random.default_rng([seed, 7])
            l = [lift_at_k(fit_predict(m.segment, m.n_segments, m.graph_cov, f.segment_logit, out.contacted,
                                       out.outcome, c).score[U], m.p[U], 100, er)
                 for c in (CPRConfig(graph_sd=float(np.sqrt(wg) * 1.5)), CPRConfig(graph_sd=1e-3))]
            res.append(dict(wg=wg, gam=gam, deg=deg, seed=seed, n=n, contacts=len(out.contacted), d=l[0] - l[1]))
    print(f"done wg={wg} gam={gam} deg={deg}", flush=True)
json.dump(res, open("results/diag_graph_regime.json", "w"))
print("| w_graph | smoothness γ | avg degree | n | contacts (median) | CPR-LP − no-graph |\n|---|---|---|---|---|---|")
for wg, gam, deg, n in itertools.product([0.3, 0.6], [5.0, 50.0], [10.0, 30.0], [5, 20, 50]):
    r = [x for x in res if (x["wg"], x["gam"], x["deg"], x["n"]) == (wg, gam, deg, n)]
    d = np.array([x["d"] for x in r])
    print(f"| {wg} | {gam:g} | {deg:g} | {n} | {int(np.median([x['contacts'] for x in r]))} | {d.mean():+.3f} ± {d.std(ddof=1)/np.sqrt(len(d)):.3f} |")
