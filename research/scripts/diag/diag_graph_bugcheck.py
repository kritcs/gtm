"""Exploratory bug check: with a strong, smooth graph and many labels, CPR-LP's oracle graph term must help.
Also reports the ceiling: lift of ranking by the true segment part + true graph field (no latent)."""
import sys, numpy as np
sys.path.insert(0, ".")
from fprior.cprlp import CPRConfig, fit_predict
from fprior.metrics import lift_at_k
from fprior.sim import FounderConfig, MarketConfig, OutreachConfig, make_founder, make_market, run_outreach
cfg = MarketConfig(w_segment=0.2, w_graph=0.7, w_latent=0.1, smoothness=50.0)
for n in [20, 100]:
    d = []
    for seed in range(6):
        m = make_market(cfg, np.random.default_rng(seed))
        rng = np.random.default_rng([seed, 3, n])
        f = make_founder(m, FounderConfig(reliability=0.5), rng)
        out = run_outreach(m, f, OutreachConfig(n_customers=n, max_contacts=2500), rng)
        U = np.setdiff1d(np.arange(m.n), out.contacted); er = np.random.default_rng([seed, 7])
        L = lambda s: lift_at_k(s[U], m.p[U], 100, er)
        g = L(fit_predict(m.segment, m.n_segments, m.graph_cov, f.segment_logit, out.contacted, out.outcome,
                          CPRConfig(graph_sd=float(np.sqrt(0.7) * 1.5))).score)
        ng = L(fit_predict(m.segment, m.n_segments, m.graph_cov, f.segment_logit, out.contacted, out.outcome,
                           CPRConfig(graph_sd=1e-3)).score)
        d.append((len(out.contacted), g, ng, L(m.p)))
    d = np.array(d)
    print(f"n={n}: contacts {d[:,0].mean():.0f}; CPR-LP {d[:,1].mean():.2f}, no-graph {d[:,2].mean():.2f}, "
          f"diff {np.mean(d[:,1]-d[:,2]):+.3f} ± {np.std(d[:,1]-d[:,2],ddof=1)/np.sqrt(6):.3f}; oracle p {d[:,3].mean():.2f}")
