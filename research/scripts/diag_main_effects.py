"""Does CPR-LP close the gap to PC-RFM if it can share evidence across segments through main effects?
Hierarchical kernel via random effects: one-hot industry (job group) and size (age band) columns as
extra_features with N(0, extra_sd^2) coefficients == kernel extra_sd^2 ([same industry] + [same size]).
Bank tier, same cells as diag_nugget_avg.py. Exploratory diagnostic (no pre-registration)."""
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from eval_pfn import load, pfn_scores
from fprior.cprlp import CPRConfig, fit_predict
from fprior.metrics import lift_at_k
from fprior.real_bank import load_bank, make_bank_market, split_pool
from fprior.sim import FounderConfig, OutreachConfig, make_founder, run_outreach
pool = load_bank(); A, B = split_pool(pool); pfn = load(ROOT / "runs/pfn_v0/model.pt", "cpu")
base = replace(CPRConfig(), graph_sd=0.0)
SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
for rho in (1.0, 0.0, -0.5):
    for n in (5, 10, 20, 50):
        d = {"cpr": [], "cpr_main": [], "labels_main": [], "pcrfm": []}
        for seed in range(SEEDS):
            market = make_bank_market(pool, A, B, 3000, np.random.default_rng([seed, 5]))
            rng = np.random.default_rng([seed, int(round(rho * 100)) + 1000, n])
            f = make_founder(market, FounderConfig(reliability=rho), rng)
            out = run_outreach(market, f, OutreachConfig(n_customers=n), rng)
            U = np.setdiff1d(np.arange(market.n), out.contacted)
            X = np.column_stack([np.eye(8)[market.industry], np.eye(4)[market.size]])
            args = (market.segment, market.n_segments, market.graph_cov, f.segment_logit, out.contacted, out.outcome)
            L = lambda s: lift_at_k(s[U], market.y[U], 100, np.random.default_rng([seed, 7]))
            lab = L(fit_predict(*args, replace(base, use_founder=False)).score)
            with torch.no_grad():
                p = pfn_scores(pfn, market, f, out, True, "cpu")
            d["cpr"].append(L(fit_predict(*args, base).score) - lab)
            d["cpr_main"].append(L(fit_predict(*args, base, extra_features=X).score) - lab)
            d["labels_main"].append(L(fit_predict(*args, replace(base, use_founder=False), extra_features=X).score) - lab)
            d["pcrfm"].append(L(p) - lab)
        print(rho, n, {k: f"{np.mean(v):+.2f}±{np.std(v, ddof=1) / np.sqrt(len(v)):.2f}" for k, v in d.items()}, flush=True)
