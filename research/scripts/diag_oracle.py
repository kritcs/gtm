"""Diagnostic: why does CPR-LP trail prior-only with an oracle founder at small n?

Runs CPR variants on the same markets / founders / outreach as the sanity ladder and
prints mean lift@100 (± SE, paired diff vs prior-only) per n.

Usage: python scripts/diag_oracle.py [--seeds 10] [--rho 1.0] [--variants a,b,...]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fprior.cprlp import CPRConfig, fit_predict  # noqa: E402
from fprior.metrics import lift_at_k  # noqa: E402
from fprior.sim import (FounderConfig, MarketConfig, OutreachConfig, make_founder,  # noqa: E402
                        make_market, run_outreach)

NS = [1, 5, 10, 20, 50]
BASE = CPRConfig()
VARIANTS = {
    "current": BASE,
    # well-specified for the oracle: (a, b) = (0, 1) fixed, no segment deviation
    "oracle_model": replace(BASE, learn_trust=False, omega_grid=None, omega=0.01),
    "oracle_trust_omega1": replace(BASE, learn_trust=False, omega_grid=None, omega=1.0),
    "oracle_trust_omegaEB": replace(BASE, learn_trust=False),
    "b_prior_1": replace(BASE, trust_mean=(0.0, 1.0)),
    "b_prior_1_tight": replace(BASE, trust_mean=(0.0, 1.0), trust_sd=(0.5, 0.25)),
    "omega_fixed_0.3": replace(BASE, omega_grid=None, omega=0.3),
    "no_graph": replace(BASE, graph_sd=1e-3),
    "omega_bma": replace(BASE, omega_average=True),
    "rank_by_p": replace(BASE, rank_by="p"),
    "bma_rank_p": replace(BASE, omega_average=True, rank_by="p"),
    "omega_bma_b1": replace(BASE, omega_average=True, trust_mean=(0.0, 1.0)),
    "bma_rank_p_b1": replace(BASE, omega_average=True, rank_by="p", trust_mean=(0.0, 1.0)),
    "oracle_model_p": replace(BASE, learn_trust=False, omega_grid=None, omega=0.01, rank_by="p"),
}
LABELS = {"labels_only": replace(BASE, use_founder=False),
          "labels_only_bma": replace(BASE, use_founder=False, omega_average=True),
          "labels_only_p": replace(BASE, use_founder=False, rank_by="p"),
          "labels_only_bma_p": replace(BASE, use_founder=False, omega_average=True, rank_by="p")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--accounts", type=int, default=3000)
    ap.add_argument("--variants", default=",".join(k for k, v in VARIANTS.items() if v is not None))
    a = ap.parse_args()
    names = a.variants.split(",")
    res = {nm: {n: [] for n in NS} for nm in names + ["prior_only"] + list(LABELS)}
    bvals = {nm: {n: [] for n in NS} for nm in names}
    for seed in range(a.seeds):
        market = make_market(MarketConfig(n_accounts=a.accounts), np.random.default_rng(seed))
        for n in NS:
            rng = np.random.default_rng([seed, int(round(a.rho * 100)) + 1000, n])
            founder = make_founder(market, FounderConfig(reliability=a.rho), rng)
            out = run_outreach(market, founder, OutreachConfig(n_customers=n), rng)
            U = np.setdiff1d(np.arange(market.n), out.contacted)
            args = (market.segment, market.n_segments, market.graph_cov, founder.segment_logit,
                    out.contacted, out.outcome)
            ev = np.random.default_rng([seed, 7])
            st = ev.bit_generator.state
            def lift(s):
                ev.bit_generator.state = st
                return lift_at_k(s[U], market.p[U], 100, ev)
            res["prior_only"][n].append(lift(founder.segment_logit[market.segment]))
            for ln, lc in LABELS.items():
                res[ln][n].append(lift(fit_predict(*args, lc).score))
            for nm in names:
                r = fit_predict(*args, VARIANTS[nm])
                res[nm][n].append(lift(r.score))
                bvals[nm][n].append(np.nan if r.trust is None else r.trust[1])
    po = {n: np.array(res["prior_only"][n]) for n in NS}
    print(f"rho={a.rho} seeds={a.seeds}  lift@100 mean (diff vs prior-only ± SE) [mean b]")
    print("variant".ljust(22) + "".join(f"n={n}".ljust(26) for n in NS))
    for nm in ["prior_only"] + list(LABELS) + names:
        row = nm.ljust(22)
        for n in NS:
            x = np.array(res[nm][n])
            d = x - po[n]
            b = f" b={np.nanmean(bvals[nm][n]):.2f}" if nm in bvals else ""
            row += f"{x.mean():.2f} ({d.mean():+.2f}±{d.std(ddof=1)/np.sqrt(len(d)):.2f}){b}".ljust(26)
        print(row)


if __name__ == "__main__":
    main()
