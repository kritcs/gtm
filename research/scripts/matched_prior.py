"""CPR-LP with the founder-trust prior matched to PC-RFM v0's training population (amortized vs explicit Bayes).

Derivation (scripts/diag_matched_prior.py, results/matched_prior_population.json): simulate 20,000 founders
from fprior/pfn/data.py PriorConfig defaults; per founder fit, by account-weighted least squares over the 32
segments, true_k − lbar = a + b (l_k − lbar) (CPR-LP's parameterization) and record the residual sd.
  all founders:   a ~ mean 0.00, sd 0.48;  b ~ mean 0.34, sd 0.55;  corr(a, b) ≈ 0;  residual sd median 0.74
  10% oracles:    a = 0, b = 1, residual 0 exactly
  90% others:     a ~ (0.00, 0.51), b ~ (0.26, 0.53), residual sd mean 0.86
CPR-LP's hand-set default (a ~ N(0, 0.5²), b ~ N(0.5, 0.5²)) is already close to the Gaussian fit.

Configs compared
  cpr_default        CPRConfig() (hand-set prior)
  cpr_matched_gauss  trust_mean (0, 0.34), trust_sd (0.48, 0.55)
  cpr_matched_mix    evidence-weighted mixture: 0.101 × oracle [b ~ N(1, 0.02²), a ~ N(0, 0.02²), omega = 0.05]
                     + 0.899 × broad [a ~ N(0, 0.51²), b ~ N(0.26, 0.53²), default omega grid]
  labels_only, pcrfm (runs/pfn_v0/model.pt)

Pre-registered (committed before running):
  MP1. On M0 (simulated) markets, in cells where |PC-RFM − cpr_default| > 0.2 lift, the better matched CPR-LP
       closes at least half of that gap on average (both then encode the same founder prior).
  MP2. On Bank (real outcomes): no prediction on sign; reported descriptively.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_pfn import load, pfn_scores  # noqa: E402
from fprior.cprlp import CPRConfig, fit_predict, fit_predict_mixture  # noqa: E402
from fprior.metrics import lift_at_k  # noqa: E402
from fprior.real_bank import load_bank, make_bank_market, split_pool  # noqa: E402
from fprior.sim import (FounderConfig, MarketConfig, OutreachConfig, make_founder,  # noqa: E402
                        make_market, run_outreach)

NS = [0, 1, 5, 10, 20, 50]
CPRS = ["cpr_default", "cpr_matched_gauss", "cpr_matched_mix"]
METHODS = ["labels_only", *CPRS, "pcrfm"]


def configs(base: CPRConfig):
    gauss = replace(base, trust_mean=(0.0, 0.34), trust_sd=(0.48, 0.55))
    oracle = replace(base, trust_mean=(0.0, 1.0), trust_sd=(0.02, 0.02), omega_grid=(0.05,))
    broad = replace(base, trust_mean=(0.0, 0.26), trust_sd=(0.51, 0.53))
    return gauss, [(0.101, oracle), (0.899, broad)]


def run_tier(tier: str, seeds: int, pfn):
    rows = []
    if tier == "m0":
        rhos, base = [1.0, 0.75, 0.5, 0.25, 0.0, -0.5], CPRConfig()
    else:
        rhos, base = [1.0, 0.5, 0.0, -0.5], replace(CPRConfig(), graph_sd=0.0)
        pool = load_bank()
        A, B = split_pool(pool)
    gauss, mix = configs(base)
    for seed in range(seeds):
        if tier == "m0":
            market = make_market(MarketConfig(), np.random.default_rng(seed))
            truth = market.p
        else:
            market = make_bank_market(pool, A, B, 3000, np.random.default_rng([seed, 5]))
            truth = market.y
        for rho in rhos:
            for n in NS:
                rng = np.random.default_rng([seed, int(round(rho * 100)) + 1000, n])
                founder = make_founder(market, FounderConfig(reliability=rho), rng)
                out = run_outreach(market, founder, OutreachConfig(n_customers=n), rng)
                U = np.setdiff1d(np.arange(market.n), out.contacted)
                args = (market.segment, market.n_segments, market.graph_cov, founder.segment_logit,
                        out.contacted, out.outcome)
                mixres = fit_predict_mixture(*args, mix)
                sc = {"labels_only": fit_predict(*args, replace(base, use_founder=False)).score,
                      "cpr_default": fit_predict(*args, base).score,
                      "cpr_matched_gauss": fit_predict(*args, gauss).score,
                      "cpr_matched_mix": mixres.logit}
                with torch.no_grad():
                    sc["pcrfm"] = pfn_scores(pfn, market, founder, out, True, "cpu")
                row = {"tier": tier, "seed": seed, "rho": rho, "n": n,
                       "w_oracle": float(mixres.component_weights[0])}
                for k in METHODS:
                    row[k] = lift_at_k(sc[k][U], truth[U], 100, np.random.default_rng([seed, 7]))
                rows.append(row)
        print(f"{tier} seed {seed} done", flush=True)
    return rows, rhos


def report(rows, rhos, tier):
    ms = lambda v: f"{np.mean(v):+.2f} ± {np.std(v, ddof=1) / np.sqrt(len(v)):.2f}"  # noqa: E731
    lines = [f"## {tier}: PC-RFM − each CPR-LP variant (lift@100, mean ± SE); mixture's posterior oracle weight", "",
             "| ρ | n | labels-only | cpr_default | cpr_matched_gauss | cpr_matched_mix | PC-RFM | PC-RFM − default | PC-RFM − gauss | PC-RFM − mix | w(oracle) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    gaps = []
    for rho in rhos:
        for n in NS:
            cell = [r for r in rows if r["rho"] == rho and r["n"] == n]
            mean = {k: np.mean([r[k] for r in cell]) for k in METHODS}
            vals = [f"{mean[k]:.2f}" for k in METHODS]
            vals += [ms([r["pcrfm"] - r[k] for r in cell]) for k in CPRS]
            vals.append(f"{np.mean([r['w_oracle'] for r in cell]):.2f}")
            lines.append(f"| {rho} | {n} | " + " | ".join(vals) + " |")
            g0 = mean["pcrfm"] - mean["cpr_default"]
            if abs(g0) > 0.2:
                best = min(abs(mean["pcrfm"] - mean["cpr_matched_gauss"]), abs(mean["pcrfm"] - mean["cpr_matched_mix"]))
                gaps.append(dict(rho=rho, n=n, gap_default=g0,
                                 gap_gauss=mean["pcrfm"] - mean["cpr_matched_gauss"],
                                 gap_mix=mean["pcrfm"] - mean["cpr_matched_mix"],
                                 closure=1 - best / abs(g0)))
    lines.append("")
    if gaps:
        lines += [f"Cells with |PC-RFM − cpr_default| > 0.2: {len(gaps)}; mean closure by the better matched variant: "
                  f"{np.mean([g['closure'] for g in gaps]):.2f}", "",
                  "| ρ | n | gap default | gap gauss | gap mix | closure |", "|---|---|---|---|---|---|"]
        lines += [f"| {g['rho']} | {g['n']} | {g['gap_default']:+.2f} | {g['gap_gauss']:+.2f} | {g['gap_mix']:+.2f} | {g['closure']:.2f} |"
                  for g in gaps]
    else:
        lines.append("No cell has |PC-RFM − cpr_default| > 0.2.")
    lines.append("")
    return lines, gaps


def main():
    seeds_m0 = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    seeds_bank = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    pfn = load(ROOT / "runs/pfn_v0/model.pt", "cpu")
    r_m0, rh_m0 = run_tier("m0", seeds_m0, pfn)
    r_bank, rh_bank = run_tier("bank", seeds_bank, pfn)
    l1, g1 = report(r_m0, rh_m0, f"M0 simulated markets ({seeds_m0} seeds, expected lift)")
    l2, g2 = report(r_bank, rh_bank, f"Bank real outcomes ({seeds_bank} seeds, realized lift)")
    md = "\n".join([__doc__.strip(), "", *l1, *l2])
    print(md)
    (ROOT / "results/matched_prior.md").write_text(md)
    (ROOT / "results/matched_prior.json").write_text(json.dumps({"m0": r_m0, "bank": r_bank, "gaps_m0": g1, "gaps_bank": g2}))


if __name__ == "__main__":
    main()
