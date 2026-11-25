"""Diagnostic for the placebo-prior failure (results/placebo_bank.md, PL2 for PC-RFM).

PC-RFM beats labels-only by ~0.3–0.45 lift even with shuffled (uninformative) founder beliefs.
Hypothesis (selection awareness): the beliefs told the model WHY the contacted accounts were
contacted, which helps it interpret biased labels. Test: outreach driven by shuffled beliefs X,
model shown beliefs X ("matched") vs an independent shuffle Y ("decoupled"), vs no founder input.
  matched ≈ decoupled  -> not selection awareness (something else about having a founder input)
  matched >> decoupled -> selection awareness
Also: outreach uniformly random (no founder steering), model shown shuffled beliefs ("random-outreach").
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_pfn import load, pfn_scores  # noqa: E402
from fprior.cprlp import CPRConfig, fit_predict  # noqa: E402
from fprior.metrics import lift_at_k  # noqa: E402
from fprior.real_bank import load_bank, make_bank_market, split_pool  # noqa: E402
from fprior.sim import Founder, FounderConfig, OutreachConfig, run_outreach  # noqa: E402

NS = [5, 10, 20, 50]
SEEDS = 30


def main():
    pool = load_bank()
    A, B = split_pool(pool)
    pfn = load(ROOT / "runs/pfn_v0/model.pt", "cpu")
    pfn_nf = load(ROOT / "runs/pfn_v0_nf/model.pt", "cpu")
    cpr = replace(CPRConfig(), graph_sd=0.0)
    res = {k: {n: [] for n in NS} for k in ["matched", "decoupled", "nofounder", "rand_outreach_shown", "rand_outreach_nofounder"]}
    for seed in range(SEEDS):
        market = make_bank_market(pool, A, B, 3000, np.random.default_rng([seed, 5]))
        X = Founder(FounderConfig(), np.random.default_rng([seed, 77]).permutation(market.segment_logit))
        Y = Founder(FounderConfig(), np.random.default_rng([seed, 78]).permutation(market.segment_logit))
        for n in NS:
            for steer in ("founder", "random"):
                rng = np.random.default_rng([seed, 4000, n])
                oc = OutreachConfig(n_customers=n) if steer == "founder" else OutreachConfig(n_customers=n, exposure=0.0, explore=1.0)
                out = run_outreach(market, X, oc, rng)
                U = np.setdiff1d(np.arange(market.n), out.contacted)
                args = (market.segment, market.n_segments, market.graph_cov, X.segment_logit, out.contacted, out.outcome)
                base = fit_predict(*args, replace(cpr, use_founder=False)).score
                er = lambda: np.random.default_rng([seed, 7])  # noqa: E731
                lb = lift_at_k(base[U], market.y[U], 100, er())
                with torch.no_grad():
                    if steer == "founder":
                        res["matched"][n].append(lift_at_k(pfn_scores(pfn, market, X, out, True, "cpu")[U], market.y[U], 100, er()) - lb)
                        res["decoupled"][n].append(lift_at_k(pfn_scores(pfn, market, Y, out, True, "cpu")[U], market.y[U], 100, er()) - lb)
                        res["nofounder"][n].append(lift_at_k(pfn_scores(pfn_nf, market, X, out, False, "cpu")[U], market.y[U], 100, er()) - lb)
                    else:
                        res["rand_outreach_shown"][n].append(lift_at_k(pfn_scores(pfn, market, X, out, True, "cpu")[U], market.y[U], 100, er()) - lb)
                        res["rand_outreach_nofounder"][n].append(lift_at_k(pfn_scores(pfn_nf, market, X, out, False, "cpu")[U], market.y[U], 100, er()) - lb)
        print(f"seed {seed} done", flush=True)
    lines = [__doc__.strip(), "", f"Lift@100 minus CPR labels-only (same contacts), {SEEDS} seeds.", "",
             "| condition | " + " | ".join(f"n = {n}" for n in NS) + " |", "|---|" + "---|" * len(NS)]
    for k, v in res.items():
        lines.append(f"| {k} | " + " | ".join(f"{np.mean(v[n]):+.2f} ± {np.std(v[n], ddof=1) / np.sqrt(len(v[n])):.2f}" for n in NS) + " |")
    md = "\n".join(lines)
    print(md)
    (ROOT / "results/diag_placebo_selection.md").write_text(md)


if __name__ == "__main__":
    main()
