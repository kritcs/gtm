"""Placebo priors on UCI Bank Marketing (evaluation-brainstorm.md idea #3).

If the gain over labels-only really comes from the founder's knowledge, it must disappear when the
beliefs are destroyed while keeping their format:
  true      — rho = 1 founder (segment rates from split A)
  shuffled  — the same belief values, randomly permuted across segments (same spread, no information)
  flat      — every segment believed at the overall base rate (no spread, no information)

Pre-registered (committed before the run):
  PL1. For true beliefs, CPR-LP and PC-RFM beat labels-only (replicates R1).
  PL2. For shuffled and flat beliefs at n >= 5, |method − labels-only| < 0.2 lift, i.e. the gain
       vanishes; shuffled may hurt at n = 0–1 (a confident wrong prior with no labels).
  PL3. Learned trust b (CPR-LP) is lower for shuffled than for true beliefs at every n >= 5.
Metric: lift@100 on uncontacted accounts with real outcomes; 30 seeds; same markets and
relabelling control as scripts/real_bank.py.
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
from fprior.cprlp import CPRConfig, fit_predict  # noqa: E402
from fprior.metrics import lift_at_k  # noqa: E402
from fprior.real_bank import load_bank, make_bank_market, split_pool  # noqa: E402
from fprior.sim import Founder, FounderConfig, OutreachConfig, run_outreach  # noqa: E402

CONDS = ["true", "shuffled", "flat"]
NS = [0, 1, 5, 10, 20, 50]
SEEDS = 30


def main():
    pool = load_bank()
    A, B = split_pool(pool)
    pfn = load(ROOT / "runs/pfn_v0/model.pt", "cpu")
    cpr = replace(CPRConfig(), graph_sd=0.0)
    rows = []
    for seed in range(SEEDS):
        market = make_bank_market(pool, A, B, 3000, np.random.default_rng([seed, 5]))
        for cond in CONDS:
            belief = market.segment_logit.copy()
            if cond == "shuffled":
                belief = np.random.default_rng([seed, 77]).permutation(belief)
            elif cond == "flat":
                belief = np.full_like(belief, market.intercept)
            founder = Founder(FounderConfig(reliability=1.0), belief)
            for n in NS:
                rng = np.random.default_rng([seed, 3000 + CONDS.index(cond), n])
                out = run_outreach(market, founder, OutreachConfig(n_customers=n), rng)
                U = np.setdiff1d(np.arange(market.n), out.contacted)
                args = (market.segment, market.n_segments, market.graph_cov, founder.segment_logit,
                        out.contacted, out.outcome)
                fit = fit_predict(*args, cpr)
                sc = {"labels_only": fit_predict(*args, replace(cpr, use_founder=False)).score,
                      "cpr_lp": fit.score}
                with torch.no_grad():
                    sc["pcrfm"] = pfn_scores(pfn, market, founder, out, True, "cpu")
                er = np.random.default_rng([seed, 7])
                row = {"seed": seed, "cond": cond, "n": n, "trust_b": float(fit.trust[1])}
                for k, v in sc.items():
                    row[f"lift_{k}"] = lift_at_k(v[U], market.y[U], 100, er)
                rows.append(row)
        print(f"seed {seed} done", flush=True)

    ms = lambda v: f"{np.mean(v):+.2f} ± {np.std(v, ddof=1) / np.sqrt(len(v)):.2f}"  # noqa: E731
    lines = [__doc__.strip(), "", f"## Results ({SEEDS} seeds)", "",
             "Paired lift@100 differences vs labels-only, and CPR-LP's learned trust b.", "",
             "| beliefs | n | CPR-LP − labels | PC-RFM − labels | learned b |", "|---|---|---|---|---|"]
    for cond in CONDS:
        for n in NS:
            cell = [r for r in rows if r["cond"] == cond and r["n"] == n]
            lines.append(f"| {cond} | {n} | {ms([r['lift_cpr_lp'] - r['lift_labels_only'] for r in cell])} | "
                         f"{ms([r['lift_pcrfm'] - r['lift_labels_only'] for r in cell])} | "
                         f"{np.mean([r['trust_b'] for r in cell]):.2f} |")
    md = "\n".join(lines)
    print(md)
    (ROOT / "results/placebo_bank.md").write_text(md)
    (ROOT / "results/placebo_bank.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
