"""Founder benefit for PC-RFM measured against its OWN no-information mode (UCI Bank Marketing).

After the placebo tests (results/diag_placebo_selection.md), the fair founder-benefit comparison for
PC-RFM is the same network shown (a) the founder's beliefs vs (b) the same values scrambled across
accounts (no information). Same markets, outreach and relabelling as scripts/real_bank.py.

Pre-registered (committed before running):
  NC1. ρ = 1: true − noise > 0 at every n in {0, 1, 5, 10, 20} (> 2 SE).
  NC2. ρ = 0.5: true − noise > 0 at n ≤ 5.
  NC3. ρ ∈ {0, −0.5}, n ≥ 5: |true − noise| < 0.2 (a useless founder adds nothing beyond the model's
       own label learning; an inverted one may cost a little at small n).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_pfn import load, pfn_scores  # noqa: E402
from fprior.metrics import lift_at_k  # noqa: E402
from fprior.real_bank import load_bank, make_bank_market, split_pool  # noqa: E402
from fprior.sim import Founder, FounderConfig, OutreachConfig, make_founder, run_outreach  # noqa: E402

RHOS = [1.0, 0.5, 0.0, -0.5]
NS = [0, 1, 5, 10, 20, 50]
SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
CKPT = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "runs/pfn_v0/model.pt")
OUT = sys.argv[3] if len(sys.argv) > 3 else "results/bank_noise_control"   # path stem, .md/.json appended


def main():
    pool = load_bank()
    A, B = split_pool(pool)
    pfn = load(CKPT, "cpu")
    rows = []
    for seed in range(SEEDS):
        market = make_bank_market(pool, A, B, 3000, np.random.default_rng([seed, 5]))
        for rho in RHOS:
            for n in NS:
                rng = np.random.default_rng([seed, int(round(rho * 100)) + 1000, n])
                founder = make_founder(market, FounderConfig(reliability=rho), rng)
                out = run_outreach(market, founder, OutreachConfig(n_customers=n), rng)
                U = np.setdiff1d(np.arange(market.n), out.contacted)
                with torch.no_grad():
                    s_true = pfn_scores(pfn, market, founder, out, True, "cpu")
                    # noise: scramble the per-account belief values across accounts
                    acc = founder.segment_logit[market.segment]
                    perm = np.random.default_rng([seed, 79, n]).permutation(acc)
                    seg_backup = market.segment
                    market.segment = np.arange(market.n)          # one "segment" per account
                    s_noise = pfn_scores(pfn, market, Founder(founder.cfg, perm), out, True, "cpu")
                    market.segment = seg_backup
                er = lambda: np.random.default_rng([seed, 7])  # noqa: E731
                rows.append({"seed": seed, "rho": rho, "n": n,
                             "true": lift_at_k(s_true[U], market.y[U], 100, er()),
                             "noise": lift_at_k(s_noise[U], market.y[U], 100, er())})
        print(f"seed {seed} done", flush=True)
    ms = lambda v: f"{np.mean(v):+.2f} ± {np.std(v, ddof=1) / np.sqrt(len(v)):.2f}"  # noqa: E731
    lines = [__doc__.strip(), "", f"## Results ({SEEDS} seeds): PC-RFM v0, true beliefs − noise beliefs (lift@100)", "",
             "| ρ | " + " | ".join(f"n = {n}" for n in NS) + " |", "|---|" + "---|" * len(NS)]
    for rho in RHOS:
        cells = [ms([r["true"] - r["noise"] for r in rows if r["rho"] == rho and r["n"] == n]) for n in NS]
        lines.append(f"| {rho} | " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    print(md)
    (ROOT / f"{OUT}.md").write_text(md)
    (ROOT / f"{OUT}.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
