"""Bank tier with hierarchical CPR-LP (main effects + cell; fprior/cprlp_hier.py) as the baseline.

Pre-registered (committed before running; motivated by results/diag_learner_gap.md):
  HB1. ρ ∈ {0, −0.5}, n ≥ 10: CPR-LP-hier − labels-only-hier ≥ −0.1 (bad founders cost little once the
       model can pool through main effects).
  HB2. ρ = 1, n ≤ 20: CPR-LP-hier − labels-only-hier > 0 (> 2 SE): founder benefit survives a stronger
       label learner.
  HB3. |PC-RFM − CPR-LP-hier| < 0.2 in at least two thirds of the 24 (ρ, n) cells (the "learner gap" is closed).
Same markets / outreach / relabelling as scripts/real_bank.py; 30 seeds; lift@100 on real outcomes.
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
from fprior.cprlp_hier import fit_predict_hier  # noqa: E402
from fprior.metrics import lift_at_k  # noqa: E402
from fprior.real_bank import load_bank, make_bank_market, split_pool  # noqa: E402
from fprior.sim import FounderConfig, OutreachConfig, make_founder, run_outreach  # noqa: E402

RHOS = [1.0, 0.5, 0.0, -0.5]
NS = [0, 1, 5, 10, 20, 50]
SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def main():
    pool = load_bank()
    A, B = split_pool(pool)
    pfn = load(ROOT / "runs/pfn_v0/model.pt", "cpu")
    cfg = replace(CPRConfig(), graph_sd=0.0)
    rows = []
    for seed in range(SEEDS):
        market = make_bank_market(pool, A, B, 3000, np.random.default_rng([seed, 5]))
        for rho in RHOS:
            for n in NS:
                rng = np.random.default_rng([seed, int(round(rho * 100)) + 1000, n])
                founder = make_founder(market, FounderConfig(reliability=rho), rng)
                out = run_outreach(market, founder, OutreachConfig(n_customers=n), rng)
                U = np.setdiff1d(np.arange(market.n), out.contacted)
                args = (market.segment, market.n_segments, market.graph_cov, founder.segment_logit,
                        out.contacted, out.outcome)
                sc = {"cpr_flat": fit_predict(*args, cfg).score,
                      "labels_hier": fit_predict_hier(*args, market.industry, market.size, replace(cfg, use_founder=False)).score,
                      "cpr_hier": fit_predict_hier(*args, market.industry, market.size, cfg).score}
                with torch.no_grad():
                    sc["pcrfm"] = pfn_scores(pfn, market, founder, out, True, "cpu")
                row = {"seed": seed, "rho": rho, "n": n}
                for k, v in sc.items():
                    row[k] = lift_at_k(v[U], market.y[U], 100, np.random.default_rng([seed, 7]))
                rows.append(row)
        print(f"seed {seed} done", flush=True)
    ms = lambda v: f"{np.mean(v):+.2f} ± {np.std(v, ddof=1) / np.sqrt(len(v)):.2f}"  # noqa: E731
    lines = [__doc__.strip(), "", f"## Results ({SEEDS} seeds; lift@100 differences)", ""]
    for title, a, b in [("CPR-LP-hier − labels-only-hier", "cpr_hier", "labels_hier"),
                        ("PC-RFM − CPR-LP-hier", "pcrfm", "cpr_hier"),
                        ("CPR-LP-hier − CPR-LP (flat)", "cpr_hier", "cpr_flat")]:
        lines += [f"**{title}**", "", "| ρ | " + " | ".join(f"n = {n}" for n in NS) + " |", "|---|" + "---|" * len(NS)]
        for rho in RHOS:
            lines.append(f"| {rho} | " + " | ".join(ms([r[a] - r[b] for r in rows if r["rho"] == rho and r["n"] == n]) for n in NS) + " |")
        lines.append("")
    md = "\n".join(lines)
    print(md)
    (ROOT / "results/real_bank_hier.md").write_text(md)
    (ROOT / "results/real_bank_hier.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
