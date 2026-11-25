"""Bad founders on UCI Bank Marketing: Gaussian CPR-LP vs Student-t CPR-LP vs PC-RFM (vs labels-only).

Follow-up to results/bank_noise_control.md (NC3 failed: PC-RFM loses ~0.2–0.3 to its own no-information
mode with an inverted founder at n = 10–20). Same markets/outreach as scripts/real_bank.py.

Pre-registered (committed before running):
  BF1. ρ = −0.5, n ≥ 5: Student-t CPR-LP − labels-only ≥ Gaussian CPR-LP − labels-only (robust prior
       discounts a systematically wrong founder at least as well).
  BF2. ρ = 1, all n: Student-t CPR-LP within 0.1 of Gaussian CPR-LP (no cost with a good founder).
Exploratory: PC-RFM in the same cells.
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
from fprior.sim import FounderConfig, OutreachConfig, make_founder, run_outreach  # noqa: E402

RHOS = [1.0, 0.0, -0.5]
NS = [0, 1, 5, 10, 20, 50]
SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def main():
    pool = load_bank()
    A, B = split_pool(pool)
    pfn = load(ROOT / "runs/pfn_v0/model.pt", "cpu")
    gauss = replace(CPRConfig(), graph_sd=0.0)
    robust = replace(gauss, robust_nu=3.0)
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
                sc = {"labels_only": fit_predict(*args, replace(gauss, use_founder=False)).score,
                      "cpr_gauss": fit_predict(*args, gauss).score,
                      "cpr_t": fit_predict(*args, robust).score}
                with torch.no_grad():
                    sc["pcrfm"] = pfn_scores(pfn, market, founder, out, True, "cpu")
                row = {"seed": seed, "rho": rho, "n": n}
                for k, v in sc.items():
                    row[k] = lift_at_k(v[U], market.y[U], 100, np.random.default_rng([seed, 7]))
                rows.append(row)
        print(f"seed {seed} done", flush=True)
    ms = lambda v: f"{np.mean(v):+.2f} ± {np.std(v, ddof=1) / np.sqrt(len(v)):.2f}"  # noqa: E731
    lines = [__doc__.strip(), "", f"## Results ({SEEDS} seeds): method − labels-only, lift@100", ""]
    for rho in RHOS:
        lines += [f"**ρ = {rho}**", "", "| method | " + " | ".join(f"n = {n}" for n in NS) + " |", "|---|" + "---|" * len(NS)]
        for m in ("cpr_gauss", "cpr_t", "pcrfm"):
            lines.append(f"| {m} | " + " | ".join(ms([r[m] - r["labels_only"] for r in rows if r["rho"] == rho and r["n"] == n]) for n in NS) + " |")
        lines.append("| cpr_t − cpr_gauss | " + " | ".join(ms([r["cpr_t"] - r["cpr_gauss"] for r in rows if r["rho"] == rho and r["n"] == n]) for n in NS) + " |")
        lines.append("")
    md = "\n".join(lines)
    print(md)
    (ROOT / "results/bank_bad_founders.md").write_text(md)
    (ROOT / "results/bank_bad_founders.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
