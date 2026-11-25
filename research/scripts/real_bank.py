"""Real-outcome tier 1 on UCI Bank Marketing: real outcomes, simulated founder.

Pre-registered (written and committed before the full run):
  R1. With the best-informed founder (rho = 1: beliefs = segment rates from split A), CPR-LP and
      PC-RFM v0 beat labels-only at n in {0, 1, 5, 10, 20}.
  R2. With wrong founders (rho in {0, -0.5}), CPR-LP and PC-RFM lose at most 0.2 lift to labels-only
      for n >= 5.
  R3. Sim-to-real gap: PC-RFM v0 (trained only on simulations) is within 0.2 lift of CPR-LP
      (no simulator training) in at least two thirds of (rho, n) cells. If it is worse by more,
      that gap is the measured cost of simulator mismatch.
  Control (added after a 2-seed smoke run, before the full run): job-group and age-band indices are
  randomly relabelled per market, because the smoke run showed the no-founder PC-RFM scoring
  lift ~3.3 with zero labels by an accidental match between category numbering and its training
  habits. CPR-LP is unaffected by relabelling.
  Caveat stated in advance: if job x age segments carry little real signal, founder priors cannot
  help much here; the rho = 1 prior-only row measures that ceiling.

Metric: lift@100 on uncontacted accounts using REAL outcomes (noisier than the simulator's
expected lift, so 30 seeds), and AUC.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fprior.cprlp import CPRConfig, fit_predict  # noqa: E402
from fprior.metrics import expected_auc, lift_at_k  # noqa: E402
from fprior.real_bank import load_bank, make_bank_market, split_pool  # noqa: E402
from fprior.sim import FounderConfig, OutreachConfig, make_founder, run_outreach  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from eval_pfn import load, pfn_scores  # noqa: E402

RHOS = [1.0, 0.5, 0.0, -0.5]
NS = [0, 1, 5, 10, 20, 50]
TOP_K = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs/pfn_v0/model.pt"))
    ap.add_argument("--ckpt-nf", default=str(ROOT / "runs/pfn_v0_nf/model.pt"))
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--n-accounts", type=int, default=3000)
    ap.add_argument("--out", default="results/real_bank.md")
    a = ap.parse_args()
    pool = load_bank()
    A, B = split_pool(pool)
    models = {"pcrfm": (load(a.ckpt, "cpu"), True), "pcrfm_nofounder": (load(a.ckpt_nf, "cpu"), False)}
    cpr = replace(CPRConfig(), graph_sd=0.0)          # no graph in this tier
    methods = ["random", "prior_only", "labels_only", "cpr_lp", *models]
    rows = []
    for seed in range(a.seeds):
        market = make_bank_market(pool, A, B, a.n_accounts, np.random.default_rng([seed, 5]))
        for rho in RHOS:
            for n in NS:
                rng = np.random.default_rng([seed, int(round(rho * 100)) + 1000, n])
                founder = make_founder(market, FounderConfig(reliability=rho), rng)
                out = run_outreach(market, founder, OutreachConfig(n_customers=n), rng)
                U = np.setdiff1d(np.arange(market.n), out.contacted)
                args = (market.segment, market.n_segments, market.graph_cov, founder.segment_logit,
                        out.contacted, out.outcome)
                sc = {"random": np.zeros(market.n), "prior_only": founder.segment_logit[market.segment],
                      "labels_only": fit_predict(*args, replace(cpr, use_founder=False)).score,
                      "cpr_lp": fit_predict(*args, cpr).score}
                with torch.no_grad():
                    for name, (m, uf) in models.items():
                        sc[name] = pfn_scores(m, market, founder, out, uf, "cpu")
                er = np.random.default_rng([seed, 7])
                row = {"seed": seed, "rho": rho, "n": n, "contacts": int(len(out.contacted))}
                for k in methods:
                    row[f"lift_{k}"] = lift_at_k(sc[k][U], market.y[U], TOP_K, er)
                    row[f"auc_{k}"] = expected_auc(sc[k][U], market.y[U])
                rows.append(row)
        print(f"seed {seed} done", flush=True)

    ms = lambda v: f"{np.mean(v):.2f} ± {np.std(v, ddof=1) / np.sqrt(len(v)):.2f}"  # noqa: E731
    seg_logit = make_bank_market(pool, A, B, 10, np.random.default_rng(0)).segment_logit
    lines = [__doc__.strip(), "",
             f"## Results ({a.seeds} seeds, markets of {a.n_accounts} real clients from split B)", "",
             f"Spread of real segment logits (split A): sd = {seg_logit.std():.2f}, "
             f"range {seg_logit.min():.2f} to {seg_logit.max():.2f}.", ""]
    for metric in ("lift", "auc"):
        lines += [f"### {'lift@100' if metric == 'lift' else 'AUC'}", ""]
        for rho in RHOS:
            hdr = ["n", "contacts", *methods, "CPR − labels", "PC-RFM − labels", "PC-RFM − CPR-LP",
                   "PC-RFM − no-founder"]
            lines += [f"**ρ = {rho}**", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
            for n in NS:
                cell = [r for r in rows if r["rho"] == rho and r["n"] == n]
                d = lambda x, y: ms([r[f"{metric}_{x}"] - r[f"{metric}_{y}"] for r in cell])  # noqa: E731
                vals = [f"{np.mean([r['contacts'] for r in cell]):.0f}"]
                vals += [ms([r[f"{metric}_{k}"] for r in cell]) for k in methods]
                vals += [d("cpr_lp", "labels_only"), d("pcrfm", "labels_only"), d("pcrfm", "cpr_lp"),
                         d("pcrfm", "pcrfm_nofounder")]
                lines.append(f"| {n} | " + " | ".join(vals) + " |")
            lines.append("")
    md = "\n".join(lines)
    print(md)
    (ROOT / a.out).write_text(md)
    (ROOT / a.out).with_suffix(".json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
