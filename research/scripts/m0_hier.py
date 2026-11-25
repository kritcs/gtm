"""M0 simulated markets: hierarchical CPR-LP vs flat CPR-LP vs PC-RFM v0 (exploratory).
Also re-checks the step-1 gate (oracle founder must beat labels-only by > 2 SE at n = 0-20) for the
hierarchical model. Same seeds / founders / outreach as scripts/sanity_ladder.py."""
import json, sys
from dataclasses import replace
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from eval_pfn import load, pfn_scores
from fprior.cprlp import CPRConfig, fit_predict
from fprior.cprlp_hier import fit_predict_hier
from fprior.metrics import lift_at_k
from fprior.sim import FounderConfig, MarketConfig, OutreachConfig, make_founder, make_market, run_outreach
RHOS = [1.0, 0.5, 0.0, -0.5]; NS = [0, 1, 5, 10, 20, 50]; SEEDS = 10
pfn = load(ROOT / "runs/pfn_v0/model.pt", "cpu"); rows = []
for seed in range(SEEDS):
    m = make_market(MarketConfig(), np.random.default_rng(seed))
    for rho in RHOS:
        for n in NS:
            rng = np.random.default_rng([seed, int(round(rho * 100)) + 1000, n])
            f = make_founder(m, FounderConfig(reliability=rho), rng)
            out = run_outreach(m, f, OutreachConfig(n_customers=n), rng)
            U = np.setdiff1d(np.arange(m.n), out.contacted)
            a = (m.segment, m.n_segments, m.graph_cov, f.segment_logit, out.contacted, out.outcome)
            sc = {"labels_flat": fit_predict(*a, replace(CPRConfig(), use_founder=False)).score,
                  "labels_hier": fit_predict_hier(*a, m.industry, m.size, replace(CPRConfig(), use_founder=False)).score,
                  "cpr_flat": fit_predict(*a, CPRConfig()).score,
                  "cpr_hier": fit_predict_hier(*a, m.industry, m.size, CPRConfig()).score}
            with torch.no_grad():
                sc["pcrfm"] = pfn_scores(pfn, m, f, out, True, "cpu")
            er = np.random.default_rng([seed, 7])
            rows.append({"seed": seed, "rho": rho, "n": n, **{k: lift_at_k(v[U], m.p[U], 100, np.random.default_rng([seed, 7])) for k, v in sc.items()}})
    print(f"seed {seed} done", flush=True)
ms = lambda v: f"{np.mean(v):+.2f} ± {np.std(v, ddof=1) / np.sqrt(len(v)):.2f}"
lines = [__doc__.strip(), ""]
for title, x, y in [("CPR-LP-hier − labels-hier (gate at ρ = 1)", "cpr_hier", "labels_hier"),
                    ("CPR-LP-hier − CPR-LP flat", "cpr_hier", "cpr_flat"), ("PC-RFM − CPR-LP-hier", "pcrfm", "cpr_hier")]:
    lines += [f"**{title}**", "", "| ρ | " + " | ".join(f"n = {n}" for n in NS) + " |", "|---|" + "---|" * len(NS)]
    for rho in RHOS:
        lines.append(f"| {rho} | " + " | ".join(ms([r[x] - r[y] for r in rows if r["rho"] == rho and r["n"] == n]) for n in NS) + " |")
    lines.append("")
md = "\n".join(lines); print(md)
(ROOT / "results/m0_hier.md").write_text(md); (ROOT / "results/m0_hier.json").write_text(json.dumps(rows))
