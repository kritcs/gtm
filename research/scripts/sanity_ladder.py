"""M0 sanity ladder, steps 1-2 (CLAUDE.md, failure mode 2).

Step 1: an oracle founder (rho = 1) must make CPR-LP beat labels-only at n <= 20.
        If not, stop and debug before anything else.
Step 2: sweep founder reliability rho to find the break-even point, and check that
        learned trust limits the harm from wrong founders (vs fixed trust).

Usage: python scripts/sanity_ladder.py [--seeds 10] [--quick]
Writes results/m0_sanity.json and results/m0_sanity.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fprior.cprlp import CPRConfig, fit_predict  # noqa: E402
from fprior.metrics import expected_auc, lift_at_k  # noqa: E402
from fprior.sim import (FounderConfig, MarketConfig, OutreachConfig, make_founder,  # noqa: E402
                        make_market, run_outreach)

RHOS = [1.0, 0.75, 0.5, 0.25, 0.0, -0.5]
NS = [0, 1, 5, 10, 20, 50]
METHODS = ["random", "popularity", "prior_only", "fixed_trust", "labels_only", "cpr_lp", "oracle_p"]
TOP_K = 100


def run_cell(market, rho, n, seed):
    rng = np.random.default_rng([seed, int(round(rho * 100)) + 1000, n])
    founder = make_founder(market, FounderConfig(reliability=rho), rng)
    out = run_outreach(market, founder, OutreachConfig(n_customers=n), rng)
    U = np.setdiff1d(np.arange(market.n), out.contacted)
    args = (market.segment, market.n_segments, market.graph_cov, founder.segment_logit,
            out.contacted, out.outcome)
    base = CPRConfig()
    fits = {
        "fixed_trust": fit_predict(*args, replace(base, learn_trust=False, omega_grid=None)),
        "labels_only": fit_predict(*args, replace(base, use_founder=False)),
        "cpr_lp": fit_predict(*args, base),
    }
    scores = {
        "random": np.zeros(market.n),
        "popularity": market.degree.astype(float),
        "prior_only": founder.segment_logit[market.segment],
        "oracle_p": market.true_logit,
        **{k: v.score for k, v in fits.items()},
    }
    p_u = market.p[U]
    row = {"seed": seed, "rho": rho, "n": n, "contacts": int(len(out.contacted)),
           "customers": out.n_customers, "trust_b": float(fits["cpr_lp"].trust[1]),
           "omega_cpr": fits["cpr_lp"].omega, "omega_labels": fits["labels_only"].omega}
    eval_rng = np.random.default_rng([seed, 7])
    for m in METHODS:
        row[f"lift_{m}"] = lift_at_k(scores[m][U], p_u, TOP_K, eval_rng)
        row[f"auc_{m}"] = expected_auc(scores[m][U], p_u)
    return row


def summarize(rows, metric="lift"):
    out = {}
    for rho in RHOS:
        for n in NS:
            cell = [r for r in rows if r["rho"] == rho and r["n"] == n]
            if not cell:
                continue
            s = {m: (float(np.mean([r[f"{metric}_{m}"] for r in cell])),
                     float(np.std([r[f"{metric}_{m}"] for r in cell], ddof=1) / np.sqrt(len(cell))))
                 for m in METHODS}
            for other in ("labels_only", "fixed_trust", "prior_only"):
                d = np.array([r[f"{metric}_cpr_lp"] - r[f"{metric}_{other}"] for r in cell])
                s[f"cpr-{other}"] = (float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))))
            s["trust_b"] = (float(np.mean([r["trust_b"] for r in cell])),
                            float(np.std([r["trust_b"] for r in cell], ddof=1) / np.sqrt(len(cell))))
            s["omega"] = float(np.median([r["omega_cpr"] for r in cell]))
            s["contacts"] = float(np.mean([r["contacts"] for r in cell]))
            out[(rho, n)] = s
    return out


def gate(summary):
    """Step 1 passes if, for rho = 1 and every n in {0,1,5,10,20}, CPR-LP beats labels-only
    on lift@100 by more than 2 standard errors (paired over seeds)."""
    checks = []
    for n in [0, 1, 5, 10, 20]:
        d, se = summary[(1.0, n)]["cpr-labels_only"]
        checks.append({"n": n, "diff": d, "se": se, "pass": d > 2 * se})
    return all(c["pass"] for c in checks), checks


def to_markdown(summary, passed, checks, n_seeds, cfg):
    f = lambda t: f"{t[0]:.2f} ± {t[1]:.2f}"  # noqa: E731
    lines = [
        "# M0 sanity ladder (simulated)",
        "",
        f"{n_seeds} seeds; one market per seed, {cfg.n_accounts} accounts, base rate {cfg.base_rate}.",
        f"Metric: expected lift@{TOP_K} on uncontacted accounts (1.0 = random); mean ± SE over seeds.",
        "`n` = customers found before scoring; outreach is steered by the founder's beliefs.",
        "",
        f"## Step 1: oracle founder must help — **{'PASS' if passed else 'FAIL'}**",
        "",
        "| n | CPR-LP − labels-only | pass (> 2 SE) |",
        "|---|---|---|",
    ]
    lines += [f"| {c['n']} | {c['diff']:.2f} ± {c['se']:.2f} | {'yes' if c['pass'] else 'no'} |" for c in checks]
    lines += ["", "## Step 2: sweep founder reliability ρ", ""]
    for rho in RHOS:
        lines += [f"### ρ = {rho}", "",
                  "| n | contacts | prior only | fixed trust | labels only | **CPR-LP** | CPR − labels | CPR − fixed | learned b | ω (median) | popularity | oracle p |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for n in NS:
            s = summary[(rho, n)]
            lines.append(f"| {n} | {s['contacts']:.0f} | {f(s['prior_only'])} | {f(s['fixed_trust'])} | "
                         f"{f(s['labels_only'])} | **{f(s['cpr_lp'])}** | {f(s['cpr-labels_only'])} | "
                         f"{f(s['cpr-fixed_trust'])} | {f(s['trust_b'])} | {s['omega']:.2f} | {f(s['popularity'])} | {f(s['oracle_p'])} |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="small market, 3 seeds")
    a = ap.parse_args()
    cfg = MarketConfig(n_accounts=1000) if a.quick else MarketConfig()
    n_seeds = 3 if a.quick else a.seeds
    rows = []
    t0 = time.time()
    for seed in range(n_seeds):
        market = make_market(cfg, np.random.default_rng(seed))
        for rho in RHOS:
            for n in NS:
                rows.append(run_cell(market, rho, n, seed))
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    summary = summarize(rows)
    passed, checks = gate(summary)
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    suffix = "_quick" if a.quick else ""
    (out_dir / f"m0_sanity{suffix}.json").write_text(json.dumps(
        {"market_config": asdict(cfg), "cpr_config": asdict(CPRConfig()), "rows": rows,
         "gate": {"passed": passed, "checks": checks}}, indent=1))
    md = to_markdown(summary, passed, checks, n_seeds, cfg)
    (out_dir / f"m0_sanity{suffix}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
