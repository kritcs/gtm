"""Baseline sweep on the relational simulator (fprior.sim_rel) over novelty e x rho x n.

Pre-registered expectation (written before running):
  1. Relational signal (similar-product adopters, co-adoption graph) helps most at low novelty.
  2. Founder priors help most at high novelty.
  3. At e = 1, graph-based methods gain ~nothing over segment-level methods.

Same founder / outreach RNG conventions as scripts/sanity_ladder.py.

Usage: python scripts/rel_baselines.py [--seeds 10]
Writes results/rel_baselines.md and results/rel_baselines.json.
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
from fprior.sim import FounderConfig, OutreachConfig, make_founder, run_outreach  # noqa: E402
from fprior.sim_rel import RelMarketConfig, make_rel_market, similar_adopter_score  # noqa: E402

NOVELTY = [0.0, 0.5, 1.0]
RHOS = [1.0, 0.5, 0.0, -0.5]
NS = [0, 1, 5, 10, 20, 50]
METHODS = ["random", "popularity", "prior_only", "similar_adopters", "prior_plus_adopters",
           "labels_only", "cpr_lp_nograph", "cpr_lp", "oracle_p"]
DIFFS = [("cpr_lp", "labels_only"), ("cpr_lp", "cpr_lp_nograph"), ("cpr_lp", "similar_adopters"),
         ("cpr_lp", "prior_only")]
TOP_K = 100

PREREG = """## Pre-registered expectation (written before running)

1. Relational signal (similar-product adopters, co-adoption graph) helps most at low novelty.
2. Founder priors help most at high novelty.
3. At e = 1, graph-based methods gain ~nothing over segment-level methods.
"""


def z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def run_cell(market, adopters, rho, n, seed):
    rng = np.random.default_rng([seed, int(round(rho * 100)) + 1000, n])
    founder = make_founder(market, FounderConfig(reliability=rho), rng)
    out = run_outreach(market, founder, OutreachConfig(n_customers=n), rng)
    U = np.setdiff1d(np.arange(market.n), out.contacted)
    args = (market.segment, market.n_segments, market.graph_cov, founder.segment_logit,
            out.contacted, out.outcome)
    base = CPRConfig()
    fits = {
        "labels_only": fit_predict(*args, replace(base, use_founder=False)),
        "cpr_lp_nograph": fit_predict(*args, replace(base, graph_sd=0.0)),
        "cpr_lp": fit_predict(*args, base),
    }
    prior = founder.segment_logit[market.segment]
    scores = {
        "random": np.zeros(market.n),
        "popularity": market.degree.astype(float),
        "prior_only": prior,
        "similar_adopters": adopters,
        "prior_plus_adopters": z(prior) + z(adopters),
        "oracle_p": market.true_logit,
        **{k: v.score for k, v in fits.items()},
    }
    p_u = market.p[U]
    row = {"seed": seed, "rho": rho, "n": n, "contacts": int(len(out.contacted)),
           "trust_b": float(fits["cpr_lp"].trust[1])}
    eval_rng = np.random.default_rng([seed, 7])
    for m in METHODS:
        row[f"lift_{m}"] = lift_at_k(scores[m][U], p_u, TOP_K, eval_rng)
        row[f"auc_{m}"] = expected_auc(scores[m][U], p_u)
    return row


def ms(v):
    v = np.asarray(v, dtype=float)
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
    return float(v.mean()), float(se)


def table(rows, e, rho, metric):
    f = lambda t: f"{t[0]:.2f} ± {t[1]:.2f}"  # noqa: E731
    short = {"popularity": "popul.", "prior_only": "prior", "similar_adopters": "sim-adopt",
             "prior_plus_adopters": "prior+adopt", "labels_only": "labels", "cpr_lp_nograph": "CPR no-graph",
             "cpr_lp": "**CPR-LP**", "oracle_p": "oracle p", "random": "random"}
    hdr = ["n", "contacts", *[short[m] for m in METHODS if m != "random"],
           *[f"{short[a]} − {short[b]}".replace("**", "") for a, b in DIFFS], "learned b"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for n in NS:
        cell = [r for r in rows if r["e"] == e and r["rho"] == rho and r["n"] == n]
        vals = [f"{np.mean([r['contacts'] for r in cell]):.0f}"]
        vals += [f(ms([r[f"{metric}_{m}"] for r in cell])) for m in METHODS if m != "random"]
        vals += [f(ms([r[f"{metric}_{a}"] - r[f"{metric}_{b}"] for r in cell])) for a, b in DIFFS]
        vals.append(f(ms([r["trust_b"] for r in cell])))
        lines.append(f"| {n} | " + " | ".join(vals) + " |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-accounts", type=int, default=3000)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    rows, t0 = [], time.time()
    for e in NOVELTY:
        cfg = RelMarketConfig(n_accounts=a.n_accounts, novelty=e)
        for seed in range(a.seeds):
            market = make_rel_market(cfg, np.random.default_rng(seed))
            adopters = similar_adopter_score(market)
            for rho in RHOS:
                for n in NS:
                    r = run_cell(market, adopters, rho, n, seed)
                    r["e"] = e
                    rows.append(r)
            print(f"e={e} seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"rel_baselines{a.suffix}.json").write_text(json.dumps(
        {"config": asdict(RelMarketConfig(n_accounts=a.n_accounts)), "cpr": asdict(CPRConfig()),
         "seeds": a.seeds, "rows": rows}, indent=1))
    lines = ["# Baselines on the relational simulator (sim_rel)", "",
             f"{a.seeds} seeds per novelty level; {a.n_accounts} accounts; RelMarketConfig defaults otherwise. "
             f"Expected lift@{TOP_K} on uncontacted accounts (1.0 = random), mean ± SE over seeds; "
             "differences are paired. Runtime "
             f"{time.time() - t0:.0f}s.", "", PREREG]
    for e in NOVELTY:
        lines += [f"## Novelty e = {e}", ""]
        for rho in RHOS:
            lines += [f"### e = {e}, ρ = {rho} (lift@{TOP_K})", "", *table(rows, e, rho, "lift"), ""]
    lines += ["## Expected AUC (same grid)", ""]
    for e in NOVELTY:
        for rho in RHOS:
            lines += [f"### e = {e}, ρ = {rho} (AUC)", "", *table(rows, e, rho, "auc"), ""]
    (out_dir / f"rel_baselines{a.suffix}.md").write_text("\n".join(lines))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
