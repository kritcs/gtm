"""Sim-to-sim stress test: PC-RFM v0 (simulator-trained) vs CPR-LP on deliberately shifted worlds.

Pre-registered (written before the full run):
  (a) Under S1 (heavy-tailed founder errors) both models lose lift vs S0, but PC-RFM loses less
      than CPR-LP (mean over rho in {1, 0.5}, n >= 5).
  (b) Under S6 (base rate 0.01 / 0.30, outside PC-RFM's training range 0.02-0.2) PC-RFM degrades
      more than CPR-LP relative to S0 (mean of PC-RFM − CPR-LP drops vs S0).
  (c) PC-RFM − CPR-LP with bad founders (rho <= 0, n >= 10) stays positive in at least 4 of the
      6 shift families (S1, S2, S3, S4, S5, S6 — S6 counts as positive only if both S6a and S6b are).
Shifts are defined in fprior/sim_shift.py. Metric: expected lift@100 on uncontacted accounts.
Lift is relative to the market's own base rate, so it is comparable across base rates only roughly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_pfn import load, pfn_scores  # noqa: E402
from fprior.cprlp import CPRConfig, fit_predict  # noqa: E402
from fprior.metrics import lift_at_k  # noqa: E402
from fprior.sim import FounderConfig, MarketConfig, OutreachConfig, make_founder, make_market  # noqa: E402
from fprior.sim_shift import SHIFTS, heavy_tail_founder, outreach_for, shift_market  # noqa: E402

RHOS = [1.0, 0.5, 0.0, -0.5]
NS = [0, 1, 5, 10, 20, 50]
METHODS = ["prior_only", "labels_only", "cpr_lp", "pcrfm", "pcrfm_nofounder"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--shifts", default=",".join(SHIFTS))
    ap.add_argument("--out", default="results/stress_sim.md")
    a = ap.parse_args()
    shifts = a.shifts.split(",")
    pf = load(ROOT / "runs/pfn_v0/model.pt", "cpu")
    pnf = load(ROOT / "runs/pfn_v0_nf/model.pt", "cpu")
    rows, t0 = [], time.time()
    for seed in range(a.seeds):
        base = make_market(MarketConfig(), np.random.default_rng(seed))
        for si, sh in enumerate(shifts):
            market = shift_market(base, sh, np.random.default_rng([seed, 500 + SHIFTS.index(sh)]))
            for rho in RHOS:
                for n in NS:
                    rng = np.random.default_rng([seed, int(round(rho * 100)) + 1000, n])
                    founder = make_founder(market, FounderConfig(reliability=rho), rng)
                    if sh == "S1_heavy_founder":
                        founder = heavy_tail_founder(market, founder,
                                                     np.random.default_rng([seed, 600, int(round(rho * 100)) + 100]))
                    out = outreach_for(sh, market, founder, OutreachConfig(n_customers=n), rng)
                    U = np.setdiff1d(np.arange(market.n), out.contacted)
                    args = (market.segment, market.n_segments, market.graph_cov, founder.segment_logit,
                            out.contacted, out.outcome)
                    sc = {"prior_only": founder.segment_logit[market.segment],
                          "labels_only": fit_predict(*args, replace(CPRConfig(), use_founder=False)).score,
                          "cpr_lp": fit_predict(*args, CPRConfig()).score,
                          "pcrfm": pfn_scores(pf, market, founder, out, True, "cpu"),
                          "pcrfm_nofounder": pfn_scores(pnf, market, founder, out, False, "cpu")}
                    er = np.random.default_rng([seed, 7])
                    row = {"shift": sh, "seed": seed, "rho": rho, "n": n, "contacts": int(len(out.contacted)),
                           "base": float(market.p.mean())}
                    for k in METHODS:
                        row[f"lift_{k}"] = lift_at_k(sc[k][U], market.p[U], 100, er)
                    rows.append(row)
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
        (ROOT / a.out).with_suffix(".json").write_text(json.dumps(rows))

    def sel(sh, rhos, ns):
        return [r for r in rows if r["shift"] == sh and r["rho"] in rhos and r["n"] in ns]

    def seed_mean_diff(cells, a_, b_):
        """mean over cells within seed, then mean ± SE over seeds"""
        seeds = sorted({r["seed"] for r in cells})
        v = np.array([np.mean([r[f"lift_{a_}"] - r[f"lift_{b_}"] for r in cells if r["seed"] == s]) for s in seeds])
        return v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")

    f = lambda t: f"{t[0]:+.2f} ± {t[1]:.2f}"  # noqa: E731
    good, bad = [1.0, 0.5], [0.0, -0.5]
    small, large = [0, 1, 5], [10, 20, 50]
    lines = [__doc__.strip(), "", f"## Summary ({a.seeds} seeds; mean over cells within seed, ± SE over seeds)", "",
             "PC-RFM − CPR-LP by founder quality and label count; worst cell of each model vs labels-only "
             "(over all ρ, n).", "",
             "| shift | base | good founder, n≤5 | good, n≥10 | bad founder, n≤5 | bad, n≥10 | "
             "CPR-LP − labels worst cell | PC-RFM − labels worst cell |",
             "|---|---|---|---|---|---|---|---|"]
    for sh in shifts:
        cells = [r for r in rows if r["shift"] == sh]
        if not cells:
            continue

        def worst(m):
            best = None
            for rho in RHOS:
                for n in NS:
                    c = [r[f"lift_{m}"] - r["lift_labels_only"] for r in cells if r["rho"] == rho and r["n"] == n]
                    v = np.mean(c)
                    if best is None or v < best[0]:
                        best = (v, rho, n)
            return f"{best[0]:+.2f} (ρ={best[1]}, n={best[2]})"
        lines.append(f"| {sh} | {np.mean([r['base'] for r in cells]):.3f} | "
                     f"{f(seed_mean_diff(sel(sh, good, small), 'pcrfm', 'cpr_lp'))} | "
                     f"{f(seed_mean_diff(sel(sh, good, large), 'pcrfm', 'cpr_lp'))} | "
                     f"{f(seed_mean_diff(sel(sh, bad, small), 'pcrfm', 'cpr_lp'))} | "
                     f"{f(seed_mean_diff(sel(sh, bad, large), 'pcrfm', 'cpr_lp'))} | {worst('cpr_lp')} | {worst('pcrfm')} |")

    # degradation vs S0 (for predictions a, b): mean lift of each model, rho in {1, .5}, n >= 5
    lines += ["", "## Mean lift by shift (ρ ∈ {1, 0.5}, n ≥ 5) — for predictions (a)/(b)", "",
              "| shift | labels-only | CPR-LP | PC-RFM | PC-RFM − CPR-LP |", "|---|---|---|---|---|"]
    for sh in shifts:
        c = sel(sh, good, [5, 10, 20, 50])
        if not c:
            continue
        mm = lambda k: np.mean([r[f"lift_{k}"] for r in c])  # noqa: E731
        lines.append(f"| {sh} | {mm('labels_only'):.2f} | {mm('cpr_lp'):.2f} | {mm('pcrfm'):.2f} | "
                     f"{f(seed_mean_diff(c, 'pcrfm', 'cpr_lp'))} |")

    # full tables
    ms = lambda v: f"{np.mean(v):.2f} ± {np.std(v, ddof=1) / np.sqrt(len(v)):.2f}"  # noqa: E731
    lines += ["", "## Full tables (lift@100, mean ± SE over seeds)", ""]
    for sh in shifts:
        lines += [f"### {sh}", ""]
        for rho in RHOS:
            hdr = ["n", "contacts", *METHODS, "PC-RFM − CPR-LP", "PC-RFM − no-founder", "CPR − labels"]
            lines += [f"**ρ = {rho}**", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
            for n in NS:
                c = [r for r in rows if r["shift"] == sh and r["rho"] == rho and r["n"] == n]
                if not c:
                    continue
                d = lambda x, y: ms([r[f"lift_{x}"] - r[f"lift_{y}"] for r in c])  # noqa: E731
                vals = [f"{np.mean([r['contacts'] for r in c]):.0f}"] + [ms([r[f"lift_{k}"] for r in c]) for k in METHODS]
                vals += [d("pcrfm", "cpr_lp"), d("pcrfm", "pcrfm_nofounder"), d("cpr_lp", "labels_only")]
                lines.append(f"| {n} | " + " | ".join(vals) + " |")
            lines.append("")
    md = "\n".join(lines)
    (ROOT / a.out).write_text(md)
    print(md[:6000])


if __name__ == "__main__":
    main()
