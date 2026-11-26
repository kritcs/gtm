"""Round 4: does a graph-aware PC-RFM (v0g) close the gap to CPR-LP, and does it use the graph? (pre-registered)

Markets: fprior.sim default (S0) and the pure-interaction shift (S2), as in stress_sim.py, same seeds and founders.
Scores, paired on identical markets / founders / outreach:
  v0g          PC-RFM trained with --graph --random-seg, given the true account graph
  v0g_permA    the same model given the graph with account labels randomly permuted (degree sequence kept,
               neighbourhoods meaningless) -> tests whether v0g uses graph *structure*, not just degree
  v01          PC-RFM v0.1 (--random-seg, no graph)
  cpr_lp       CPR-LP (uses the graph covariance)
Metric: expected lift@100 on uncontacted accounts. Predictions: results/round4_preregistration.md (G3, G4).

  python scripts/eval_v0g.py --ckpt runs/pfn_v0g/model.pt --ckpt-v01 runs/pfn_v01/model.pt --out results/eval_v0g.md
"""

from __future__ import annotations

import argparse
import json
import sys
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
from fprior.sim_shift import SHIFTS, outreach_for, shift_market  # noqa: E402

RHOS = [1.0, 0.5, 0.0]
NS = [0, 5, 10, 20]
SH = ["S0_default", "S2_interaction"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-v01", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="results/eval_v0g.md")
    a = ap.parse_args()
    g, v01 = load(a.ckpt, "cpu"), load(a.ckpt_v01, "cpu")
    assert g.graph and not v01.graph
    rows = []
    for seed in range(a.seeds):
        base = make_market(MarketConfig(), np.random.default_rng(seed))
        for sh in SH:
            m = shift_market(base, sh, np.random.default_rng([seed, 500 + SHIFTS.index(sh)]))
            perm = np.random.default_rng([seed, 900]).permutation(m.n)
            mp = replace(m, A=m.A[perm][:, perm])
            for rho in RHOS:
                for n in NS:
                    rng = np.random.default_rng([seed, int(round(rho * 100)) + 1000, n])
                    f = make_founder(m, FounderConfig(reliability=rho), rng)
                    out = outreach_for(sh, m, f, OutreachConfig(n_customers=n), rng)
                    U = np.setdiff1d(np.arange(m.n), out.contacted)
                    sc = {"v0g": pfn_scores(g, m, f, out, True, "cpu"),
                          "v0g_permA": pfn_scores(g, mp, f, out, True, "cpu"),
                          "v01": pfn_scores(v01, m, f, out, True, "cpu"),
                          "cpr_lp": fit_predict(m.segment, m.n_segments, m.graph_cov, f.segment_logit,
                                                out.contacted, out.outcome, CPRConfig()).score}
                    er = np.random.default_rng([seed, 7])
                    row = {"shift": sh, "seed": seed, "rho": rho, "n": n}
                    row.update({f"lift_{k}": lift_at_k(v[U], m.p[U], 100, er) for k, v in sc.items()})
                    rows.append(row)
        print(f"seed {seed} done", flush=True)
    Path(a.out).with_suffix(".json").write_text(json.dumps(rows))

    def d(sh, x, y, rhos=RHOS, ns=NS):
        cell = {}
        for r in rows:
            if r["shift"] == sh and r["rho"] in rhos and r["n"] in ns:
                cell.setdefault(r["seed"], []).append(r[f"lift_{x}"] - r[f"lift_{y}"])
        v = np.array([np.mean(c) for c in cell.values()])   # seed-level means (seeds independent)
        return v.mean(), v.std(ddof=1) / np.sqrt(len(v))

    f2 = lambda m_s: f"{m_s[0]:+.3f} ± {m_s[1]:.3f}"  # noqa: E731
    L = [f"# Round 4: graph-aware PC-RFM (v0g) — {a.seeds} seeds", "",
         "Paired lift@100 differences, seed-level mean ± SE.", "",
         "| shift | ρ | n | v0g − v01 | v0g − v0g(permuted graph) | v0g − CPR-LP | v01 − CPR-LP |", "|---|---|---|---|---|---|---|"]
    for sh in SH:
        for rho in RHOS:
            for n in NS:
                L.append(f"| {sh} | {rho} | {n} | " + " | ".join(
                    f2(d(sh, x, y, [rho], [n])) for x, y in
                    [("v0g", "v01"), ("v0g", "v0g_permA"), ("v0g", "cpr_lp"), ("v01", "cpr_lp")]) + " |")
    g3 = d("S0_default", "v0g", "v01", ns=[5, 10, 20])
    g6 = d("S0_default", "v0g", "v0g_permA", ns=[5, 10, 20])
    g6b = d("S2_interaction", "v0g", "v0g_permA", ns=[5, 10, 20])
    sc = [("G3 v0g − v01 > 2 SE on S0 (all ρ, n ≥ 5)", g3[0] > 2 * g3[1], f2(g3)),
          ("G4 true graph − permuted graph > 2 SE on S0 (all ρ, n ≥ 5)", g6[0] > 2 * g6[1], f2(g6)),
          ("G4b same on S2 (not a prediction)", g6b[0] > 2 * g6b[1], f2(g6b))]
    L += ["", "## Mechanical scoring", ""] + [f"- {'HOLDS' if h else 'FAILS'} — {n_}: {t}" for n_, h, t in sc]
    md = "\n".join(L)
    print(md)
    Path(a.out).write_text(md + "\n")


if __name__ == "__main__":
    main()
