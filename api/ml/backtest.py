"""Backtest and ablation: does "market + founder" beat either alone?

Leakage-safe by construction:
- Everything is trained on a snapshot that drops every post, edge and feature
  dated after ASOF (2024-12): the text model, LightGCN, the ridge map.
- The product (Ledgerlight) is held out of graph training entirely; its
  embedding comes from its description plus its pre-cutoff customers (the
  simulated founder), exactly as in the live reading.
- Candidates: companies active both before and after ASOF that had not
  adopted by ASOF. Positives: those that adopt after ASOF. Only new adopters count.

Baselines:
- random (expected), popularity (size of the company's known stack),
- description match: cosine between the product description and each
  company's text. This stands in for "an LLM ranking the same candidates from
  descriptions"; a real LLM baseline must only be scored on months after its
  training cutoff, or it remembers rather than forecasts.

Run:  uv run python -m api.ml.backtest
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from ..insight.ranking import pain_match
from ..insight.stats import bootstrap_recall_ci, pct_rank
from ..market.graph import MarketGraph, build_graph, snapshot
from ..market.simulate import PRODUCT_DESCRIPTION, World, generate, month_index, month_label
from .market import MarketModel

ASOF = month_index("2024-12")
K = 50
FOUNDER_PAIN = ("Month-end close takes weeks of spreadsheet work reconciling billing invoices against the CRM by hand. "
                "Nobody trusts the revenue numbers.")
CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "backtest.json"


def assert_no_leakage(snap: MarketGraph, asof: int):
    assert all(p.month <= asof for p in snap.posts.values()), "post after cutoff"
    assert all(m <= asof for (m, _) in snap.adopted.values()), "edge after cutoff"
    assert all(m <= asof for fs in snap.features.values() for (_, m, _, _) in fs), "feature after cutoff"


def run(world: World | None = None, full: MarketGraph | None = None, k: int = K, lambda0: float = 1.0,
        tau: float = 0.25, verbose: bool = True) -> dict:
    t0 = time.time()
    world = world or generate()
    full = full or build_graph(world)
    snap = snapshot(full, ASOF)
    assert_no_leakage(snap, ASOF)
    model = MarketModel(snap)
    pid = world.product_tool_id

    early = sorted(snap.adopters(pid))
    later = {c: m for c, m in full.adopters(pid).items() if m > ASOF}
    cand = np.array([c.id for c in world.companies
                     if c.id not in set(early) and snap.active(c.id, 0, ASOF + 1)
                     and full.active(c.id, ASOF + 1, 10**6)])
    pos = np.isin(cand, list(later))
    n_pos = int(pos.sum())

    # scores on the candidate set -------------------------------------------
    y_hat, meta = model.market_scores(PRODUCT_DESCRIPTION, early)
    fvec = model.text_vec(FOUNDER_PAIN + " " + PRODUCT_DESCRIPTION)
    pain, _ = pain_match(model, fvec)
    founder = 0.7 * pain + 0.3 * (model.company_text @ model.lsa.transform([FOUNDER_PAIN])[0])
    dist = model.experience_distance(early)
    lam = lambda0 * np.exp(-dist / tau)
    pop = np.array([len(snap.company_tools(c)) for c in range(len(world.companies))], float)
    desc = model.company_text @ model.lsa.transform([PRODUCT_DESCRIPTION])[0]

    def both_scores(idx):
        return pct_rank(y_hat[idx]) + lam[idx] * pct_rank(founder[idx])

    methods = {
        "popularity": pop[cand],
        "description match (LLM stand-in)": desc[cand],
        "market model alone": y_hat[cand],
        "founder alone": founder[cand],
        "market + founder": both_scores(cand),
    }
    rows = {}
    for name, sc in methods.items():
        top = np.argsort(-sc, kind="stable")[:k]
        in_top = np.zeros(len(cand), bool)
        in_top[top] = True
        hits = in_top[pos]
        lo, hi = bootstrap_recall_ci(hits, n_pos, k)
        rows[name] = {"recall": float(hits.mean()) if n_pos else 0.0, "ci95": [lo, hi], "hits": int(hits.sum())}
    rows = {"random (expected)": {"recall": k / len(cand), "ci95": [k / len(cand)] * 2, "hits": None}} | rows

    both, mk, fd = rows["market + founder"], rows["market model alone"], rows["founder alone"]
    out = {
        "simulated": True, "asof": month_label(ASOF), "test_window": f"{month_label(ASOF + 1)}..",
        "k": k, "candidates": int(len(cand)), "positives": n_pos, "early_customers": len(early),
        "lambda0": lambda0, "tau": tau, "methods": rows,
        "combination_beats_both_point": both["recall"] > max(mk["recall"], fd["recall"]),
        "combination_beats_both_ci": both["ci95"][0] > max(mk["ci95"][1], fd["ci95"][1]),
        "fold_in": meta, "train": model.info, "seconds": round(time.time() - t0, 1),
        "leakage_checks": ["posts <= asof", "edges <= asof", "features <= asof",
                           "text model fit on snapshot only", "product held out of graph training"],
    }
    if verbose:
        print(f"Backtest as of {out['asof']}: {out['candidates']} candidates, {n_pos} new adopters, Recall@{k}")
        for name, r in rows.items():
            print(f"  {name:34s} {r['recall']:.3f}   95% CI [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]")
        print("  combination beats both parts:", "yes (CI)" if out["combination_beats_both_ci"]
              else "point estimate only" if out["combination_beats_both_point"] else "no")
    return out


def cached(world=None, full=None) -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    out = run(world, full, verbose=False)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    res = run()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(res, indent=2))
    print(f"wrote {CACHE}", file=sys.stderr)
