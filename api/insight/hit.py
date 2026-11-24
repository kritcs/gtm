"""The hit: a pattern among the founder's customers they never mentioned.

For every feature seen in a customer's six months before signing:
  k      = customers with the feature in their window
  p_i    = base rate in customer i's matched group (same size bucket and stage,
           same calendar window, active companies only)
  p      = P(X >= k), X ~ PoissonBinomial(p_1..p_n)
  q      = Benjamini-Hochberg over every feature seen (the whole family scanned)
Shown only if: support >= min(3, n), shrunk lift >= 5, q < 0.05, and the
feature isn't something the founder already said.
"""
from __future__ import annotations

import re

import numpy as np

from ..cards import Card, receipt
from ..market.extract import FEATURE_LABELS
from ..market.graph import MarketGraph
from ..market.simulate import month_label
from .stats import benjamini_hochberg, poisson_binomial_tail, shrunk_lift

WINDOW = 6
SYNONYMS = {
    "role:revops": ["revops", "revenue operations", "rev ops"],
    "role:controller": ["controller"], "role:finops": ["finops"], "role:sales_ops": ["sales ops"],
    "attr:recently_raised": ["raised", "funding", "series"],
    "attr:usage_based_pricing": ["usage-based", "usage based"],
    "attr:multi_entity": ["entities", "entity"], "attr:intl_expansion": ["europe", "expanding", "international"],
    "attr:remote_first": ["remote"],
}


def feature_label(key: str) -> str:
    if key.startswith("stack:"):
        return f"adopted {key[6:]}"
    return FEATURE_LABELS.get(key, key)


def mentioned(key: str, transcript: str) -> bool:
    t = transcript.lower()
    words = SYNONYMS.get(key, [])
    if key.startswith("stack:"):
        words = [key[6:].lower()]
    return any(re.search(r"\b" + re.escape(w) + r"\b", t) for w in words)


def find_hit(g: MarketGraph, customers: list[tuple[int, int]], transcript: str, *,
             alpha: float = 0.05, min_lift: float = 5.0) -> tuple[Card, list[dict]]:
    w = g.world
    n = len(customers)
    if n == 0:
        return _no_hit("No customers named, so no shared pattern to test."), []
    windows = {cid: g.feature_window(cid, s - WINDOW, s) for cid, s in customers}
    cust_ids = {cid for cid, _ in customers}

    # matched comparison groups, one per customer
    groups = {}
    for cid, s in customers:
        c = w.companies[cid]
        members = [o.id for o in w.companies if o.id not in cust_ids and o.size == c.size
                   and o.stage == c.stage and g.active(o.id, s - WINDOW, s)]
        groups[cid] = (s, members, [g.feature_window(o, s - WINDOW, s) for o in members])

    feats = sorted({f for win in windows.values() for f in win})
    rows = []
    for f in feats:
        k = sum(f in windows[cid] for cid, _ in customers)
        ps = []
        for cid, (s, members, mwins) in groups.items():
            x = sum(f in mw for mw in mwins)
            ps.append((x + 0.5) / (len(members) + 1))  # Jeffreys-style smoothing
        p = poisson_binomial_tail(ps, k)
        base = float(np.mean(ps))
        rows.append({"feature": f, "label": feature_label(f), "k": k, "n": n, "base_rate": base,
                     "group_sizes": [len(m) for _, m, _ in groups.values()], "p_chance": p,
                     "lift": (k / n) / base, "shrunk_lift": shrunk_lift(k, n, base),
                     "mentioned": mentioned(f, transcript)})
    qs = benjamini_hochberg([r["p_chance"] for r in rows])
    for r, q in zip(rows, qs):
        r["q_value"] = q
        r["passes"] = (r["k"] >= min(3, n) and r["shrunk_lift"] >= min_lift and q < alpha and not r["mentioned"])
    rows.sort(key=lambda r: (not r["passes"], r["q_value"], -r["shrunk_lift"]))

    best = rows[0] if rows and rows[0]["passes"] else None
    if best is None:
        return _no_hit("No shared customer trait passed the significance test."), rows

    f = best["feature"]
    recs = []
    for cid, s in customers:
        if f in windows[cid]:
            m, pid, span = windows[cid][f]
            post = g.posts[pid]
            recs.append(receipt(post, post.text.split("\n", 1)[0], w.companies[cid].name))
    k, n = best["k"], best["n"]
    front = f"{k} of {n} customers {best['label']} in the 6 months before buying. Base rate: {best['base_rate']:.0%}."
    q = best["q_value"]
    if q > 0:
        n = 1 / q
        mag = 10 ** (len(str(int(n))) - 2)
        back = f"Odds this is luck: about 1 in {round(n / mag) * mag:,.0f}, after testing {len(rows)} traits."
    else:
        back = f"Odds this is luck: effectively zero, after testing {len(rows)} traits."
    clarity = float(np.clip(-np.log10(max(best["q_value"], 1e-12)) / 3, 0.25, 1.0))
    card = Card(id="hit", kind="hit", front_text=front, back_text=back, clarity=clarity, receipts=recs,
                stats={k_: best[k_] for k_ in ("feature", "k", "n", "base_rate", "p_chance", "q_value",
                                                "lift", "shrunk_lift", "group_sizes")}
                | {"features_tested": len(rows), "window_months": WINDOW,
                   "signing": {w.companies[c].name: month_label(s) for c, s in customers},
                   "assumption": "independence across customers conditional on matching (size, stage, window)"})
    return card, rows


def _no_hit(text: str) -> Card:
    return Card(id="hit", kind="hit", front_text=text, clarity=0.3,
                back_text="Nothing passed: support >= 3, shrunk lift >= 5, and BH q < 0.05.")
