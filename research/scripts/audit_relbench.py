"""Benchmark audit, first cut: how memorizable are RelBench recommendation test answers?

PRE-REGISTERED (written before any slice was computed; see note on what was already known):
  A1. Most test positives are "memorizable": repeats of a past (user, item) interaction or within 2 hops
      of the user in the training interaction graph.
  A2. A repeat-past-interactions heuristic backfilled with popularity reaches >= 50% of the best MAP
      published in the RelBench paper for the same task.
  A3. On the novel-pair slice (neither repeat nor graph-neighbourhood) all heuristics are near zero.
  Known before writing this (RelBench paper, arXiv 2407.20060, Table 15): repeated links are only 3.76%
  of rel-hm and 4.73% of rel-avito test links, so the "repeat" half of A1 was already known to be false;
  the 2-hop half and A2/A3 were not known.

Definitions (history = all interactions with time <= the test timestamp T; label window is (T, T + delta]):
  repeat       (u, i) occurred in history
  cf-nbr       not repeat; i is among the top-20 co-interacted items (same-user co-occurrence in a recent
               window) of some item u interacted with — a tight "close in the graph" definition
  2-hop        i was interacted by some user who shares any item with u (literal 2-hop; measured on a
               random sample of test users because it is expensive and close to "everything" in dense graphs)
  novel        neither repeat nor cf-nbr
  cold-user    u has no history;  new-item: i has no history
  head-1/10%   i in the top 1% / 10% of items by history count
Metric: RelBench MAP@12 (relbench.metrics.map), reported x100 like the paper.
Data: RelBench v1 on the Hugging Face Hub (stanford-star/relbench-v1), CDLA-Permissive-2.0 for task
definitions; upstream licences per database.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/relbench"
K = 12
TOPN_CF = 20

# Published test MAP from the RelBench paper (arXiv 2407.20060, Table 5 / Table 8), x100.
PUBLISHED = {
    "rel-hm/user-item-purchase": {"Global Popularity": 0.30, "Past Visit": 0.89, "LightGBM": 0.38,
                                  "RDL (GraphSAGE)": 0.80, "RDL (ID-GNN)": 2.81},
    "rel-avito/user-ad-visit": {"Global Popularity": 0.00, "Past Visit": 1.95, "LightGBM": 0.06,
                                "RDL (GraphSAGE)": 0.02, "RDL (ID-GNN)": 3.66},
}
TASKS = {
    "rel-hm/user-item-purchase": dict(hist="rel-hm/db/transactions.parquet", u="customer_id", i="article_id",
                                      t="t_dat", test="rel-hm/tasks/user-item-purchase/test.parquet",
                                      tu="customer_id", ti="article_id", recent_days=7, cf_days=28),
    "rel-avito/user-ad-visit": dict(hist="rel-avito/db/VisitStream.parquet", u="UserID", i="AdID", t="ViewDate",
                                    test="rel-avito/tasks/user-ad-visit/test.parquet", tu="UserID", ti="AdID",
                                    recent_days=4, cf_days=20),
}


def relbench_map(pred: np.ndarray, targets: list) -> float:
    """Exactly relbench.metrics.map applied as in RecommendationTask.evaluate; returns x100."""
    isin = np.stack([np.isin(p, np.asarray(t)) for p, t in zip(pred, targets)])
    cnt = np.array([len(t) for t in targets])
    keep = cnt > 0
    isin, cnt = isin[keep], cnt[keep]
    prec = np.cumsum(isin, axis=1) / (np.arange(K) + 1)
    return float(((prec * isin).sum(1) / np.minimum(cnt, K)).mean() * 100)


def top_items(series: pd.Series, k: int) -> np.ndarray:
    return series.value_counts().index.values[:k]


def run_task(name: str, cfg: dict, n_2hop: int, rng: np.random.Generator) -> dict:
    t0 = time.time()
    test = pd.read_parquet(DATA / cfg["test"])
    T = test["timestamp"].max()
    test = test[test["timestamp"] == T].reset_index(drop=True)
    test["targets"] = test[cfg["ti"]].apply(lambda x: np.asarray(x if x is not None else [], dtype=np.int64))
    test = test[test["targets"].apply(len) > 0].reset_index(drop=True)
    h = pd.read_parquet(DATA / cfg["hist"], columns=[cfg["u"], cfg["i"], cfg["t"]])
    h = h[h[cfg["t"]] <= T]
    h.columns = ["u", "i", "t"]
    print(f"[{name}] T={T.date()} history {len(h):,} rows, test users {len(test):,}", flush=True)

    # --- heuristics
    glob = top_items(h["i"], K)
    recent = top_items(h.loc[h["t"] > T - pd.Timedelta(days=cfg["recent_days"]), "i"], 200)
    test_users = set(test[cfg["tu"]].tolist())
    hu = h[h["u"].isin(test_users)]
    last = hu.groupby(["u", "i"])["t"].agg(["max", "size"]).reset_index().sort_values(
        ["u", "max", "size"], ascending=[True, False, False])
    past = last.groupby("u")["i"].apply(lambda s: s.values[:K]).to_dict()

    def repeat_pred(u):
        p = list(past.get(u, []))
        for x in recent:
            if len(p) >= K:
                break
            if x not in p:
                p.append(x)
        return np.array(p[:K])

    preds = {
        "global_popularity": np.tile(glob, (len(test), 1)),
        "recent_popularity": np.tile(recent[:K], (len(test), 1)),
        "repeat_then_recent_pop": np.stack([repeat_pred(u) for u in test[cfg["tu"]]]),
    }

    # --- item co-occurrence neighbourhood (recent window)
    w = h[h["t"] > T - pd.Timedelta(days=cfg["cf_days"])][["u", "i"]].drop_duplicates()
    deg = w.groupby("u").size()
    w = w[w["u"].map(deg) <= 200]                 # cap very active users to bound cost
    uidx = {u: k for k, u in enumerate(w["u"].unique())}
    iids = w["i"].unique()
    iidx = {i: k for k, i in enumerate(iids)}
    X = sp.csr_matrix((np.ones(len(w)), (w["u"].map(uidx).values, w["i"].map(iidx).values)),
                      shape=(len(uidx), len(iids)))
    C = (X.T @ X).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()
    nbr = {}
    for r in range(C.shape[0]):
        s, e = C.indptr[r], C.indptr[r + 1]
        if e > s:
            top = np.argsort(-C.data[s:e])[:TOPN_CF]
            nbr[iids[r]] = set(iids[C.indices[s:e][top]].tolist())
    print(f"[{name}] co-occurrence: {len(iids):,} items, {C.nnz:,} pairs ({time.time() - t0:.0f}s)", flush=True)

    # --- per-positive slice labels
    hist_pairs = set(zip(hu["u"].values, hu["i"].values))
    user_items = hu.groupby("u")["i"].apply(lambda s: set(s.values)).to_dict()
    icount = h["i"].value_counts()
    r1 = icount.iloc[: max(1, int(0.01 * len(icount)))].index
    r10 = icount.iloc[: max(1, int(0.10 * len(icount)))].index
    head1, head10, seen_items = set(r1), set(r10), set(icount.index)
    pos = []
    for u, tg in zip(test[cfg["tu"]].values, test["targets"].values):
        ui = user_items.get(u, set())
        cfn = set().union(*[nbr.get(j, set()) for j in ui]) if ui else set()
        for i in tg:
            rep = (u, i) in hist_pairs
            cf = (not rep) and (i in cfn)
            pos.append(dict(u=u, i=int(i), repeat=rep, cf_nbr=cf, novel=not rep and not cf,
                            cold_user=len(ui) == 0, new_item=i not in seen_items,
                            head1=i in head1, head10=i in head10))
    P = pd.DataFrame(pos)

    # --- literal 2-hop on a sample of test users (users x items matrix over full history)
    hu_all = h[["u", "i"]].drop_duplicates()
    au = {u: k for k, u in enumerate(hu_all["u"].unique())}
    ai = {i: k for k, i in enumerate(hu_all["i"].unique())}
    XA = sp.csr_matrix((np.ones(len(hu_all), dtype=np.float32),
                        (hu_all["u"].map(au).values, hu_all["i"].map(ai).values)), shape=(len(au), len(ai)))
    XAT = XA.T.tocsr()
    sample = rng.choice(len(test), size=min(n_2hop, len(test)), replace=False)
    two_hop_hits, two_hop_tot, reach_sizes = 0, 0, []
    for r in sample:
        u = test.loc[r, cfg["tu"]]
        if u not in au:
            continue
        users = (XA @ XA[au[u]].T).nonzero()[0]                 # users sharing an item with u
        items = np.unique(XAT[:, users].nonzero()[0]) if len(users) else np.array([], int)  # their items
        reach_sizes.append(len(items) / len(ai))
        s = set(items.tolist())
        for i in test.loc[r, "targets"]:
            two_hop_tot += 1
            two_hop_hits += (ai.get(int(i), -1) in s) and ((u, i) not in hist_pairs)
    two_hop = dict(sample_users=len(sample), share_positives_2hop_not_repeat=two_hop_hits / max(two_hop_tot, 1),
                   median_share_of_catalogue_reachable=float(np.median(reach_sizes)) if reach_sizes else None)

    # --- scores overall and per slice
    targets = list(test["targets"].values)
    overall = {k: relbench_map(v, targets) for k, v in preds.items()}
    slices = {}
    for sl in ["repeat", "cf_nbr", "novel", "cold_user", "new_item", "head10"]:
        keep = P[P[sl]].groupby("u")["i"].apply(set).to_dict()
        rows = [r for r, u in enumerate(test[cfg["tu"]].values) if u in keep]
        if not rows:
            slices[sl] = {"users": 0}
            continue
        tsl = [np.array([i for i in targets[r] if i in keep[test.loc[r, cfg["tu"]]]]) for r in rows]
        slices[sl] = {"users": len(rows), **{k: relbench_map(v[rows], tsl) for k, v in preds.items()}}
    shares = {c: float(P[c].mean()) for c in ["repeat", "cf_nbr", "novel", "cold_user", "new_item", "head1", "head10"]}
    best = max(PUBLISHED[name].values())
    out = dict(task=name, T=str(T.date()), test_users=len(test), positives=len(P), shares=shares,
               two_hop=two_hop, overall=overall, slices=slices, published=PUBLISHED[name],
               best_published=best,
               repeat_heuristic_share_of_best=overall["repeat_then_recent_pop"] / best,
               seconds=round(time.time() - t0, 1))
    print(json.dumps(out, indent=1, default=str), flush=True)
    return out


def main():
    rng = np.random.default_rng(0)
    n_2hop = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    res = [run_task(n, c, n_2hop, rng) for n, c in TASKS.items()]
    (ROOT / "results/audit_relbench.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
