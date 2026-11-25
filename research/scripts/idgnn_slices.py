"""Per-slice score of the published RelBench ID-GNN on rel-hm/user-item-purchase (benchmark audit, Part 3).

ID-GNN is trained on Colab with RelBench's own example script and default hyperparameters (see
`docs/benchmark-audit.md` and the launcher in the work log); its test top-12 predictions are saved and scored
here against the same slices as `scripts/audit_relbench.py` (repeat / cf-nbr / novel / cold-user / new-item /
head-10%). Slice labels are recomputed here with the same definitions; the script checks that the slice shares
match `results/audit_relbench.json` before scoring.

PRE-REGISTERED (written and committed before ID-GNN was trained):
  I1 (reproduction): our ID-GNN test MAP@12 is within 0.5 of the published 2.81. If not, slices are still
     reported but labelled "our ID-GNN run", not "published ID-GNN".
  I2 (novel pairs): ID-GNN MAP on the novel slice is at most 2x recent popularity's novel-slice MAP
     (0.58 -> threshold 1.16). I.e. the strongest published model does not substantially beat popularity on
     the 82% of positives that are neither repeats nor co-purchase neighbours.
  I3 (hit composition): more than half of ID-GNN's correct top-12 hits are repeat or cf-nbr positives, although
     those are only 17.5% of all test positives.
Metric: RelBench MAP@12 x100, computed exactly as in audit_relbench.relbench_map.

  python scripts/idgnn_slices.py runs/idgnn_hm/test_pred.csv     # RelBench write_prediction_table output

EXTENSION (pre-registered 2026-11-25 ~14:20, before ID-GNN was trained on these tasks): the same scorer for the four
contrast tasks of `audit_relbench_ext.py` (K = task eval_k; slices as defined there):
  python scripts/idgnn_slices.py runs/idgnn_<task>/test_pred.csv rel-trial/condition-sponsor-run
  J1 (reproduction): on each task our ID-GNN test MAP is within 1.0 of the published value (11.36, 19.00, 12.72,
     10.83); tasks that fail are labelled "our ID-GNN run".
  J2 (memorisation explains rel-trial): on both rel-trial tasks, > 50% of ID-GNN's correct hits are repeat or
     cf-nbr positives.
  J3 (rel-stack learns beyond memorisation): on user-post-comment, ID-GNN's novel-slice MAP is >= 1.0 and > 2x the
     best heuristic's novel-slice MAP (heuristics: audit_relbench_ext.json).
  J4 (novel pairs are hard for ID-GNN too): on every task, ID-GNN's novel-slice MAP < half its overall MAP.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_relbench as A  # noqa: E402
from audit_relbench import DATA, TASKS, TOPN_CF, relbench_map, top_items  # noqa: E402
from audit_relbench_ext import TASKS as EXT_TASKS  # noqa: E402

TASKS = {**TASKS, **EXT_TASKS}

NAME = "rel-hm/user-item-purchase"


def slice_labels(cfg):
    test = pd.read_parquet(DATA / cfg["test"])
    T = test["timestamp"].max()
    test = test[test["timestamp"] == T].reset_index(drop=True)
    test["targets"] = test[cfg["ti"]].apply(lambda x: np.asarray(x if x is not None else [], dtype=np.int64))
    test = test[test["targets"].apply(len) > 0].reset_index(drop=True)
    h = pd.read_parquet(DATA / cfg["hist"], columns=[cfg["u"], cfg["i"], cfg["t"]])
    h = h[h[cfg["t"]] <= T]
    h.columns = ["u", "i", "t"]
    recent = top_items(h.loc[h["t"] > T - pd.Timedelta(days=cfg["recent_days"]), "i"], A.K)
    hu = h[h["u"].isin(set(test[cfg["tu"]].tolist()))]
    w = h[h["t"] > T - pd.Timedelta(days=cfg["cf_days"])][["u", "i"]].drop_duplicates()
    deg = w.groupby("u").size()
    w = w[w["u"].map(deg) <= 200]
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
    hist_pairs = set(zip(hu["u"].values, hu["i"].values))
    user_items = hu.groupby("u")["i"].apply(lambda s: set(s.values)).to_dict()
    icount = h["i"].value_counts()
    head10 = set(icount.iloc[: max(1, int(0.10 * len(icount)))].index)
    seen = set(icount.index)
    pos = []
    for u, tg in zip(test[cfg["tu"]].values, test["targets"].values):
        ui = user_items.get(u, set())
        cfn = set().union(*[nbr.get(j, set()) for j in ui]) if ui else set()
        for i in tg:
            rep = (u, i) in hist_pairs
            cf = (not rep) and (i in cfn)
            pos.append(dict(u=u, i=int(i), repeat=rep, cf_nbr=cf, novel=not rep and not cf,
                            cold_user=len(ui) == 0, new_item=i not in seen, head10=i in head10))
    return test, pd.DataFrame(pos), recent


def main():
    name = sys.argv[2] if len(sys.argv) > 2 else NAME
    cfg = TASKS[name]
    A.K = K = cfg.get("k", 12)
    test, P, recent = slice_labels(cfg)
    T = pd.read_parquet(DATA / cfg["test"])["timestamp"].max()
    pdf = pd.read_csv(sys.argv[1])
    if "timestamp" in pdf:
        pdf = pdf[pd.to_datetime(pdf["timestamp"]) == T]
    users = pdf[cfg["tu"]].values
    pcol = cfg["ti"] if cfg["ti"] in pdf else pdf.columns[-1]
    pred_raw = np.stack([np.asarray(json.loads(x), dtype=np.int64)[:K] for x in pdf[pcol].values])
    refs = json.loads((ROOT / "results/audit_relbench.json").read_text())
    if name != NAME:
        refs = json.loads((ROOT / "results/audit_relbench_ext.json").read_text())
    ref = next(r for r in refs if r["task"] == name)
    for c in ["repeat", "cf_nbr", "novel"]:
        assert abs(P[c].mean() - ref["shares"][c]) < 1e-9, f"slice share mismatch for {c}"
    row = {u: k for k, u in enumerate(users)}
    missing = [u for u in test[cfg["tu"]].values if u not in row]
    assert not missing, f"{len(missing)} test users have no ID-GNN prediction"
    preds = {"idgnn": pred_raw[[row[u] for u in test[cfg["tu"]].values]][:, :K],
             "recent_popularity": np.tile(recent, (len(test), 1))}
    targets = list(test["targets"].values)
    out = {"overall": {k: relbench_map(v, targets) for k, v in preds.items()}, "slices": {}}
    for sl in ["repeat", "cf_nbr", "novel", "cold_user", "new_item", "head10"]:
        keep = P[P[sl]].groupby("u")["i"].apply(set).to_dict()
        rows = [r for r, u in enumerate(test[cfg["tu"]].values) if u in keep]
        tsl = [np.array([i for i in targets[r] if i in keep[test.loc[r, cfg["tu"]]]]) for r in rows]
        out["slices"][sl] = {"users": len(rows), **{k: relbench_map(v[rows], tsl) for k, v in preds.items()}}
    # hit composition: which slice do ID-GNN's correct top-12 items fall in?
    lab = {(r.u, r.i): ("repeat" if r.repeat else "cf_nbr" if r.cf_nbr else "novel") for r in P.itertuples()}
    hits = {"repeat": 0, "cf_nbr": 0, "novel": 0}
    for u, p, t in zip(test[cfg["tu"]].values, preds["idgnn"], targets):
        for i in set(p.tolist()) & set(t.tolist()):
            hits[lab[(u, int(i))]] += 1
    out["hits"] = hits
    tot = max(1, sum(hits.values()))
    o, nov = out["overall"]["idgnn"], out["slices"]["novel"]
    verdict = [] if name != NAME else [
        ("I1 reproduction", abs(o - 2.81) <= 0.5, f"MAP {o:.2f} vs published 2.81"),
        ("I2 novel ≤ 2× recent popularity", nov["idgnn"] <= 2 * nov["recent_popularity"],
         f"novel MAP {nov['idgnn']:.2f} vs recent-pop {nov['recent_popularity']:.2f}"),
        ("I3 hits mostly repeat/cf-nbr", (hits["repeat"] + hits["cf_nbr"]) / tot > 0.5,
         f"{hits['repeat']} repeat, {hits['cf_nbr']} cf-nbr, {hits['novel']} novel hits"),
    ]
    if name != NAME:
        pub = A.PUBLISHED[name]["RDL (ID-GNN)"]
        hb = max(ref["overall"], key=ref["overall"].get)
        hnov = ref["slices"]["novel"].get(hb, 0.0)
        verdict = [("J1 reproduction within 1.0", abs(o - pub) <= 1.0, f"MAP {o:.2f} vs published {pub:.2f}"),
                   ("J4 novel MAP < half overall", nov["idgnn"] < 0.5 * o, f"novel {nov['idgnn']:.2f} vs overall {o:.2f}")]
        if name.startswith("rel-trial"):
            verdict.append(("J2 hits mostly repeat/cf-nbr", (hits["repeat"] + hits["cf_nbr"]) / tot > 0.5,
                            f"{hits['repeat']} repeat, {hits['cf_nbr']} cf-nbr, {hits['novel']} novel hits"))
        if name == "rel-stack/user-post-comment":
            verdict.append(("J3 novel MAP >= 1.0 and > 2x best heuristic", nov["idgnn"] >= 1.0 and nov["idgnn"] > 2 * hnov,
                            f"novel {nov['idgnn']:.2f} vs {hb} {hnov:.2f}"))
    lines = [f"# ID-GNN per-slice audit, {name}", "", f"MAP@{K} x100.", "",
             "| slice | users | ID-GNN | recent popularity |", "|---|---|---|---|",
             f"| overall | {len(test)} | {o:.2f} | {out['overall']['recent_popularity']:.2f} |"]
    lines += [f"| {k} | {v['users']} | {v['idgnn']:.2f} | {v['recent_popularity']:.2f} |" for k, v in out["slices"].items()]
    lines += ["", "## Mechanical scoring", ""] + [f"- {'HOLDS' if h else 'FAILS'} — {n}: {d}" for n, h, d in verdict]
    md = "\n".join(lines)
    print(md)
    stem = "idgnn_slices" if name == NAME else "idgnn_slices_" + name.split("/")[1]
    (ROOT / f"results/{stem}.md").write_text(md + "\n")
    (ROOT / f"results/{stem}.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
