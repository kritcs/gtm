"""Benchmark audit, contrast tasks: rel-trial and rel-stack recommendation (same slices as audit_relbench.py).

Why: on rel-hm / rel-avito a no-learning repeat + popularity heuristic reached 93% / 88% of the best published MAP
and novel pairs (82–93% of positives) were scored ~0 by everything. Is that specific to those two tasks or general?
rel-trial has many repeated links; rel-stack has few but ID-GNN is far ahead of every baseline.

Known before writing this (RelBench paper arXiv 2407.20060, checked in the PDF): Table 5 test MAP (Past Visit /
best = ID-GNN): condition-sponsor-run 8.42 / 11.36, site-sponsor-run 17.31 / 19.00, user-post-comment 1.42 / 12.72,
post-post-related 1.74 / 10.83. Table 15 "%Repeated links" (test): 38.37, 23.29, 4.09, 1.39. No slice had been
computed on these tasks when the predictions below were written.

PRE-REGISTERED (rules use the same metric and heuristics as audit_relbench.py):
  E1 (critique holds where repeats are common): on both rel-trial tasks, repeat_then_recent_pop reaches >= 75% of
     the best published test MAP.
  E2 (critique does NOT transfer to rel-stack): on both rel-stack tasks, repeat_then_recent_pop reaches < 25% of the
     best published test MAP, i.e. the learned model beats memorisation by a wide margin there.
  E3 (slice shares): the novel slice is <= 60% of test positives on both rel-trial tasks and >= 80% on both rel-stack
     tasks.
  E4 (novel pairs are hard for heuristics everywhere): on every task, the best heuristic's novel-slice MAP is below
     half of its overall MAP.
Task-specific settings, fixed before running (windows scaled to each task's horizon):
  rel-trial: K = 10 (task eval_k), recent popularity window 365 d, co-occurrence window 1,095 d.
  rel-stack: K = 100 (task eval_k), recent popularity window 91 d, co-occurrence window 365 d.
History tables (all interactions with time <= T):
  condition-sponsor-run: (condition, sponsor) pairs of the same study, time = conditions_studies.date
  site-sponsor-run:      (facility, sponsor) pairs of the same study, time = facilities_studies.date
  user-post-comment:     (user, post) from comments, time = comment CreationDate
  post-post-related:     (post, related post) from postLinks, time = link CreationDate

  python scripts/audit_relbench_ext.py      # writes results/audit_relbench_ext.{md,json}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_relbench as A  # noqa: E402

D = A.DATA
DER = D / "derived"

A.PUBLISHED.update({  # RelBench paper Table 5, test MAP; columns as in audit_relbench.PUBLISHED
    "rel-trial/condition-sponsor-run": {"Global Popularity": 2.52, "Past Visit": 8.42, "LightGBM": 4.82,
                                        "RDL (GraphSAGE)": 2.89, "RDL (ID-GNN)": 11.36},
    "rel-trial/site-sponsor-run": {"Global Popularity": 3.75, "Past Visit": 17.31, "LightGBM": 8.40,
                                   "RDL (GraphSAGE)": 10.70, "RDL (ID-GNN)": 19.00},
    "rel-stack/user-post-comment": {"Global Popularity": 0.02, "Past Visit": 1.42, "LightGBM": 0.04,
                                    "RDL (GraphSAGE)": 0.11, "RDL (ID-GNN)": 12.72},
    "rel-stack/post-post-related": {"Global Popularity": 1.46, "Past Visit": 1.74, "LightGBM": 2.00,
                                    "RDL (GraphSAGE)": 0.07, "RDL (ID-GNN)": 10.83},
})


def build_histories():
    DER.mkdir(exist_ok=True)
    ss = pd.read_parquet(D / "rel-trial/db/sponsors_studies.parquet", columns=["nct_id", "sponsor_id"]).drop_duplicates()
    for src, tab, out in [("condition_id", "conditions_studies", "trial_condition_sponsor"),
                          ("facility_id", "facilities_studies", "trial_facility_sponsor")]:
        f = DER / f"{out}.parquet"
        if not f.exists():
            x = pd.read_parquet(D / f"rel-trial/db/{tab}.parquet", columns=["nct_id", src, "date"])
            x.merge(ss, on="nct_id")[[src, "sponsor_id", "date"]].astype({src: "int64", "sponsor_id": "int64"}).to_parquet(f)
    f = DER / "stack_user_post.parquet"
    if not f.exists():
        c = pd.read_parquet(D / "rel-stack/db/comments.parquet", columns=["UserId", "PostId", "CreationDate"]).dropna()
        c.astype({"UserId": "int64", "PostId": "int64"}).to_parquet(f)
    f = DER / "stack_post_related.parquet"
    if not f.exists():
        p = pd.read_parquet(D / "rel-stack/db/postLinks.parquet", columns=["PostId", "RelatedPostId", "CreationDate"]).dropna()
        p.astype({"PostId": "int64", "RelatedPostId": "int64"}).to_parquet(f)


TASKS = {
    "rel-trial/condition-sponsor-run": dict(k=10, hist="derived/trial_condition_sponsor.parquet", u="condition_id",
                                            i="sponsor_id", t="date", tu="condition_id", ti="sponsor_id",
                                            test="rel-trial/tasks/condition-sponsor-run/test.parquet",
                                            recent_days=365, cf_days=1095),
    "rel-trial/site-sponsor-run": dict(k=10, hist="derived/trial_facility_sponsor.parquet", u="facility_id",
                                       i="sponsor_id", t="date", tu="facility_id", ti="sponsor_id",
                                       test="rel-trial/tasks/site-sponsor-run/test.parquet",
                                       recent_days=365, cf_days=1095),
    "rel-stack/user-post-comment": dict(k=100, hist="derived/stack_user_post.parquet", u="UserId", i="PostId",
                                        t="CreationDate", tu="UserId", ti="PostId",
                                        test="rel-stack/tasks/user-post-comment/test.parquet",
                                        recent_days=91, cf_days=365),
    "rel-stack/post-post-related": dict(k=100, hist="derived/stack_post_related.parquet", u="PostId",
                                        i="RelatedPostId", t="CreationDate", tu="PostId", ti="postLinksIdList",
                                        test="rel-stack/tasks/post-post-related/test.parquet",
                                        recent_days=91, cf_days=365),
}


def main():
    build_histories()
    rng = np.random.default_rng(0)
    res = []
    for name, cfg in TASKS.items():
        A.K = cfg["k"]  # relbench_map and run_task read the module-level K
        r = A.run_task(name, cfg, 300, rng)
        r["k"] = cfg["k"]
        res.append(r)
    (ROOT / "results/audit_relbench_ext.json").write_text(json.dumps(res, indent=1, default=str))

    H = ["global_popularity", "recent_popularity", "repeat_then_recent_pop"]
    L = ["# Benchmark audit, contrast tasks (rel-trial, rel-stack)", "",
         "MAP@k x100 (k = task eval_k). Heuristics computed here; published numbers from RelBench Table 5 (test).", "",
         "| task | k | repeat | cf-nbr | novel | repeat+pop | best published | share of best |", "|---|---|---|---|---|---|---|---|"]
    for r in res:
        s = r["shares"]
        L.append(f"| {r['task']} | {r['k']} | {s['repeat']:.1%} | {s['cf_nbr']:.1%} | {s['novel']:.1%} | "
                 f"{r['overall']['repeat_then_recent_pop']:.2f} | {r['best_published']:.2f} | "
                 f"{r['repeat_heuristic_share_of_best']:.0%} |")
    L += ["", "Per slice (MAP x100):", "", "| task | slice | users | " + " | ".join(H) + " |", "|---|---|---|" + "---|" * len(H)]
    for r in res:
        L.append(f"| {r['task']} | overall | {r['test_users']} | " + " | ".join(f"{r['overall'][h]:.2f}" for h in H) + " |")
        for sl, v in r["slices"].items():
            L.append(f"| | {sl} | {v['users']} | " + " | ".join(f"{v[h]:.2f}" if h in v else "–" for h in H) + " |")
    R = {r["task"]: r for r in res}
    tr = ["rel-trial/condition-sponsor-run", "rel-trial/site-sponsor-run"]
    st = ["rel-stack/user-post-comment", "rel-stack/post-post-related"]
    sh = lambda t: R[t]["repeat_heuristic_share_of_best"]  # noqa: E731

    def e4(t):
        b = max(H, key=lambda h: R[t]["overall"][h])
        nov = R[t]["slices"]["novel"].get(b, 0.0)
        return nov < 0.5 * R[t]["overall"][b], f"{t.split('/')[1]}: {b} novel {nov:.2f} vs overall {R[t]['overall'][b]:.2f}"

    e4s = [e4(t) for t in R]
    v = [("E1 rel-trial repeat+pop >= 75% of best", all(sh(t) >= 0.75 for t in tr), ", ".join(f"{sh(t):.0%}" for t in tr)),
         ("E2 rel-stack repeat+pop < 25% of best", all(sh(t) < 0.25 for t in st), ", ".join(f"{sh(t):.0%}" for t in st)),
         ("E3 novel share <= 60% (trial), >= 80% (stack)",
          all(R[t]["shares"]["novel"] <= 0.6 for t in tr) and all(R[t]["shares"]["novel"] >= 0.8 for t in st),
          ", ".join(f"{R[t]['shares']['novel']:.0%}" for t in tr + st)),
         ("E4 best heuristic novel MAP < half its overall, every task", all(h for h, _ in e4s), "; ".join(d for _, d in e4s))]
    L += ["", "## Mechanical scoring", ""] + [f"- {'HOLDS' if h else 'FAILS'} — {n}: {d}" for n, h, d in v]
    md = "\n".join(L)
    print(md)
    (ROOT / "results/audit_relbench_ext.md").write_text(md + "\n")


if __name__ == "__main__":
    main()
