"""EXPLORATORY (not pre-registered; written after seeing ID-GNN's rel-stack/user-post-comment slices).

ID-GNN scored 9.5 MAP on novel (user, post) pairs and 17.7 on users with no comment history, so it uses something
the comment-history heuristic does not. Hypothesis: users comment on their *own* threads — posts they wrote, answers
to their questions, or questions they answered — a relation in the posts table (OwnerUserId, ParentId), not in
comments. This script measures (1) the share of test positives in the user's own threads (posts created <= T), and
(2) MAP@100 of a no-learning "own thread, newest first, then comment repeats, then recent popularity" heuristic,
and (3) the share of ID-GNN hits that are own-thread.

  python scripts/audit_stack_ownthread.py runs/idgnn_ext/user-post-comment.csv
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
A.K = K = 100


def main():
    test = pd.read_parquet(D / "rel-stack/tasks/user-post-comment/test.parquet")
    T = test["timestamp"].max()
    test = test[test["timestamp"] == T].reset_index(drop=True)
    test["targets"] = test["PostId"].apply(lambda x: np.asarray(x if x is not None else [], dtype=np.int64))
    test = test[test["targets"].apply(len) > 0].reset_index(drop=True)
    P = pd.read_parquet(D / "rel-stack/db/posts.parquet", columns=["Id", "OwnerUserId", "ParentId", "CreationDate"])
    P = P[P["CreationDate"] <= T].dropna(subset=["OwnerUserId"])
    owner = dict(zip(P["Id"].astype(int), P["OwnerUserId"].astype(int)))
    created = dict(zip(P["Id"].astype(int), P["CreationDate"]))
    users = set(test["UserId"].tolist())
    own = P[P["OwnerUserId"].isin(users)]
    thread = {u: set() for u in users}
    kids = P.dropna(subset=["ParentId"]).groupby(P["ParentId"].dropna().astype(int))["Id"].apply(lambda s: s.astype(int).tolist()).to_dict()
    for pid, u, par in zip(own["Id"].astype(int), own["OwnerUserId"].astype(int), own["ParentId"]):
        thread[u].add(pid)                      # own post
        thread[u].update(kids.get(pid, []))     # answers to own question
        if pd.notna(par):
            thread[u].add(int(par))             # question the user answered
            thread[u].update(kids.get(int(par), []))  # sibling answers in that thread
    C = pd.read_parquet(D / "derived/stack_user_post.parquet")
    C = C[C["CreationDate"] <= T]
    past = C[C["UserId"].isin(users)].sort_values("CreationDate", ascending=False).groupby("UserId")["PostId"].apply(
        lambda s: list(dict.fromkeys(s.tolist()))).to_dict()
    recent = A.top_items(C.loc[C["CreationDate"] > T - pd.Timedelta(days=91), "PostId"], K)

    def pred(u):
        p = sorted(thread[u], key=lambda x: created.get(x, T), reverse=True)
        for x in list(past.get(u, [])) + list(recent):
            if len(p) >= K:
                break
            if x not in p:
                p.append(x)
        p = list(dict.fromkeys(p))[:K]
        return np.array(p + [-1] * (K - len(p)))

    targets = list(test["targets"].values)
    H = np.stack([pred(u) for u in test["UserId"]])
    pos = [(u, int(i)) for u, t in zip(test["UserId"], targets) for i in t]
    own_share = np.mean([i in thread[u] for u, i in pos])
    out = {"positives": len(pos), "own_thread_share": float(own_share),
           "own_thread_heuristic_map": A.relbench_map(H, targets)}
    if len(sys.argv) > 1:
        pdf = pd.read_csv(sys.argv[1])
        row = {u: json.loads(x)[:K] for u, x in zip(pdf["UserId"], pdf["PostId"])}
        hits = [(u, i) for u, t in zip(test["UserId"], targets) for i in set(row[u]) & set(t.tolist())]
        out["idgnn_hits"] = len(hits)
        out["idgnn_hits_own_thread_share"] = float(np.mean([i in thread[u] for u, i in hits]))
    print(json.dumps(out, indent=1))
    (ROOT / "results/audit_stack_ownthread.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
