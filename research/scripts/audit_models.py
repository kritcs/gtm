"""Benchmark audit, part 2: how do LEARNED recommenders score on each slice of rel-hm?

PRE-REGISTERED (written before any model in this script was trained or scored):
  LM1. Learned models (item-CF, BPR matrix factorization, LightGCN) beat the repeat + recent-popularity
       heuristic overall only modestly (or not at all).
  LM2. They score near zero on the novel-pair slice (positives that are neither repeats nor among the
       top-20 co-purchased items of something the user bought) — same order as the heuristics there.
  LM3. Any overall gain over the heuristics comes mostly from the repeat slice.
  If LM2 fails (a learned model scores well on novel pairs), that refutes the audit's critique and must be
  reported prominently.

Setup (rel-hm user-item-purchase, RelBench test timestamp T = 2020-09-14, label window 7 days):
  - Test users: a random subsample of --max-users of the 67,144 test users with >= 1 target (seeded).
  - Training data for every model: interactions with t <= T in the last --window-days days, for the sampled
    test users plus --extra-users other active users (all users with --max-users 0). Nothing after T is used.
    RelBench's own GNNs instead train on the train/val task tables; this is a simpler, leakage-free setup.
  - Item attributes are not used (article table not downloaded): models use ids only.
  - Hybrids repeat+<model>: the user's past items first (as in the repeat heuristic), then the model.
  - Every model's top-12 is backfilled with recent popularity (last 7 days) when it has < 12 items; users with
    no training interactions get recent popularity (same fallback as the heuristics).
  - Slices and MAP@12 are exactly as in scripts/audit_relbench.py (repeat, cf_nbr = top-20 co-purchased in
    the last 28 days, novel, cold_user, new_item, head10). Slice MAP: users with >= 1 positive in the slice,
    targets restricted to that slice, predictions unchanged.

Usage:
  local subsample:  .venv/bin/python scripts/audit_models.py --max-users 20000 --device cpu
  full (Colab GPU): python research/scripts/audit_models.py --max-users 0 --extra-users 0 --device cuda --epochs 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_relbench import DATA, K, PUBLISHED, TASKS, TOPN_CF, relbench_map  # noqa: E402

TASK = "rel-hm/user-item-purchase"


def load(cfg):
    test = pd.read_parquet(DATA / cfg["test"])
    T = test["timestamp"].max()
    test = test[test["timestamp"] == T].reset_index(drop=True)
    test["targets"] = test[cfg["ti"]].apply(lambda x: np.asarray(x if x is not None else [], dtype=np.int64))
    test = test[test["targets"].apply(len) > 0].reset_index(drop=True)
    h = pd.read_parquet(DATA / cfg["hist"], columns=[cfg["u"], cfg["i"], cfg["t"]])
    h = h[h[cfg["t"]] <= T]
    h.columns = ["u", "i", "t"]
    return test.rename(columns={cfg["tu"]: "u"}), h, T


# ------------------------------------------------------------------ slices (as in audit_relbench.py)

def slice_labels(test, h, T, cfg):
    users = set(test["u"].tolist())
    hu = h[h["u"].isin(users)]
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
    for u, tg in zip(test["u"].values, test["targets"].values):
        ui = user_items.get(u, set())
        cfn = set().union(*[nbr.get(j, set()) for j in ui]) if ui else set()
        for i in tg:
            rep = (u, i) in hist_pairs
            cf = (not rep) and (i in cfn)
            pos.append(dict(u=u, i=int(i), repeat=rep, cf_nbr=cf, novel=not rep and not cf,
                            cold_user=len(ui) == 0, new_item=i not in seen, head10=i in head10))
    return pd.DataFrame(pos), hu


def score_all(preds, test, P):
    targets = list(test["targets"].values)
    overall = {k: relbench_map(v, targets) for k, v in preds.items()}
    slices = {}
    for sl in ["repeat", "cf_nbr", "novel", "cold_user", "new_item", "head10"]:
        keep = P[P[sl]].groupby("u")["i"].apply(set).to_dict()
        rows = [r for r, u in enumerate(test["u"].values) if u in keep]
        if not rows:
            slices[sl] = {"users": 0}
            continue
        tsl = [np.array([i for i in targets[r] if i in keep[test.loc[r, "u"]]]) for r in rows]
        slices[sl] = {"users": len(rows), **{k: relbench_map(v[rows], tsl) for k, v in preds.items()}}
    return overall, slices


# ------------------------------------------------------------------ helpers

def backfill(top: list, recent: np.ndarray) -> np.ndarray:
    p = list(dict.fromkeys(top))[:K]
    for x in recent:
        if len(p) >= K:
            break
        if x not in p:
            p.append(x)
    return np.array(p[:K], dtype=np.int64)


def topk_dense(U: torch.Tensor, V: torch.Tensor, bias: torch.Tensor | None, rows: np.ndarray,
               item_ids: np.ndarray, chunk: int = 2048) -> list:
    """rows: index into U for each test user (-1 = cold) -> list of top-K item ids."""
    out = []
    for s in range(0, len(rows), chunk):
        r = torch.as_tensor(rows[s:s + chunk], device=U.device)
        u = torch.where(r[:, None] >= 0, U[r.clamp(min=0)], torch.zeros(1, U.shape[1], device=U.device))
        sc = u @ V.T
        if bias is not None:
            sc = sc + bias[None, :]
        out += [item_ids[ix] for ix in torch.topk(sc, K, dim=1).indices.cpu().numpy()]
    return out


# ------------------------------------------------------------------ models

def item_cf(train: pd.DataFrame, hu: pd.DataFrame, test_users: np.ndarray, recent, n_nbr: int = 50):
    """Item-item cosine on the training window; user score = sum of sims to the user's history items."""
    uid = {u: k for k, u in enumerate(train["u"].unique())}
    iids = train["i"].unique()
    iid = {i: k for k, i in enumerate(iids)}
    X = sp.csr_matrix((np.ones(len(train), dtype=np.float32),
                       (train["u"].map(uid).values, train["i"].map(iid).values)), shape=(len(uid), len(iids)))
    X.data[:] = 1.0
    norm = np.sqrt(np.asarray(X.sum(0)).ravel()) + 1e-9
    C = (X.T @ X).tocsr().astype(np.float32)
    C.setdiag(0)
    C.eliminate_zeros()
    C = sp.diags(1 / norm) @ C @ sp.diags(1 / norm)
    C = C.tocsr()
    rows, cols, vals = [], [], []
    for r in range(C.shape[0]):   # keep top-n neighbours per item
        s, e = C.indptr[r], C.indptr[r + 1]
        if e > s:
            top = np.argsort(-C.data[s:e])[:n_nbr]
            rows += [r] * len(top)
            cols += C.indices[s:e][top].tolist()
            vals += C.data[s:e][top].tolist()
    S = sp.csr_matrix((vals, (rows, cols)), shape=C.shape)
    hist = hu[hu["i"].isin(iid)].groupby("u")["i"].apply(lambda s: [iid[i] for i in set(s.values)]).to_dict()
    preds = []
    for u in test_users:
        items = hist.get(u, [])
        if not items:
            preds.append(backfill([], recent))
            continue
        v = np.asarray(S[items].sum(0)).ravel()
        nz = np.flatnonzero(v)
        top = nz[np.argsort(-v[nz])[:K]] if len(nz) else []
        preds.append(backfill([iids[j] for j in top], recent))
    return np.stack(preds)


def bpr_train(n_users, n_items, pairs: np.ndarray, dim, epochs, lr, reg, batch, device, lightgcn_adj=None,
              layers=2, seed=0, log=None):
    g = torch.Generator().manual_seed(seed)
    U = torch.nn.Parameter(0.1 * torch.randn(n_users, dim, generator=g))
    V = torch.nn.Parameter(0.1 * torch.randn(n_items, dim, generator=g))
    b = torch.nn.Parameter(torch.zeros(n_items))
    U.data, V.data, b.data = U.data.to(device), V.data.to(device), b.data.to(device)
    opt = torch.optim.Adam([U, V, b], lr=lr)
    P = torch.as_tensor(pairs, device=device)

    def embed():
        if lightgcn_adj is None:
            return U, V
        E = torch.cat([U, V], 0)
        acc = E
        for _ in range(layers):
            E = torch.sparse.mm(lightgcn_adj, E)
            acc = acc + E
        acc = acc / (layers + 1)
        return acc[:n_users], acc[n_users:]

    for ep in range(epochs):
        perm = torch.randperm(len(P), device=device)
        tot = 0.0
        for s in range(0, len(P), batch):
            idx = perm[s:s + batch]
            u, i = P[idx, 0], P[idx, 1]
            j = torch.randint(0, n_items, (len(idx),), device=device)
            Ue, Ve = embed()
            x = (Ue[u] * (Ve[i] - Ve[j])).sum(1) + b[i] - b[j]
            loss = -torch.nn.functional.logsigmoid(x).mean() + reg * (U[u].pow(2).sum(1) + V[i].pow(2).sum(1)
                                                                       + V[j].pow(2).sum(1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if log:
            log(f"  epoch {ep + 1}/{epochs} loss {tot / len(P):.4f}")
    with torch.no_grad():
        Ue, Ve = embed()
    return Ue.detach(), Ve.detach(), b.detach()


def norm_adj(n_users, n_items, pairs, device):
    r = np.concatenate([pairs[:, 0], pairs[:, 1] + n_users])
    c = np.concatenate([pairs[:, 1] + n_users, pairs[:, 0]])
    n = n_users + n_items
    deg = np.bincount(r, minlength=n).astype(np.float32)
    d = 1 / np.sqrt(np.maximum(deg, 1))
    v = d[r] * d[c]
    A = torch.sparse_coo_tensor(np.vstack([r, c]), v, (n, n)).coalesce()
    return A.to(device)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-users", type=int, default=20000, help="test-user subsample (0 = all)")
    ap.add_argument("--extra-users", type=int, default=100000, help="other active users added to training (0 = all)")
    ap.add_argument("--window-days", type=int, default=90)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/audit_models_subsample")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    cfg = TASKS[TASK]
    test, h, T = load(cfg)
    rng = np.random.default_rng(a.seed)
    if a.max_users and a.max_users < len(test):
        test = test.iloc[np.sort(rng.choice(len(test), a.max_users, replace=False))].reset_index(drop=True)
    test_users = test["u"].values
    log(f"T={T.date()} test users {len(test):,}, history {len(h):,}")

    P, hu = slice_labels(test, h, T, cfg)
    log(f"slices: {P[['repeat', 'cf_nbr', 'novel', 'cold_user', 'new_item']].mean().round(4).to_dict()}")

    recent = h.loc[h["t"] > T - pd.Timedelta(days=cfg["recent_days"]), "i"].value_counts().index.values[:200]
    last = hu.groupby(["u", "i"])["t"].agg(["max", "size"]).reset_index().sort_values(
        ["u", "max", "size"], ascending=[True, False, False])
    past = last.groupby("u")["i"].apply(lambda s: list(s.values[:K])).to_dict()
    preds = {
        "recent_popularity": np.tile(recent[:K], (len(test), 1)),
        "repeat_then_recent_pop": np.stack([backfill(past.get(u, []), recent) for u in test_users]),
    }

    # training window: sampled test users + extra active users
    win = h[h["t"] > T - pd.Timedelta(days=a.window_days)][["u", "i"]].drop_duplicates()
    if a.extra_users:
        others = np.setdiff1d(win["u"].unique(), test_users)
        keep = set(test_users.tolist()) | set(rng.choice(others, min(a.extra_users, len(others)), replace=False).tolist())
        win = win[win["u"].isin(keep)]
    log(f"training window {a.window_days}d: {len(win):,} pairs, {win['u'].nunique():,} users, {win['i'].nunique():,} items")

    preds["item_cf"] = item_cf(win, hu, test_users, recent)
    log("item-CF done")

    uid = {u: k for k, u in enumerate(win["u"].unique())}
    item_ids = win["i"].unique()
    iid = {i: k for k, i in enumerate(item_ids)}
    pairs = np.column_stack([win["u"].map(uid).values, win["i"].map(iid).values]).astype(np.int64)
    rows = np.array([uid.get(u, -1) for u in test_users])
    dev = torch.device(a.device)
    for name, adj in [("bpr_mf", None), ("lightgcn", "yes")]:
        A = norm_adj(len(uid), len(item_ids), pairs, dev) if adj else None
        log(f"training {name}")
        U, V, b = bpr_train(len(uid), len(item_ids), pairs, a.dim, a.epochs, lr=5e-3 if adj else 1e-2,
                            reg=a.reg, batch=a.batch, device=dev, lightgcn_adj=A, seed=a.seed, log=log)
        top = topk_dense(U, V, b, rows, item_ids)
        # cold users (no training interactions) fall back to recent popularity, as the heuristics do
        preds[name] = np.stack([backfill(list(t) if r >= 0 else [], recent) for t, r in zip(top, rows)])
        log(f"{name} done")

    # hybrids: the user's past items first (as the repeat heuristic), then the model's ranking
    for m in ["item_cf", "bpr_mf", "lightgcn"]:
        preds[f"repeat+{m}"] = np.stack([backfill(past.get(u, []) + list(preds[m][r]), recent)
                                         for r, u in enumerate(test_users)])
    overall, slices = score_all(preds, test, P)
    shares = {c: float(P[c].mean()) for c in ["repeat", "cf_nbr", "novel", "cold_user", "new_item", "head10"]}
    out = dict(task=TASK, T=str(T.date()), args=vars(a), test_users=len(test), positives=len(P), shares=shares,
               overall=overall, slices=slices, published=PUBLISHED[TASK], seconds=round(time.time() - t0, 1))
    print(json.dumps(out, indent=1, default=str))
    (ROOT / f"{a.out}.json").write_text(json.dumps(out, indent=1, default=str))

    methods = list(preds)
    lines = [__doc__.strip(), "", f"## Results: {len(test):,} test users (seed {a.seed}), {len(P):,} positives, "
             f"runtime {out['seconds']:.0f}s on {a.device}", "",
             "Slice shares of positives: " + ", ".join(f"{k} {v:.1%}" for k, v in shares.items()), "",
             "MAP@12 × 100 (RelBench metric). Published full-test numbers for reference (arXiv 2407.20060): "
             + ", ".join(f"{k} {v}" for k, v in PUBLISHED[TASK].items()) + ".", "",
             "| slice (users) | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods),
             "| **overall** (" + f"{len(test):,}) | " + " | ".join(f"{overall[m]:.2f}" for m in methods) + " |"]
    for sl, d in slices.items():
        if d.get("users"):
            lines.append(f"| {sl} ({d['users']:,}) | " + " | ".join(f"{d[m]:.2f}" for m in methods) + " |")
    (ROOT / f"{a.out}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-(len(slices) + 3):]))


if __name__ == "__main__":
    main()
