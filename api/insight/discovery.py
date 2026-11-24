"""Discovery, the stranger's post, and the blind spot.

discovery(S) = fit(S) * d(S, E) as written in the handoff rewards far-away junk,
because d has no upper bound. So: keep segments whose fit is in the top fifth,
then pick the one farthest from the founder's experience, and require at
least 8 companies with their own quote describing the pain.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ..cards import Card, receipt
from ..market.simulate import month_label
from ..ml.market import MarketModel
from ..ml.text import STOP, TOKEN, stem
from .ranking import Ranking
from .stats import pct_rank

MIN_SEGMENT = 8
PAIN_QUOTE_MIN = 0.12  # floor; the working threshold is relative (see quote_threshold)


def quote_threshold(rk: Ranking) -> float:
    """A company 'describes the pain' if its best post is in the top 30% of candidates and above a floor."""
    return max(PAIN_QUOTE_MIN, float(np.quantile(rk.pain[rk.candidates], 0.7)))


def kmeans(X: np.ndarray, k: int, iters: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    C = X[[int(rng.integers(len(X)))]]
    for _ in range(1, k):  # k-means++ init
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1).min(1)
        C = np.vstack([C, X[rng.choice(len(X), p=d / d.sum())]])
    for _ in range(iters):
        lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    return lab


def pain_sentence(text: str, founder_vec: np.ndarray, model: MarketModel) -> str:
    body = text.split("\n", 1)[-1]
    sents = [s.strip() + "." for s in body.split(".") if len(s.split()) >= 5 and "Stack:" not in s and "http" not in s]
    if not sents:
        return body[:140]
    v = model.lsa.transform_hybrid(sents)
    return sents[int(np.argmax(v @ founder_vec))]


def distinctive_phrase(quotes: list[str], market_quotes: list[str], founder_words: set[str]) -> str:
    def grams(q):  # adjacent in the original text, both content words
        w = TOKEN.findall(q.lower())
        return {f"{a} {b}" for a, b in zip(w, w[1:])
                if a not in STOP and b not in STOP and len(a) > 2 and len(b) > 2}
    seg = Counter(g for q in quotes for g in grams(q))
    mkt = Counter(g for q in market_quotes for g in grams(q))
    best, score = "", 0.0
    generic = {"still", "every", "done", "eats", "spends", "days", "week", "team", "between", "against", "miss"}
    for gram, n in seg.items():
        parts = gram.split()
        if n < 3 or any(stem(t) in founder_words or t in generic for t in parts):
            continue
        s = n * np.log1p(n) / (1 + mkt.get(gram, 0))
        if s > score:
            best, score = gram, s
    return best


def find_discovery(model: MarketModel, rk: Ranking, founder_vec: np.ndarray, founder_words: set[str],
                   experience: list[int], k: int = 12) -> Card:
    w = model.graph.world
    idx = np.flatnonzero(rk.candidates)
    X = np.hstack([model.company_graph[idx], model.company_text[idx]])
    lab = kmeans(X, k)
    fit_c = 0.7 * pct_rank(rk.pain[idx]) + 0.3 * pct_rank(rk.y_hat[idx])
    thr = quote_threshold(rk)
    segs = []
    for j in range(k):
        mem = idx[lab == j]
        if len(mem) < MIN_SEGMENT:
            continue
        quoted = [c for c in mem if rk.pain[c] >= thr]
        segs.append({"j": j, "members": mem, "quoted": quoted, "fit": float(fit_c[lab == j].mean()),
                     "d": float(rk.dist[mem].mean())})
    if not segs:
        return Card(id="discovery", kind="discovery", front_text="No segment outside your current customers qualified.",
                    clarity=0.3)
    fits = np.array([s["fit"] for s in segs])
    cut = np.quantile(fits, 0.8)
    eligible = [s for s in segs if s["fit"] >= cut and len(s["quoted"]) >= MIN_SEGMENT] or \
               [max(segs, key=lambda s: (len(s["quoted"]) >= MIN_SEGMENT, s["fit"]))]
    best = max(eligible, key=lambda s: s["d"])
    quoted = sorted(best["quoted"], key=lambda c: -rk.pain[c])
    posts = model.graph.posts
    quotes = [pain_sentence(posts[int(rk.pain_post[c])].text, founder_vec, model) for c in quoted]
    others = [c for c in idx[:400] if c not in set(best["members"])]
    mq = [pain_sentence(posts[int(rk.pain_post[c])].text, founder_vec, model) for c in others if rk.pain_post[c] > 0][:200]
    phrase = distinctive_phrase(quotes, mq, founder_words)
    inds = Counter(w.companies[c].industry for c in best["members"]).most_common(2)
    who = " and ".join(i for i, _ in inds)
    front = (f'{len(quoted)} {who} companies post about the same problem. Their term: "{phrase}".' if phrase
             else f"{len(quoted)} {who} companies post about the same problem.")
    # distinct wordings first: eight companies saying it eight ways is the evidence
    seen, first, rest = set(), [], []
    for c, q in zip(quoted, quotes):
        (rest if q in seen else first).append((c, q))
        seen.add(q)
    recs = [receipt(posts[int(rk.pain_post[c])], q, w.companies[c].name) for c, q in (first + rest)[:8]]
    clarity = float(np.clip(0.4 + 0.6 * best["fit"], 0, 1)) if len(quoted) >= MIN_SEGMENT else 0.35
    return Card(
        id="discovery", kind="discovery", front_text=front,
        back_text=(f"Segment of {len(best['members'])} companies, fit in the top fifth of {len(segs)} segments, "
                   f"and the farthest of those from the companies you know (mean distance {best['d']:.2f})."),
        clarity=clarity, receipts=recs,
        stats={"fit": best["fit"], "distance": best["d"], "segment_size": int(len(best["members"])),
               "quoted": len(quoted), "segments_considered": len(segs), "fit_cut": float(cut), "phrase": phrase,
               "industries": dict(inds)},
        companies=[_co(w.companies[c], rk, c) for c in quoted[:12]])


def find_needed(model: MarketModel, rk: Ranking, founder_vec: np.ndarray, now: int) -> Card:
    g = model.graph
    asks = [p for p in g.posts.values() if p.kind == "ask" and p.month >= now - 18]
    vecs = model.post_vecs[[model.post_row[p.id] for p in asks]]
    sims = vecs @ founder_vec
    p = asks[int(np.argmax(sims))]
    # the other companies in that same moment: similar pain within +/- 2 months
    thr = quote_threshold(rk)
    moment = []
    for cid, ps in g.posts_by_company.items():
        if not rk.candidates[cid]:
            continue
        near = [q for q in ps if abs(q.month - p.month) <= 2]
        if near:
            s = max(float(model.post_vecs[model.post_row[q.id]] @ founder_vec) for q in near)
            if s >= thr:
                moment.append((s, cid, max(near, key=lambda q: model.post_vecs[model.post_row[q.id]] @ founder_vec)))
    moment.sort(reverse=True)
    w = g.world
    recs = [receipt(p, p.title + " " + p.text, None)]
    recs += [receipt(q, pain_sentence(q.text, founder_vec, model), w.companies[c].name) for _, c, q in moment[:5]]
    return Card(id="needed", kind="needed",
                front_text=f"“{p.title}” ({p.date})",
                back_text=f"Closest public post to your problem. {len(moment)} companies posted about the same problem "
                          f"within two months of it.",
                clarity=float(np.clip(sims.max() * 1.6, 0.3, 1.0)), receipts=recs,
                stats={"similarity": float(sims.max()), "author": p.author, "title": p.title,
                       "moment_companies": len(moment), "window": f"{month_label(p.month - 2)}..{month_label(p.month + 2)}"},
                companies=[_co(w.companies[c], rk, c) for _, c, _ in moment[:5]])


def find_blindspot(model: MarketModel, rk: Ranking, experience: list[int], now: int,
                   skip: tuple[str, ...] = ()) -> Card:
    """High model score, no founder coverage, thin data: say so plainly."""
    w, g = model.graph.world, model.graph
    pr = pct_rank(rk.y_hat)
    known_inds = {w.companies[c].industry for c in experience}
    rows = []
    for ind in sorted({c.industry for c in w.companies}):
        mem = [c.id for c in w.companies if c.industry == ind and rk.candidates[c.id]]
        if not mem or ind in known_inds or ind in skip:
            continue
        recent = sum(1 for c in mem for p in g.posts_by_company.get(c, []) if p.month >= now - 12)
        score = float(pr[mem].mean())
        rows.append({"industry": ind, "companies": len(mem), "recent_posts": recent, "model_score": score,
                     "uncertainty": score / np.sqrt(1 + recent)})
    med = float(np.median([x["recent_posts"] / x["companies"] for x in rows]))
    thin = [x for x in rows if x["recent_posts"] / x["companies"] <= med] or rows
    r = max(thin, key=lambda x: x["model_score"])
    return Card(id="blindspot", kind="blindspot",
                front_text=f"{r['industry'].capitalize()}: {r['companies']} companies, "
                           f"{r['recent_posts'] / r['companies']:.1f} posts each this year. Not enough data to rank.",
                back_text="The market model scores this group above average, but none of your customers are in it "
                          "and there is little recent data. Scores here are low-confidence.",
                clarity=0.3, stats=r, revisable=False)


def _co(c, rk: Ranking, cid: int) -> dict:
    return {"id": c.id, "name": c.name, "industry": c.industry, "size": c.size, "stage": c.stage,
            "score": round(float(rk.score[cid]), 3), "pain": round(float(rk.pain[cid]), 3)}
