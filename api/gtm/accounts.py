"""Ranked accounts with evidence chips. Every chip cites a post and passes the Reviewer."""
from __future__ import annotations

import numpy as np

from ..cards import receipt
from ..insight.discovery import pain_sentence, quote_threshold
from ..insight.hit import feature_label
from ..insight.ranking import Ranking
from ..market.extract import Triple
from ..market.simulate import TOOLS, month_label
from ..ml.market import MarketModel
from .reviewer import verify, verify_claim

FIRST = ("Priya Tom Ana Marcus Lena Diego Hana Omar Grace Felix Ines Kofi Mira Jonas Ruth Theo Wen Sara "
         "Malik Elena").split()
TRIGGER_BOOST = 0.5
RECENT = 6


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def contact_for(company, buyer: str) -> dict:
    first = FIRST[company.id % len(FIRST)]
    return {"first": first, "role": buyer or "Head of Finance",
            "email": f"{first.lower()}@{company.domain}"}


def build_accounts(model: MarketModel, rk: Ranking, *, hit_feature: str | None, buyer: str, now: int,
                   founder_vec: np.ndarray, n: int = 5) -> tuple[list[dict], list[dict]]:
    """Returns (accounts, reviewer_log)."""
    g, w = model.graph, model.graph.world
    score = rk.score.copy()
    trig = {}
    if hit_feature:
        for c in np.flatnonzero(rk.candidates):
            win = g.feature_window(int(c), now - RECENT, now + 1)
            if hit_feature in win:
                trig[int(c)] = win[hit_feature]
                score[c] += TRIGGER_BOOST
    order = [int(c) for c in np.argsort(-score) if rk.candidates[c]]
    accounts, log = [], []
    finance_cats = set(TOOLS["billing"][0]) | set(TOOLS["crm"][0]) | set(TOOLS["accounting"][0])
    for c in order:
        co = w.companies[c]
        chips = []
        if c in trig:
            m, pid, span = trig[c]
            post = g.posts[pid]
            chips.append({"label": f"{_cap(feature_label(hit_feature))} · {month_label(m)}",
                          "kind": "trigger", "receipt": receipt(post, post.text.split(chr(10))[0], co.name).model_dump(),
                          "triple": Triple(c, "HAS_FEATURE", hit_feature, m, pid, span, "dictionary")})
        if rk.pain[c] >= quote_threshold(rk) and rk.pain_post[c] > 0:
            post = g.posts[int(rk.pain_post[c])]
            q = pain_sentence(post.text, founder_vec, model)
            chips.append({"label": f'"{q}"', "kind": "pain", "receipt": receipt(post, q, co.name).model_dump()})
        stack = [(t, m) for t, m in g.company_tools(c).items() if w.tools[t].name in finance_cats]
        if stack:
            t, m = max(stack, key=lambda x: x[1])
            pid = g.adopted[(c, t)][1]
            post = g.posts[pid]
            chips.append({"label": f"Uses {w.tools[t].name} · since {month_label(m)}", "kind": "stack",
                          "receipt": receipt(post, w.tools[t].name, co.name).model_dump(),
                          "triple": Triple(c, "ADOPTED", w.tools[t].name, m, pid, w.tools[t].name, "dictionary", t)})
        kept = []
        for ch in chips:
            post_text = g.posts[ch["receipt"]["post_id"]].text
            tr = ch.pop("triple", None)
            v = verify(tr, post_text) if tr else verify_claim(ch["receipt"]["quote"], post_text)
            log.append({"account": co.name, "chip": ch["label"], "ok": v.ok, "reason": v.reason})
            if v.ok:
                kept.append(ch)
        if not kept:  # citations are mandatory: no evidence, no account
            continue
        accounts.append({"id": c, "name": co.name, "domain": co.domain, "industry": co.industry, "size": co.size,
                         "stage": co.stage, "score": round(float(score[c]), 3),
                         "market": round(float(rk.y_hat[c]), 3), "founder": round(float(rk.s_founder[c]), 3),
                         "lambda": round(float(rk.lam[c]), 3), "distance": round(float(rk.dist[c]), 3),
                         "triggered": c in trig, "chips": kept, "contact": contact_for(co, buyer)})
        if len(accounts) >= n:
            break
    return accounts, log
