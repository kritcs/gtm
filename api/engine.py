"""Engine (built once at startup) and Session (one founder, one reading).

The reading is a pipeline of agents. Each step emits an event to the
session feed (and to Band when configured), so the UI can show the open
kitchen while the founder watches company names flicker past.
"""
from __future__ import annotations

import asyncio
import re
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from . import demo
from .bottle.interview import Bottle, Rule, absorb, next_question
from .cards import Card
from .gtm import outreach
from .gtm.accounts import build_accounts
from .insight.discovery import find_blindspot, find_discovery, find_needed
from .insight.hit import find_hit
from .insight.ranking import Constraints, rank
from .integrations.band import BandClient
from .integrations.flexprice import FlexpriceClient, enrichment_cost, plan_enrichment
from .market.graph import build_graph
from .market.simulate import N_MONTHS, SIZE_ORDER, generate, month_label
from .ml.market import MarketModel
from .ml.text import tokens

START_CREDITS = 30.0


class Engine:
    def __init__(self, seed: int = 7):
        t = time.time()
        self.world = generate(seed)
        self.graph = build_graph(self.world)
        self.model = MarketModel(self.graph)
        self.now = N_MONTHS - 1
        self.names = {c.name.lower(): c.id for c in self.world.companies}
        self.script = demo.script(self.graph)
        self.voice = outreach.voice_profile(demo.FOUNDER["sent_emails"])
        self.band = BandClient()
        self.flex = FlexpriceClient()
        self.sessions: dict[str, Session] = {}
        self.build_seconds = round(time.time() - t, 2)

    def summary(self) -> dict:
        w, g = self.world, self.graph
        return {"simulated": True, "seed": w.seed, "companies": len(w.companies), "tools": len(w.tools),
                "posts": len(w.posts), "triples_kept": len(g.triples), "triples_rejected": len(g.rejected),
                "adoption_edges": len(g.adopted), "through": month_label(self.now),
                "model": self.model.info, "build_seconds": self.build_seconds,
                "sample_names": [c.name for c in w.companies[::13]][:60],
                "integrations": {"band": self.band.status, "flexprice": self.flex.status,
                                 "resend": "live" if outreach.os.environ.get("RESEND_API_KEY") else "local outbox"}}


@dataclass
class Session:
    engine: Engine
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    bottle: Bottle = field(default_factory=Bottle)
    constraints: Constraints = field(default_factory=Constraints)
    cards: dict[str, Card] = field(default_factory=dict)
    accounts: list[dict] = field(default_factory=list)
    drafts: dict[str, dict] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    queues: list[asyncio.Queue] = field(default_factory=list)
    reviewer_log: list[dict] = field(default_factory=list)
    credits: float = START_CREDITS
    ledger: list[dict] = field(default_factory=list)
    band_chat: str | None = None
    wallet: dict = field(default_factory=dict)
    hit_rows: list[dict] = field(default_factory=list)
    hit_feature: str | None = None
    customers_signing: list[tuple[int, int]] = field(default_factory=list)

    # ------------------------------------------------------------- feed
    async def emit(self, agent: str, kind: str, content: str, **meta):
        body = self.engine.band.payload(agent, kind, content, meta)
        status = await self.engine.band.post_event(self.band_chat, body)
        item = {"ts": time.time(), "agent": agent, "type": kind, "content": content, "meta": meta, "band": status}
        self.events.append(item)
        for q in list(self.queues):
            q.put_nowait(item)

    async def start(self):
        self.band_chat = await self.engine.band.create_room()
        self.wallet = await self.engine.flex.open_account(self.id, int(START_CREDITS))
        await self.emit("Interviewer", "task", "Session opened. Starting the interview.",
                        band_room=self.band_chat, flexprice=self.wallet)

    # -------------------------------------------------------- interview
    def question(self):
        return next_question(self.bottle)

    async def answer(self, text: str) -> dict:
        learned = absorb(self.bottle, text, self.engine.names)
        w = self.engine.world
        for c in learned["companies"]:
            c["name"] = w.companies[c["id"]].name
        await self.emit("Extractor", "tool_result",
                        f"Heard {len(learned['phrases'])} phrases, {len(learned['companies'])} companies, "
                        f"{len(learned['rules'])} rules.", learned=learned)
        return {"learned": learned, "next": self.question(), "bottle": self.bottle.to_dict(w)}

    # ----------------------------------------------------------- reading
    def _signing(self) -> list[tuple[int, int]]:
        ad = self.engine.graph.adopters(self.engine.world.product_tool_id)
        return [(c, ad.get(c, self.engine.now)) for c in self.bottle.customers]

    def _founder_vec(self) -> np.ndarray:
        return self.engine.model.text_vec(self.bottle.founder_text())

    async def run_reading(self) -> list[Card]:
        e, w = self.engine, self.engine.world
        b = self.bottle
        await self.emit("Extractor", "thought", f"Experience set E: {', '.join(w.companies[c].name for c in b.experience) or 'empty'}.")
        if any(r.kind == "segment" and "big" in r.text for r in b.rules):
            pass  # the founder's lost-deal remark is a rule with trust, not yet a hard filter

        self.customers_signing = self._signing()
        await asyncio.sleep(0.2)
        await self.emit("Market model", "tool_call", "Cold start: description -> embedding, then fold in named customers.")
        rk = await asyncio.to_thread(self._rank)
        await self.emit("Market model", "tool_result",
                        f"Scored {rk.meta['candidates']} companies. Fold-in moved the product embedding by {rk.meta['shift']:.2f}.",
                        **{k: v for k, v in rk.meta.items() if isinstance(v, (int, float, str))})

        await self.emit("Hit finder", "tool_call", "Testing every feature in customers' six months before signing "
                                                   "against matched companies, with BH correction.")
        hit, rows = await asyncio.to_thread(find_hit, e.graph, self.customers_signing, b.transcript)
        self.hit_rows = rows
        self.hit_feature = hit.stats.get("feature")
        await self.emit("Hit finder", "tool_result", hit.front_text, q=hit.stats.get("q_value"),
                        tested=len(rows))
        if self.hit_feature:
            b.rules.append(Rule(f"r{len(b.rules) + 1}", "trigger", self.hit_feature, "found in the market data",
                                alpha=1.0, beta=1.0))

        fv = self._founder_vec()
        fwords = set(t for t in tokens(b.transcript) if "_" not in t)
        disc = await asyncio.to_thread(find_discovery, e.model, rk, fv, fwords, b.experience)
        await self.emit("Discovery", "tool_result", disc.front_text, **{k: v for k, v in disc.stats.items() if not isinstance(v, dict)})
        need = await asyncio.to_thread(find_needed, e.model, rk, fv, e.now)
        await self.emit("Pain matcher", "tool_result", need.front_text, similarity=need.stats.get("similarity"))
        blind = await asyncio.to_thread(find_blindspot, e.model, rk, b.experience, e.now,
                                        tuple(disc.stats.get("industries", {})))
        await self.emit("Blind spot", "thought", blind.front_text)

        await self._review_graph_sample()
        accounts, log = await asyncio.to_thread(build_accounts, e.model, rk, hit_feature=self.hit_feature,
                                                buyer=self._buyer(), now=e.now, founder_vec=fv, n=8)
        self.accounts = accounts
        for item in log:
            self.reviewer_log.append(item)
        bad = [x for x in log if not x["ok"]]
        await self.emit("Reviewer", "tool_result", f"Checked {len(log)} evidence chips on {len(accounts)} accounts; "
                                                   f"rejected {len(bad)}.")

        reflection = self._reflection()
        nxt = self._next_step()
        cards = [reflection, hit, disc, need, blind, nxt]
        for c in cards:
            self._verify_receipts(c)
            self.cards[c.id] = c
        await self.emit("Writer", "tool_result", "Reading ready.", cards=len(cards))
        return cards

    def _rank(self):
        e, b = self.engine, self.bottle
        ad = set(e.graph.adopters(e.world.product_tool_id))
        rk = rank(e.model, description=demo.FOUNDER["description"], customers=b.customers, experience=b.experience,
                  founder_text=b.founder_text(), exclude=set(b.experience) | ad, constraints=self.constraints)
        self.ranking = rk
        return rk

    def _buyer(self) -> str:
        """Prefer who the founder said *signs* (question 4) over anyone named in passing."""
        r = sorted((x for x in self.bottle.rules if x.kind == "buyer"), key=lambda x: "signs" not in x.quote)
        return r[0].text if r else "Head of Finance"

    def _reflection(self) -> Card:
        ph = self.bottle.phrases[:2]
        n = len(self.bottle.customers)
        said = " and ".join(f"“{p}”" for p in ph) if ph else "very little so far"
        return Card(id="reflection", kind="reflection", front_text=f"You said {said}. {n} customers, one pain.",
                    back_text="Your words, kept as rules with trust that replies will update.",
                    clarity=0.9, revisable=False,
                    stats={"rules": [r.to_dict() for r in self.bottle.rules]})

    def _next_step(self) -> Card:
        if not self.accounts:
            return Card(id="next_step", kind="next_step", front_text="No account has enough evidence yet.", clarity=0.3)
        a = self.accounts[0]
        ch = a["chips"][0]
        from .cards import Receipt
        return Card(id="next_step", kind="next_step",
                    front_text=f"Write to the {a['contact']['role']} at {a['name']} first.",
                    back_text=f"Why: {ch['label']}. Score {a['score']} (market {a['market']}, you {a['founder']}, "
                              f"weight on you {a['lambda']}).",
                    clarity=0.85, receipts=[Receipt(**c["receipt"]) for c in a["chips"]],
                    companies=[{"id": a["id"], "name": a["name"]}], stats={"account": a["id"]})

    def _verify_receipts(self, card: Card):
        posts = self.engine.graph.posts
        for r in card.receipts:
            p = posts.get(r.post_id)
            r.verified = bool(p) and (r.quote in p.text or r.quote in (p.title + " " + p.text))

    async def _review_graph_sample(self):
        """Show the Reviewer rejecting unsupported triples from the noisy LLM pass."""
        rej = self.engine.graph.rejected
        await self.emit("Reviewer", "thought", f"While building the graph I rejected {len(rej)} unsupported triples.")
        for tr, why in rej[:2]:
            await self.emit("Reviewer", "error",
                            f"Rejected ({self.engine.world.companies[tr.company_id].name}, {tr.relation}, {tr.obj}): {why}",
                            extractor=tr.extractor, post_id=tr.post_id)

    # -------------------------------------------------------- corrections
    async def feedback(self, card_id: str, text: str) -> Card:
        card = self.cards[card_id]
        t = text.lower().strip()
        rules = [r for r in self.bottle.rules if r.kind in ("pain", "trigger")]
        if re.match(r"^(yes|yeah|exactly|right|true)\b", t):
            for r in rules:
                r.alpha += 1
            card = card.model_copy(update={"clarity": min(1.0, card.clarity + 0.1), "version": card.version + 1,
                                           "diff": "You confirmed this. I'll lean on it more."})
            self.cards[card_id] = card
            await self.emit("Writer", "thought", f"Founder confirmed the {card_id} card.")
            return card

        before = {c["id"] for c in card.companies}
        m_big, m_small = re.search(r"too (big|large)|enterprise", t), re.search(r"too small|too early", t)
        m_not = re.search(r"not (\w[\w-]+)", t)
        changed = False
        if m_big or m_small:
            # drop the biggest (or smallest) size tier the card was showing
            sizes = sorted({self.engine.world.companies[c].size for c in before} or {"201-1000"}, key=SIZE_ORDER.index)
            if m_big:
                self.constraints.max_size = SIZE_ORDER[max(0, SIZE_ORDER.index(sizes[-1]) - 1)]
            else:
                self.constraints.min_size = SIZE_ORDER[min(len(SIZE_ORDER) - 1, SIZE_ORDER.index(sizes[0]) + 1)]
            changed = True
        elif m_not and m_not.group(1) in {c.industry for c in self.engine.world.companies}:
            self.constraints.exclude_industries += (m_not.group(1),)
            changed = True
        if not changed:
            for r in rules:
                r.beta += 1
            card = card.model_copy(update={"clarity": max(0.15, card.clarity - 0.25), "version": card.version + 1,
                                           "diff": "You disagreed. I've marked this as a guess and lowered my trust in it."})
            self.cards[card_id] = card
            await self.emit("Writer", "thought", f"Founder disagreed with {card_id}; trust lowered.")
            return card

        await self.emit("Writer", "tool_call", f"Re-running with constraint: {self._constraint_text()}.")
        rk = await asyncio.to_thread(self._rank)
        fv = self._founder_vec()
        fwords = set(t for t in tokens(self.bottle.transcript) if "_" not in t)
        e = self.engine
        if card_id == "discovery":
            new = await asyncio.to_thread(find_discovery, e.model, rk, fv, fwords, self.bottle.experience)
        elif card_id == "needed":
            new = await asyncio.to_thread(find_needed, e.model, rk, fv, e.now)
        else:
            self.accounts, _ = await asyncio.to_thread(build_accounts, e.model, rk, hit_feature=self.hit_feature,
                                                       buyer=self._buyer(), now=e.now, founder_vec=fv, n=8)
            new = self._next_step()
        removed = [c for c in before if not self.constraints.allows(e.world.companies[c])]
        new = new.model_copy(update={"version": card.version + 1,
                                     "diff": f"Removed {len(removed)} companies over the size you sell to "
                                             f"({self._constraint_text()}). Re-ranked {rk.meta['candidates']} companies."})
        self._verify_receipts(new)
        self.cards[card_id] = new
        if card_id != "next_step":
            self.accounts, _ = await asyncio.to_thread(build_accounts, e.model, rk, hit_feature=self.hit_feature,
                                                       buyer=self._buyer(), now=e.now, founder_vec=fv, n=8)
        await self.emit("Writer", "tool_result", new.diff or "")
        return new

    def _constraint_text(self) -> str:
        c = self.constraints
        parts = []
        if c.max_size:
            parts.append(f"at most {c.max_size} people")
        if c.min_size:
            parts.append(f"at least {c.min_size} people")
        if c.exclude_industries:
            parts.append("not " + ", ".join(c.exclude_industries))
        return "; ".join(parts) or "none"

    # -------------------------------------------------------- GTM engine
    async def enrich(self) -> dict:
        w = self.engine.world
        pool = [{"id": a["id"], "name": a["name"], "score": max(a["score"], 0.01),
                 "cost": enrichment_cost(w.companies[a["id"]])} for a in self.accounts
                if not a.get("enriched")]
        balance = self.credits
        live = await self.engine.flex.balance(self.wallet.get("wallet_id"))
        if live is not None:
            balance = live
        plan = plan_enrichment(pool, balance)
        for p in plan:
            ev = self.engine.flex.event(self.id, p["name"], p["cost"], "contact-resolver")
            status = await self.engine.flex.ingest(ev)
            self.credits -= p["cost"]
            self.ledger.append({"account": p["name"], "credits": p["cost"], "balance": round(self.credits, 2),
                                "flexprice": status, "event_id": ev["event_id"], "ratio": round(p["score"] / p["cost"], 4)})
            for a in self.accounts:
                if a["id"] == p["id"]:
                    a["enriched"] = True
            await self.emit("Flexprice", "tool_call", f"Spent {p['cost']:.0f} credits enriching {p['name']} "
                                                       f"(score/cost {p['score'] / p['cost']:.3f}).", flexprice=status)
        skipped = [p["name"] for p in pool if p not in plan]
        if skipped:
            await self.emit("Flexprice", "thought", f"Out of budget for {len(skipped)} accounts: {', '.join(skipped)}.")
        return {"balance": round(self.credits, 2), "start": START_CREDITS, "ledger": self.ledger, "skipped": skipped}

    async def draft(self, account_id: int) -> dict:
        a = next(x for x in self.accounts if x["id"] == account_id)
        names = [self.engine.world.companies[c].name for c in self.bottle.customers]
        d = outreach.draft_email(a, self.engine.voice, demo.FOUNDER["product"], names)
        self.drafts[d["id"]] = d
        await self.emit("Writer", "tool_result", f"Drafted an email to {d['to_role']} at {d['account']}.",
                        flags=d["flags"])
        return d

    async def approve(self, draft_id: str, body: str | None = None) -> dict:
        d = self.drafts[draft_id]
        if body:
            d["body"] = body
            d["flags"] = outreach.ai_check(body)
        d["status"] = "approved"
        res = await outreach.send(d)
        d["status"], d["delivery"] = "sent", res
        await self.emit("Writer", "tool_result", f"Approved and sent to {d['account']} via {res['channel']}.")
        return d

    async def reply(self, draft_id: str, text: str, sender: str = "") -> dict:
        d = self.drafts[draft_id]
        d["status"], d["reply"] = "replied", {"text": text, "from": sender, "ts": time.time()}
        acct = next((a for a in self.accounts if a["id"] == d["account_id"]), None)
        updated = []
        for r in self.bottle.rules:
            used = r.kind in ("pain", "buyer") or (r.kind == "trigger" and acct and acct.get("triggered")
                                                   and r.text == self.hit_feature)
            if used:
                r.alpha += 1
                updated.append(r.to_dict())
        await self.emit("Webhook", "tool_result", f"{d['account']} replied. Card moved to Replied.",
                        rules_updated=len(updated))
        return {"draft": d, "rules": updated}

    def pipeline(self) -> dict:
        cols = {"Suggested": [], "Drafted": [], "Sent": [], "Replied": []}
        by_acct = {d["account_id"]: d for d in self.drafts.values()}
        for a in self.accounts[:5]:
            d = by_acct.get(a["id"])
            col = {"draft": "Drafted", "approved": "Sent", "sent": "Sent", "replied": "Replied"}.get(
                d["status"] if d else "", "Suggested")
            cols[col].append({"account": a["name"], "account_id": a["id"], "draft_id": d["id"] if d else None})
        return cols
