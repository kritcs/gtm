"""The bottle: a story-first interview, live extraction, and founder rules with Beta trust.

Following Polanyi (experts know more than they can say), the interview asks
about specific moments rather than rules. Extraction turns the answers into:
- the experience set E (named companies, resolved against the market graph),
- key phrases (the founder's own words, reflected back later),
- pain / trigger / buyer rules, each with trust q_k ~ Beta(alpha_k, beta_k)
  that replies update (Kahneman and Klein: trust intuition where it gets feedback).
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

QUESTIONS = [
    ("customer", "Name your best recent customers. What was going on at those companies when they bought?"),
    ("lost", "Which deal did you lose that you expected to win, and why did it fall through?"),
    ("before", "What were customers using before you?"),
    ("buyer", "Who signs the contract, and who uses the product day to day?"),
]

PAIN_WORDS = ["spreadsheet", "reconcil", "manual", "by hand", "close", "trust", "match", "invoice",
              "export", "numbers", "weeks", "drowning", "mess", "error", "late"]
TRIGGER_PAT = re.compile(r"(had just [^.,;]+|just (?:raised|closed|hired|launched|moved)[^.,;]*|right before[^.,;]*|"
                         r"after (?:they|we|a|the) [^.,;]+|when (?:they|the) [^.,;]+)", re.I)
BUYER_PAT = re.compile(r"\b(CFO|VP (?:of )?Finance|Head of Finance|[Cc]ontroller|Finance (?:lead|team)|CEO|COO|"
                       r"VP (?:of )?Sales|Head of (?:Revenue|Sales|Ops))\b")
SIZE_PAT = re.compile(r"\b(too (?:big|large|enterprise)|too small|too early)\b", re.I)


# How sure the founder sounded sets the prior's pseudo-counts (alpha + beta).
CERTAIN = ("every", "always", "all of", "definitely", "the same story", "exactly", "never", "clearly", "each")
HEDGED = ("maybe", "might", "i think", "probably", "sometimes", "not sure", "i guess", "kind of", "sort of", "hunch")
CONFIDENCE = {"sure": (8.0, 2.0), "fairly sure": (4.0, 2.0), "hunch": (2.0, 2.0), "untested": (1.0, 1.0)}


def confidence_of(sentence: str) -> str:
    s = sentence.lower()
    if any(h in s for h in HEDGED):
        return "hunch"
    if any(c in s for c in CERTAIN):
        return "sure"
    return "fairly sure"


def sentence_with(text: str, phrase: str) -> str:
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if phrase.lower()[:24] in sent.lower():
            return sent
    return text


@dataclass
class Rule:
    id: str
    kind: str  # pain | trigger | buyer | segment
    text: str
    quote: str
    alpha: float = 4.0
    beta: float = 2.0
    conf: str = "fairly sure"
    credits: list = field(default_factory=list)  # accounts whose replies credited this belief

    @property
    def trust(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def to_dict(self) -> dict:
        text = self.text
        if ":" in text and " " not in text:  # a feature key found in the market data, e.g. role:revops
            from ..insight.hit import feature_label
            text = f"{feature_label(text)} (found in the market data)"
        return {"id": self.id, "kind": self.kind, "text": text, "key": self.text, "quote": self.quote,
                "alpha": self.alpha, "beta": self.beta, "trust": round(self.trust, 3), "conf": self.conf,
                "credits": list(self.credits)}


@dataclass
class Bottle:
    answers: list[dict] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)
    customers: list[int] = field(default_factory=list)
    lost: list[int] = field(default_factory=list)
    mentioned: list[int] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)

    @property
    def transcript(self) -> str:
        return " ".join(a["text"] for a in self.answers)

    @property
    def experience(self) -> list[int]:
        return list(dict.fromkeys(self.customers + self.lost + self.mentioned))

    def text_of(self, kind: str) -> str:
        return " ".join(r.text for r in self.rules if r.kind == kind)

    def founder_text(self) -> str:
        stories = [a["text"] for a in self.answers if a["key"] in ("customer", "before")]
        return " ".join([self.text_of("pain")] * 3 + [self.text_of("trigger")] + stories)

    def to_dict(self, world=None) -> dict:
        name = (lambda c: world.companies[c].name) if world else (lambda c: c)
        return {"answers": self.answers, "phrases": self.phrases,
                "customers": [name(c) for c in self.customers], "lost": [name(c) for c in self.lost],
                "mentioned": [name(c) for c in self.mentioned], "rules": [r.to_dict() for r in self.rules]}


def resolve_companies(text: str, names: dict[str, int]) -> list[int]:
    """Exact (case-insensitive) match first, then fuzzy on capitalized word runs."""
    found = []
    low = text.lower()
    for nm, cid in names.items():
        if re.search(r"\b" + re.escape(nm) + r"\b", low):
            found.append((low.index(nm), cid))
    if not found:
        for run in re.findall(r"\b[A-Z][a-z]+(?:\s?[A-Z][a-z]+)?\b", text):
            m = difflib.get_close_matches(run.lower(), names.keys(), n=1, cutoff=0.88)
            if m:
                found.append((text.index(run), names[m[0]]))
    return [cid for _, cid in sorted(found)]


PAIN_WEIGHT = {"spreadsheet": 2, "reconcil": 2, "by hand": 2, "manual": 2, "match": 1.5, "invoice": 1.5,
               "trust": 1.5, "close": 1, "export": 1, "numbers": 1, "weeks": 1, "drowning": 1, "mess": 1}


def _soft_lower(s: str) -> str:
    """Lowercase the first letter unless it starts an acronym or a name (CFO, Stripe)."""
    if not s or (len(s) > 1 and s[1].isupper()):
        return s
    return s[0].lower() + s[1:] if s.split()[0].lower() in {"exporting", "reconciling", "matching", "said",
                                                          "nobody", "spreadsheets", "sometimes"} else s


def key_phrases(text: str, limit: int = 3) -> list[str]:
    """Whole clauses that carry pain words, best first, in the founder's own casing."""
    clauses = [c.strip() for c in re.split(r"[.;!?]|, but | because ", text) if c.strip()]
    scored = []
    for i, c in enumerate(clauses):
        low = c.lower()
        sc = sum(wt for k, wt in PAIN_WEIGHT.items() if k in low)
        words = c.split()
        if sc >= 1.5 and len(words) > 9 and " and " in c:  # keep the half that carries the pain
            halves = [h for h in c.split(" and ") if len(h.split()) >= 3]
            if halves:
                c = max(halves, key=lambda h: sum(wt for k, wt in PAIN_WEIGHT.items() if k in h.lower()))
                words = c.split()
        if sc >= 1.5 and len(words) >= 3:
            short = " ".join(words[:10]).rstrip(" ,")
            short = re.sub(r"^((They|We|Their|Our|And|Honestly|I)\s+(were|was|are)?|said)\s*", "", short).strip()
            scored.append((-sc, i, _soft_lower(short)))
    return [p for _, _, p in sorted(scored)[:limit]]


def next_question(bottle: Bottle) -> dict | None:
    i = len(bottle.answers)
    if i >= len(QUESTIONS):
        return None
    key, q = QUESTIONS[i]
    return {"index": i, "key": key, "text": q, "total": len(QUESTIONS)}


def absorb(bottle: Bottle, text: str, names: dict[str, int]) -> dict:
    """Process one answer; returns what was learned (for the live UI)."""
    i = len(bottle.answers)
    key = QUESTIONS[i][0] if i < len(QUESTIONS) else "extra"
    bottle.answers.append({"key": key, "text": text})
    learned: dict = {"phrases": [], "companies": [], "rules": []}

    cos = resolve_companies(text, names)
    target = {"customer": bottle.customers, "lost": bottle.lost}.get(key, bottle.mentioned)
    for c in cos:
        if c not in bottle.experience:
            target.append(c)
            learned["companies"].append({"id": c, "role": "customer" if key == "customer" else
                                         "lost deal" if key == "lost" else "mentioned"})
    ph = [p for p in key_phrases(text) if p not in bottle.phrases]
    bottle.phrases += ph
    learned["phrases"] = ph

    def add(kind, txt, quote):
        conf = confidence_of(sentence_with(quote, txt))
        a, b = CONFIDENCE[conf]
        r = Rule(f"r{len(bottle.rules) + 1}", kind, txt, quote, alpha=a, beta=b, conf=conf)
        bottle.rules.append(r)
        learned["rules"].append(r.to_dict())

    if key in ("customer", "before"):
        for p in ph:
            add("pain", p, text)
    for m in TRIGGER_PAT.findall(text):
        add("trigger", " ".join(m.split(" and ")[0].split()[:8]).lower(), text)
    for m in dict.fromkeys(BUYER_PAT.findall(text)):
        add("buyer", m, text)
    sm = SIZE_PAT.search(text)
    if sm:
        add("segment", f"not {sm.group(1).lower().replace('too ', '')} companies", text)
    return learned
