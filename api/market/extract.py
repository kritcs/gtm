"""Extraction: post text -> cited triples (company, relation, object, month, post_id, span).

Two passes, as in the real pipeline:
1. A dictionary pass over known tool names, role patterns and attribute
   phrases, with a negation filter ("No Kafka experience needed",
   "migrating off X" are not adoption).
2. An "LLM pass" for whatever the dictionary missed. Here it is simulated:
   it is deliberately noisy (a few percent of its triples are not supported
   by the text), which is what the Reviewer agent exists to catch.

Every triple keeps the exact span it was read from, so a claim built on it
can be checked against the post.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .simulate import World

NEGATION = re.compile(r"\b(no|not|without|migrating off|moving off|replacing)\s+(\w+\s+){0,2}$", re.I)

ROLE_FEATURES = {
    "role:revops": re.compile(r"\b(RevOps|Revenue Operations)\b"),
    "role:controller": re.compile(r"\bController\b"),
    "role:finops": re.compile(r"\bFinOps\b"),
    "role:sales_ops": re.compile(r"\bSales Ops\b"),
    "role:data_eng": re.compile(r"\bData Engineer\b"),
    "role:ml_eng": re.compile(r"\bML Engineer\b"),
    "role:sre": re.compile(r"\bSRE\b"),
    "role:designer": re.compile(r"\bProduct Designer\b"),
    "role:pm": re.compile(r"\bProduct Manager\b"),
    "role:ae": re.compile(r"\bAccount Executive\b"),
    "role:sdr": re.compile(r"\bSDR\b"),
    "role:csm": re.compile(r"\bCustomer Success\b"),
    "role:security": re.compile(r"\bSecurity Engineer\b"),
    "role:growth": re.compile(r"\bGrowth Marketer\b"),
    "role:clinical_ops": re.compile(r"\bClinical Ops\b"),
    "role:logistics_analyst": re.compile(r"\bLogistics Analyst\b"),
    "role:billing_specialist": re.compile(r"\bBilling Specialist\b"),
}
ATTR_FEATURES = {
    "attr:usage_based_pricing": re.compile(r"usage-based pricing", re.I),
    "attr:multi_entity": re.compile(r"multiple legal entities", re.I),
    "attr:intl_expansion": re.compile(r"expanding into Europe", re.I),
    "attr:recently_raised": re.compile(r"just raised our [\w ]+", re.I),
    "attr:remote_first": re.compile(r"\| REMOTE \|"),
}
FEATURE_LABELS = {
    "role:revops": "hired RevOps", "role:controller": "hired a Controller", "role:finops": "hired FinOps",
    "role:sales_ops": "hired Sales Ops", "role:data_eng": "hired data engineers", "role:ml_eng": "hired ML engineers",
    "role:sre": "hired SREs", "role:designer": "hired a designer", "role:pm": "hired product managers",
    "role:ae": "hired account executives", "role:sdr": "hired SDRs", "role:csm": "hired customer success",
    "role:security": "hired security engineers", "role:growth": "hired growth marketers",
    "role:clinical_ops": "hired clinical ops", "role:logistics_analyst": "hired logistics analysts",
    "role:billing_specialist": "hired billing specialists",
    "attr:usage_based_pricing": "moved to usage-based pricing", "attr:multi_entity": "added legal entities",
    "attr:intl_expansion": "expanded into Europe", "attr:recently_raised": "had just raised",
    "attr:remote_first": "went remote-first",
}


@dataclass
class Triple:
    company_id: int
    relation: str  # "ADOPTED" | "HAS_FEATURE"
    obj: str  # tool name or feature key
    month: int
    post_id: int
    span: str
    extractor: str  # "dictionary" | "llm"
    tool_id: int | None = None


def _tool_patterns(world: World):
    pats = []
    for t in world.tools:
        pats.append((t, re.compile(r"(?<![\w.])" + re.escape(t.name) + r"(?![\w])")))
    return pats


def extract(world: World, llm_noise: float = 0.04, seed: int = 11) -> list[Triple]:
    rng = np.random.default_rng(seed)
    pats = _tool_patterns(world)
    out: list[Triple] = []
    for p in world.posts:
        if p.kind != "hiring" or p.company_id is None:
            continue
        found = set()
        for t, pat in pats:
            for mt in pat.finditer(p.text):
                prefix = p.text[max(0, mt.start() - 40): mt.start()]
                if NEGATION.search(prefix):
                    continue
                found.add(t.id)
                out.append(Triple(p.company_id, "ADOPTED", t.name, p.month, p.id,
                                  p.text[max(0, mt.start() - 8): mt.end() + 8], "dictionary", t.id))
                break
        header = p.text.split("\n", 1)[0]
        for key, pat in ROLE_FEATURES.items():
            mt = pat.search(header)
            if mt:
                out.append(Triple(p.company_id, "HAS_FEATURE", key, p.month, p.id, mt.group(0), "dictionary"))
        for key, pat in ATTR_FEATURES.items():
            mt = pat.search(p.text)
            if mt:
                out.append(Triple(p.company_id, "HAS_FEATURE", key, p.month, p.id, mt.group(0).strip("| "), "dictionary"))
        # simulated LLM pass: resolves free text, occasionally hallucinates a tool
        if rng.random() < llm_noise:
            fake = world.tools[int(rng.integers(0, world.product_tool_id))]
            if fake.id not in found:
                out.append(Triple(p.company_id, "ADOPTED", fake.name, p.month, p.id,
                                  f"(inferred) {fake.name}", "llm", fake.id))
    return out
