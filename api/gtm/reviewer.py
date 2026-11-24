"""Reviewer agent: checks that each extracted triple is supported by its cited post.

A triple survives only if the cited post's text actually contains the object
(a tool name not under negation, or a feature's pattern). Rejected triples
never reach the graph, and every verdict is emitted to the open-kitchen feed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..market.extract import ATTR_FEATURES, NEGATION, ROLE_FEATURES, Triple


@dataclass
class Verdict:
    ok: bool
    reason: str


def verify(triple: Triple, post_text: str) -> Verdict:
    if triple.relation == "ADOPTED":
        pat = re.compile(r"(?<![\w.])" + re.escape(triple.obj) + r"(?![\w])")
        mt = pat.search(post_text)
        if not mt:
            return Verdict(False, f"'{triple.obj}' does not appear in post {triple.post_id}")
        if NEGATION.search(post_text[max(0, mt.start() - 40): mt.start()]):
            return Verdict(False, f"'{triple.obj}' is negated in post {triple.post_id}")
        return Verdict(True, f"'{triple.obj}' named in post {triple.post_id}")
    if triple.relation == "HAS_FEATURE":
        pat = ROLE_FEATURES.get(triple.obj) or ATTR_FEATURES.get(triple.obj)
        if pat is None:
            return Verdict(False, f"unknown feature {triple.obj}")
        if pat.search(post_text):
            return Verdict(True, f"{triple.obj} supported by post {triple.post_id}")
        return Verdict(False, f"{triple.obj} not supported by post {triple.post_id}")
    return Verdict(False, f"unknown relation {triple.relation}")


def verify_claim(text: str, post_text: str) -> Verdict:
    """Check a free-text evidence quote: it must appear verbatim in the post."""
    if text.strip() and text.strip() in post_text:
        return Verdict(True, "quote found verbatim")
    return Verdict(False, "quote not found in cited post")
