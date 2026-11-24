"""The market graph: reviewed triples, indexed by company, tool, feature and month.

Schema mirrors the brief's Neo4j model:
  (:Company)-[:ADOPTED {month, post_id}]->(:Tool)
  (:Company)-[:POSTED]->(:Post)
  (:Company)-[:HAS_FEATURE {month, post_id}]->(:Feature)
It lives in memory for the demo; `to_cypher()` emits the same graph as Cypher
for loading into Neo4j.

`asof` builds a leakage-safe snapshot: nothing dated after the cut-off
survives, which is what the backtest trains on.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..gtm.reviewer import verify
from .extract import Triple, extract
from .simulate import World


@dataclass
class MarketGraph:
    world: World
    triples: list[Triple]
    rejected: list[tuple[Triple, str]]
    asof: int | None = None
    adopted: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)  # (c,t)->(month, post_id)
    features: dict[int, list[tuple[str, int, int, str]]] = field(default_factory=lambda: defaultdict(list))
    posts_by_company: dict[int, list] = field(default_factory=lambda: defaultdict(list))
    posts: dict[int, object] = field(default_factory=dict)
    tools_by_company: dict[int, dict[int, int]] = field(default_factory=lambda: defaultdict(dict))

    def __post_init__(self):
        lim = self.asof if self.asof is not None else 10**9
        for p in self.world.posts:
            if p.month <= lim:
                self.posts[p.id] = p
                if p.company_id is not None:
                    self.posts_by_company[p.company_id].append(p)
        for tr in self.triples:
            if tr.month > lim:
                continue
            if tr.relation == "ADOPTED":
                key = (tr.company_id, tr.tool_id)
                if key not in self.adopted or tr.month < self.adopted[key][0]:
                    self.adopted[key] = (tr.month, tr.post_id)
            else:
                self.features[tr.company_id].append((tr.obj, tr.month, tr.post_id, tr.span))
        for (c, t), (m, _) in self.adopted.items():
            self.tools_by_company[c][t] = m

    # ------------------------------------------------------------------ queries
    def adopters(self, tool_id: int) -> dict[int, int]:
        return {c: m for (c, t), (m, _) in self.adopted.items() if t == tool_id}

    def company_tools(self, company_id: int) -> dict[int, int]:
        return self.tools_by_company.get(company_id, {})

    def feature_window(self, company_id: int, lo: int, hi: int) -> dict[str, tuple[int, int, str]]:
        """Features first observed in [lo, hi): key -> (month, post_id, span)."""
        out: dict[str, tuple[int, int, str]] = {}
        for key, m, pid, span in self.features.get(company_id, []):
            if lo <= m < hi and key not in out:
                out[key] = (m, pid, span)
        for t, m in self.company_tools(company_id).items():
            if lo <= m < hi and t != self.world.product_tool_id:
                k = f"stack:{self.world.tools[t].name}"
                out.setdefault(k, (m, self.adopted[(company_id, t)][1], self.world.tools[t].name))
        return out

    def active(self, company_id: int, lo: int, hi: int) -> bool:
        return any(lo <= p.month < hi for p in self.posts_by_company.get(company_id, []))

    def to_cypher(self, limit: int | None = None) -> str:
        w = self.world
        lines = []
        for c in w.companies[:limit]:
            lines.append(f"MERGE (:Company {{id:{c.id}, name:{c.name!r}, industry:{c.industry!r}, "
                         f"stage:{c.stage!r}, size:{c.size!r}}})")
        for t in w.tools:
            lines.append(f"MERGE (:Tool {{id:{t.id}, name:{t.name!r}, category:{t.category!r}}})")
        for (c, t), (m, pid) in list(self.adopted.items())[: (limit or 10**9) * 10]:
            lines.append(f"MATCH (c:Company {{id:{c}}}),(t:Tool {{id:{t}}}) "
                         f"MERGE (c)-[:ADOPTED {{month:{m}, post_id:{pid}}}]->(t)")
        return ";\n".join(lines) + ";"


def build_graph(world: World, asof: int | None = None) -> MarketGraph:
    """Extract, review, and index. Only reviewer-accepted triples enter the graph."""
    raw = extract(world)
    post_text = {p.id: p.text for p in world.posts}
    kept, rejected = [], []
    for tr in raw:
        v = verify(tr, post_text[tr.post_id])
        (kept if v.ok else rejected).append(tr if v.ok else (tr, v.reason))
    return MarketGraph(world, kept, rejected, asof=asof)


def snapshot(g: MarketGraph, asof: int) -> MarketGraph:
    """A leakage-safe view of an already-reviewed graph."""
    return MarketGraph(g.world, g.triples, g.rejected, asof=asof)
