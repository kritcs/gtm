"""The demo founder: a scripted persona derived from the simulated market.

Ledgerlight's real (simulated) early customers play the founder's named
customers, exactly as the handoff suggests ("have a real tool with lots of
history play your product"). The script never mentions RevOps, so the hit
is something the founder didn't say.
"""
from __future__ import annotations

from .market.graph import MarketGraph
from .market.simulate import PRODUCT_DESCRIPTION, PRODUCT_NAME, month_index

FOUNDER = {
    "name": "Sam",
    "product": PRODUCT_NAME,
    "description": PRODUCT_DESCRIPTION,
    "sent_emails": [
        "Hey Priya,\n\nSaw you're closing the books with three people now. Congrats on the raise. "
        "We built Ledgerlight because our last close took eleven days. Happy to show you what it does to yours.\n\nSam",
        "Hey Tom,\n\nQuick one. You mentioned the Stripe export breaks every month. "
        "That's the exact thing we fix. 15 minutes Thursday?\n\nSam",
    ],
}


def pick_customers(g: MarketGraph, n: int = 3) -> list[tuple[int, int]]:
    w = g.world
    asof = month_index("2024-12")
    ad = g.adopters(w.product_tool_id)
    early = sorted((m, c) for c, m in ad.items() if m <= asof)
    picks = [(c, m) for m, c in early
             if "role:revops" in g.feature_window(c, m - 6, m)
             and w.companies[c].industry not in ("logistics", "healthtech")]
    return picks[:n]


def pick_lost(g: MarketGraph, exclude: set[int]) -> int:
    w = g.world
    ad = g.adopters(w.product_tool_id)
    big = [c for c in w.companies if c.size == "1000+" and c.id not in ad and c.id not in exclude
           and c.industry in ("saas-billing", "fintech", "martech")]
    return big[0].id


def script(g: MarketGraph) -> dict:
    w = g.world
    cust = pick_customers(g)
    names = [w.companies[c].name for c, _ in cust]
    lost = w.companies[pick_lost(g, {c for c, _ in cust})].name
    c1, c2, c3 = (names + ["", "", ""])[:3]
    answers = [
        f"The last one that got really excited was {c1}. Their CFO had just closed a messy quarter and said nobody "
        f"trusted the revenue numbers. They were reconciling Stripe against Salesforce in spreadsheets every month end. "
        f"Honestly {c2} and {c3} were the same story.",
        f"We lost {lost}. I thought we had it, but they were too big. They had a whole team doing the close "
        f"and went with consultants on top of NetSuite.",
        "Spreadsheets. Exporting invoices and deals into one giant sheet and matching them by hand, "
        "sometimes for two weeks after month end.",
        "The VP Finance signs. The controller feels the pain first because she owns the close.",
    ]
    return {"founder": FOUNDER, "answers": answers, "customers": names, "lost": lost,
            "correction": "No, they're too big."}
