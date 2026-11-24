"""End-to-end check: scripted interview -> reading -> GTM loop, with the asserts from the plan."""
import asyncio
import sys
import time

sys.path.insert(0, ".")
from api.engine import Engine, Session  # noqa: E402


async def main(verbose=True):
    t = time.time()
    eng = Engine()
    if verbose:
        print(f"engine built in {eng.build_seconds}s")
    s = Session(eng)
    await s.start()
    for a in eng.script["answers"]:
        r = await s.answer(a)
    assert r["next"] is None
    t1 = time.time()
    cards = await s.run_reading()
    dt = time.time() - t1
    by = {c.id: c for c in cards}
    if verbose:
        print(f"reading in {dt:.1f}s")
        for c in cards:
            print(f"[{c.kind}] ({c.clarity:.2f}) {c.front_text}  receipts={len(c.receipts)}")
            print("     ", c.back_text[:220])
    # asserts from the plan
    for c in cards:
        if c.kind not in ("reflection", "blindspot"):
            assert c.receipts, f"{c.kind} has no receipt"
            assert all(r.verified and r.date and r.url for r in c.receipts), f"{c.kind} unverified receipt"
    hit = by["hit"]
    assert hit.stats.get("q_value", 1) < 0.05, "hit not significant"
    assert hit.stats["feature"] == "role:revops", hit.stats
    assert "revops" not in s.bottle.transcript.lower()
    assert by["discovery"].stats["quoted"] >= 8, by["discovery"].stats
    assert dt < 10, "reading over the 10 s budget"
    # correction
    new = await s.feedback("discovery", eng.script["correction"])
    if verbose:
        print("correction:", new.front_text, "|", new.diff)
    # GTM loop
    en = await s.enrich()
    d = await s.draft(s.accounts[0]["id"])
    if verbose:
        print("enrich:", en["balance"], "left;", len(en["ledger"]), "enriched; skipped", en["skipped"])
        print("draft flags:", d["flags"]); print(d["body"])
    try:
        from api.gtm.outreach import send
        await send(d)
        raise SystemExit("sent without approval!")
    except PermissionError:
        pass
    await s.approve(d["id"])
    await s.reply(d["id"], "Yes, this is exactly our close. Thursday works.")
    assert s.pipeline()["Replied"], s.pipeline()
    print("demo_check OK" + (f" ({time.time() - t:.1f}s)" if verbose else ""))


if __name__ == "__main__":
    asyncio.run(main())
