import asyncio

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.gtm import outreach
from api.gtm.reviewer import verify
from api.insight.stats import benjamini_hochberg, pct_rank, poisson_binomial_tail, shrunk_lift
from api.market.extract import Triple
from api.market.graph import snapshot
from api.ml.backtest import ASOF, assert_no_leakage


# ------------------------------------------------------------------ stats
def test_poisson_binomial_matches_independent_product():
    assert poisson_binomial_tail([0.05, 0.05, 0.05], 3) == pytest.approx(0.05**3)
    assert poisson_binomial_tail([0.2, 0.5], 0) == pytest.approx(1.0)
    # P(X >= 1) = 1 - prod(1 - p)
    assert poisson_binomial_tail([0.1, 0.3], 1) == pytest.approx(1 - 0.9 * 0.7)


def test_bh_is_monotone_and_at_least_p():
    p = [0.001, 0.04, 0.03, 0.5]
    q = benjamini_hochberg(p)
    assert all(qi >= pi for qi, pi in zip(q, p))
    assert q[0] == pytest.approx(0.004)
    assert max(q) <= 1


def test_shrunk_lift_tames_small_samples():
    assert shrunk_lift(3, 3, 0.05) < (3 / 3) / 0.05
    assert shrunk_lift(30, 30, 0.05) > shrunk_lift(3, 3, 0.05)


def test_pct_rank_puts_scales_together():
    a = pct_rank(np.array([10.0, -5.0, 3.0]))
    assert list(a) == [1.0, 0.0, 0.5]


# ------------------------------------------------------------------ engine
@pytest.fixture(scope="module")
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def engine(client):
    from api.main import STATE
    return STATE["engine"]


def test_reviewer_rejects_unsupported_and_negated(engine):
    post = next(p for p in engine.world.posts if p.kind == "hiring" and "Stack:" in p.text)
    assert not verify(Triple(post.company_id, "ADOPTED", "NotATool", post.month, post.id, "", "llm"), post.text).ok
    assert not verify(Triple(0, "ADOPTED", "Kafka", 0, 0, "", "llm"), "No Kafka experience needed.").ok
    assert verify(Triple(0, "ADOPTED", "Kafka", 0, 0, "", "llm"), "Stack: Kafka, Go.").ok
    assert engine.graph.rejected, "the noisy LLM pass should produce rejections"


def test_snapshot_has_no_leakage(engine):
    assert_no_leakage(snapshot(engine.graph, ASOF), ASOF)


def test_send_requires_approval():
    d = {"status": "draft", "id": "x", "subject": "s", "body": "b", "to": "a@b.example"}
    with pytest.raises(PermissionError):
        asyncio.run(outreach.send(d))


def test_ai_check_flags_generated_phrasing():
    assert outreach.ai_check("I hope this finds you well! Let's leverage synergy.")
    assert not outreach.ai_check("Saw you're hiring. 15 minutes Thursday?")


def test_full_flow_over_http(client, engine):
    r = client.post("/api/sessions").json()
    sid = r["id"]
    for a in engine.script["answers"]:
        out = client.post(f"/api/sessions/{sid}/answer", json={"text": a}).json()
    assert out["next"] is None
    cards = {c["id"]: c for c in client.post(f"/api/sessions/{sid}/reading").json()["cards"]}
    assert set(cards) == {"reflection", "hit", "discovery", "needed", "blindspot", "next_step"}
    assert cards["hit"]["stats"]["feature"] == "role:revops"
    assert cards["hit"]["stats"]["q_value"] < 0.05
    assert cards["discovery"]["stats"]["quoted"] >= 8
    for c in cards.values():
        for rc in c["receipts"]:
            assert rc["verified"]
            assert client.get(rc["url"]).status_code == 200

    fixed = client.post(f"/api/sessions/{sid}/cards/discovery/feedback", json={"text": "No, they're too big."}).json()
    assert fixed["version"] == 2 and "Removed" in fixed["diff"]

    accts = client.get(f"/api/sessions/{sid}/accounts").json()["accounts"]
    assert accts and all(a["chips"] for a in accts)
    en = client.post(f"/api/sessions/{sid}/enrich").json()
    assert en["balance"] >= 0 and en["ledger"]

    d = client.post(f"/api/sessions/{sid}/drafts", json={"account_id": accts[0]["id"]}).json()
    # a reply to an unsent draft is refused
    assert client.post("/api/webhooks/reply", json={"draft_id": d["id"], "text": "hi"}).status_code == 409
    client.post(f"/api/sessions/{sid}/drafts/{d['id']}/approve", json={})
    rep = client.post("/api/webhooks/reply", json={"text": f"Sounds good. ref {d['id']}"}).json()
    assert rep["draft"]["status"] == "replied"
    # the reply credits only beliefs that found this account, and each credited trust goes up
    assert rep["rules"] and all(u["trust"] > u["before"] for u in rep["rules"])
    assert all(accts[0]["name"] in u["credits"] for u in rep["rules"])
    if accts[0]["triggered"]:
        assert any(u["key"] == "role:revops" for u in rep["rules"])
    s = engine.sessions[sid]
    assert s.outcomes and s.positives == [accts[0]["id"]]
    assert s.ranking.meta["founder_weight"] > 0
    cols = client.get(f"/api/sessions/{sid}/pipeline").json()["columns"]
    assert [x["account"] for x in cols["Replied"]] == [accts[0]["name"]]
