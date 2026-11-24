"""Flexprice: prepaid credits that gate account enrichment.

Each founder session is a Flexprice customer with a PRE_PAID wallet. Every
enrichment spends credits and is metered as a usage event. Enrichment runs in
order of score / cost (greedy knapsack) until the wallet can't afford the
next account, so the founder's budget goes to the accounts most likely to matter.

API (https://docs.flexprice.io):
  POST /customers                      {external_id, name}
  POST /wallets                        {currency, external_customer_id, wallet_type, initial_credits_to_load}
  POST /events                         {event_name, external_customer_id, event_id, timestamp, properties}
  GET  /wallets/{id}/balance/real-time
Env: FLEXPRICE_API_KEY, FLEXPRICE_BASE_URL (default https://us.api.flexprice.io/v1).
Without a key, a local ledger mirrors the same events so the meter still works.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import httpx

BASE = os.environ.get("FLEXPRICE_BASE_URL", "https://us.api.flexprice.io/v1")
EVENT_NAME = os.environ.get("FLEXPRICE_EVENT_NAME", "account_enrichment")


class FlexpriceClient:
    def __init__(self):
        self.key = os.environ.get("FLEXPRICE_API_KEY")
        self.live = bool(self.key)
        self._client = httpx.AsyncClient(timeout=5.0) if self.live else None
        self.status = "live" if self.live else "local ledger (set FLEXPRICE_API_KEY to meter in Flexprice)"

    @property
    def headers(self):
        return {"x-api-key": self.key or "", "Content-Type": "application/json"}

    async def _post(self, path: str, body: dict) -> tuple[int, dict]:
        if not self.live:
            return 0, {}
        try:
            r = await self._client.post(f"{BASE}{path}", json=body, headers=self.headers)
            return r.status_code, (r.json() if r.content else {})
        except Exception as e:
            return -1, {"error": e.__class__.__name__}

    async def open_account(self, external_id: str, credits: int) -> dict:
        out = {"customer": None, "wallet_id": None}
        code, _ = await self._post("/customers", {"external_id": external_id, "name": f"Founder session {external_id}"})
        out["customer"] = code
        code, w = await self._post("/wallets", {"currency": "USD", "external_customer_id": external_id,
                                                "wallet_type": "PRE_PAID", "conversion_rate": "1",
                                                "initial_credits_to_load": str(credits)})
        out["wallet_id"] = (w.get("id") if isinstance(w, dict) else None)
        return out

    def event(self, external_id: str, account: str, credits: float, provider: str) -> dict:
        return {"event_name": EVENT_NAME, "external_customer_id": external_id,
                "event_id": f"enrich_{uuid.uuid4().hex[:12]}",
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source": "gtm-engine",
                "properties": {"account": account, "credits": str(credits), "provider": provider}}

    async def ingest(self, body: dict) -> str:
        if not self.live:
            return "local"
        code, _ = await self._post("/events", body)
        return "accepted" if code == 202 else f"http {code}"

    async def balance(self, wallet_id: str | None) -> float | None:
        if not (self.live and wallet_id):
            return None
        try:
            r = await self._client.get(f"{BASE}/wallets/{wallet_id}/balance/real-time", headers=self.headers)
            d = r.json()
            return float(d.get("real_time_credit_balance") or d.get("credit_balance") or d.get("balance"))
        except Exception:
            return None


def enrichment_cost(company) -> float:
    """Bigger companies cost more to enrich (more contacts to resolve)."""
    return {"1-50": 3.0, "51-200": 5.0, "201-1000": 8.0, "1000+": 13.0}[company.size]


def plan_enrichment(accounts: list[dict], balance: float) -> list[dict]:
    """Greedy by score / cost; skip what we can't afford, keep going for cheaper ones."""
    order = sorted(accounts, key=lambda a: -a["score"] / a["cost"])
    plan, left = [], balance
    for a in order:
        if a["cost"] <= left:
            left -= a["cost"]
            plan.append(a)
    return plan
