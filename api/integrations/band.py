"""Band: the open-kitchen feed, posted into a real Band room when configured.

Band (https://docs.band.ai) gives agents shared chat rooms. Our pipeline is a
set of agents (Interviewer, Extractor, Market model, Hit finder, Discovery,
Pain matcher, Reviewer, Writer). Every step they take is posted to the
session's Band room as an *event* (tool_call / tool_result / thought / error),
which Band shows to humans without needing an @mention.

Env:
  BAND_API_KEY   agent API key (X-API-Key) from POST /api/v1/me/agents/register
  BAND_CHAT_ID   optional existing room; otherwise one room is created per session
  BAND_BASE_URL  default https://app.band.ai/api/v1/agent
Without a key, the same payloads are recorded locally and shown in the UI feed.
"""
from __future__ import annotations

import os

import httpx

BASE = os.environ.get("BAND_BASE_URL", "https://app.band.ai/api/v1/agent")
TYPES = {"tool_call", "tool_result", "thought", "error", "task"}


class BandClient:
    def __init__(self):
        self.key = os.environ.get("BAND_API_KEY")
        self.default_chat = os.environ.get("BAND_CHAT_ID")
        self.live = bool(self.key)
        self._client = httpx.AsyncClient(timeout=4.0) if self.live else None
        self.status = "live" if self.live else "local (set BAND_API_KEY to post to a Band room)"

    @property
    def headers(self):
        return {"X-API-Key": self.key or "", "Content-Type": "application/json"}

    async def create_room(self) -> str | None:
        if not self.live:
            return None
        if self.default_chat:
            return self.default_chat
        try:
            r = await self._client.post(f"{BASE}/chats", json={"chat": {}}, headers=self.headers)
            r.raise_for_status()
            data = r.json()
            return (data.get("data") or data).get("id")
        except Exception as e:  # never block the demo on the network
            self.status = f"live, room creation failed: {e.__class__.__name__}"
            return None

    def payload(self, agent: str, kind: str, content: str, metadata: dict | None = None) -> dict:
        return {"event": {"content": f"[{agent}] {content}",
                          "message_type": kind if kind in TYPES else "thought",
                          "metadata": {"agent": agent} | (metadata or {})}}

    async def post_event(self, chat_id: str | None, body: dict) -> str:
        if not (self.live and chat_id):
            return "local"
        try:
            r = await self._client.post(f"{BASE}/chats/{chat_id}/events", json=body, headers=self.headers)
            return "sent" if r.status_code < 300 else f"http {r.status_code}"
        except Exception as e:
            return f"failed: {e.__class__.__name__}"
