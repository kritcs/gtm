"""The shared contract: every insight returns a Card with receipts."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Receipt(BaseModel):
    post_id: int
    url: str
    date: str
    quote: str
    company: str | None = None
    verified: bool = True


class Card(BaseModel):
    id: str
    kind: str  # reflection | hit | discovery | needed | blindspot | next_step
    front_text: str
    back_text: str = ""
    clarity: float = Field(ge=0, le=1)
    receipts: list[Receipt] = []
    stats: dict[str, Any] = {}
    companies: list[dict[str, Any]] = []
    revisable: bool = True
    version: int = 1
    diff: str | None = None


def receipt(post, quote: str, company: str | None = None) -> Receipt:
    return Receipt(post_id=post.id, url=post.url, date=post.date, quote=quote, company=company)
