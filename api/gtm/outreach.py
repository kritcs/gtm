"""Outreach in the founder's voice, an AI-sounding check, mandatory approval, and sending.

Voice comes from the founder's own sent emails: greeting, sign-off, sentence
length, and the reason they started the company. Nothing sends without an
explicit approval. Sending goes through Resend when RESEND_API_KEY and
DEMO_SEND_TO are set (the demo redirects every email to a teammate's inbox);
otherwise it lands in a local outbox.
"""
from __future__ import annotations

import os
import re
import uuid

import httpx

AI_TELLS = ["delve", "leverage", "synergy", "game-changer", "game changer", "seamless", "unlock", "revolutioniz",
            "i hope this finds you well", "in today's fast-paced", "cutting-edge", "robust", "elevate",
            "streamline your", "empower", "not only", "transformative", "at the end of the day", "!"]


def voice_profile(samples: list[str]) -> dict:
    greet = "Hey" if sum(s.lower().startswith("hey") for s in samples) >= len(samples) / 2 else "Hi"
    sign = samples[0].strip().split("\n")[-1].strip() if samples else ""
    sents = [x for s in samples for x in re.split(r"[.?!]\s", s) if x.strip()]
    avg = sum(len(x.split()) for x in sents) / max(len(sents), 1)
    reason = ""
    for s in samples:
        m = re.search(r"because ([^.]+)\.", s)
        if m:
            reason = m.group(1)
            break
    return {"greeting": greet, "signoff": sign, "avg_sentence_words": round(avg, 1), "reason": reason}


def ai_check(text: str) -> list[str]:
    low = text.lower()
    flags = [f'sounds generated: "{t}"' for t in AI_TELLS if t in low]
    if text.count("—") > 1:
        flags.append("too many em dashes")
    for s in re.split(r"(?<=[.?!])\s", text):
        if len(s.split()) > 28:
            flags.append(f"long sentence ({len(s.split())} words)")
    return flags


def draft_email(account: dict, voice: dict, product: str, customers: list[str]) -> dict:
    c = account["contact"]
    chips = {ch["kind"]: ch for ch in account["chips"]}
    lines = []
    if "trigger" in chips:
        role = re.sub(r"^Hired (an? )?", "", chips["trigger"]["label"].split(" · ")[0], flags=re.I)
        lines.append(f"Saw {account['name']} is hiring for {role}.")
    if "pain" in chips:
        lines.append("Your job post mentions month-end reconciliation problems.")
    if not lines and "stack" in chips:
        lines.append(f"Noticed you run {chips['stack']['label'].split(' · ')[0][5:]}.")
    why = f"We built {product} because {voice['reason']}." if voice.get("reason") else f"That's what {product} fixes."
    proof = f"{customers[0]} and {customers[1]} use it to close faster." if len(customers) >= 2 else ""
    body = "\n\n".join([f"{voice['greeting']} {c['first']},", " ".join(lines), f"{why} {proof}".strip(),
                        "15 minutes next week?", voice.get("signoff") or "Sam"])
    did = uuid.uuid4().hex[:8]
    return {"id": did, "account_id": account["id"], "account": account["name"], "to": c["email"],
            "to_name": c["first"], "to_role": c["role"], "subject": f"{account['name']}'s month-end close",
            "body": body, "flags": ai_check(body), "status": "draft",
            "why": (f"{c['first']}, {c['role']}, {account['name']}. Evidence: "
                    + "; ".join(ch["label"].strip('"') for ch in account["chips"][:2]) + ".")}


async def send(draft: dict) -> dict:
    if draft["status"] != "approved":
        raise PermissionError("human approval is required before any message goes out")
    key, to = os.environ.get("RESEND_API_KEY"), os.environ.get("DEMO_SEND_TO")
    if key and to:
        body = {"from": os.environ.get("RESEND_FROM", "onboarding@resend.dev"), "to": [to],
                "subject": f"{draft['subject']} [ref {draft['id']}]",
                "text": draft["body"] + f"\n\n(demo: originally addressed to {draft['to']})"}
        try:
            async with httpx.AsyncClient(timeout=6) as cl:
                r = await cl.post("https://api.resend.com/emails", json=body, headers={"Authorization": f"Bearer {key}"})
            return {"channel": "resend", "status": r.status_code, "to": to}
        except Exception as e:
            return {"channel": "resend", "status": f"failed: {e.__class__.__name__}", "to": to}
    return {"channel": "local outbox", "status": "queued", "to": draft["to"]}
