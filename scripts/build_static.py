"""Build a static copy of the demo for GitHub Pages: no server, every click still works.

It runs the real engine through the demo flow (interview, reading, corrections, enrich, a draft,
approval and a reply for each top account, with and without the "too big" correction) and records
every API response and agent event. apps/web/static-shim.js replays them in the browser by
intercepting fetch and EventSource, so app.js runs unchanged.

    uv run python scripts/build_static.py             # writes dist/
    uv run python scripts/build_static.py --publish   # and pushes dist/ to the gh-pages branch
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient  # noqa: E402

from api.main import STATE, app  # noqa: E402

WEB, DIST = ROOT / "apps" / "web", ROOT / "dist"
GETS = ("accounts", "pipeline", "graph", "system", "bottle")
NO_TEXT = "No, that's not quite right."


def main():
    t0 = time.time()
    with TestClient(app) as c:
        eng = STATE["engine"]
        D: dict = {"static": {}, "feedback": {}, "states": {}, "enrich": {}, "drafts": {}, "approve": {}, "replies": {}}

        def call(sid, method, path, body=None):
            """Run one request and return (response, events it emitted)."""
            s = eng.sessions.get(sid) if sid else None
            n = len(s.events) if s else 0
            r = c.request(method, path, json=body)
            r.raise_for_status()
            s = s or eng.sessions.get(r.json().get("id", ""))
            evs = list(s.events[n:]) if s else []
            return {"r": r.json(), "ev": json.loads(json.dumps(evs, default=str))}

        for p in ("/api/health", "/api/world", "/api/world/map", "/api/demo/script", "/api/model-proof", "/api/backtest"):
            D["static"][p] = c.get(p).json()
        script = D["static"]["/api/demo/script"]

        def interview():
            st = call(None, "POST", "/api/sessions")
            sid = st["r"]["id"]
            answers = [call(sid, "POST", f"/api/sessions/{sid}/answer", {"text": a}) for a in script["answers"]]
            reading = call(sid, "POST", f"/api/sessions/{sid}/reading")
            return sid, st, answers, reading

        def snapshot(sid, key):
            D["states"][key] = {g: c.get(f"/api/sessions/{sid}/{g}").json() for g in GETS}

        for fixed in (False, True):
            fk = "fixed" if fixed else "base"
            D["drafts"][fk], D["approve"][fk], D["replies"][fk] = {}, {}, {}
            sid, st, answers, reading = interview()
            if not fixed:  # the interview and reading are the same either way; record once
                D["start"], D["answers"], D["reading"] = st, answers, reading
                for card in reading["r"]["cards"]:  # a yes or a no on every card, each from a fresh session
                    s2, *_ = interview()
                    yes = call(s2, "POST", f"/api/sessions/{s2}/cards/{card['id']}/feedback", {"text": "Yes"})
                    s3, *_ = interview()
                    no_text = script["correction"] if card["id"] == "discovery" else NO_TEXT
                    no = call(s3, "POST", f"/api/sessions/{s3}/cards/{card['id']}/feedback", {"text": no_text})
                    D["feedback"][card["id"]] = {"yes": yes, "no": no}
            else:
                call(sid, "POST", f"/api/sessions/{sid}/cards/discovery/feedback", {"text": script["correction"]})
            snapshot(sid, fk)
            D["enrich"][fk] = call(sid, "POST", f"/api/sessions/{sid}/enrich")
            top = [a["id"] for a in c.get(f"/api/sessions/{sid}/accounts").json()["accounts"]]
            for aid in top:  # one full outreach thread per account, each in its own session
                s, *_ = interview()
                if fixed:
                    call(s, "POST", f"/api/sessions/{s}/cards/discovery/feedback", {"text": script["correction"]})
                d = call(s, "POST", f"/api/sessions/{s}/drafts", {"account_id": aid})
                D["drafts"][fk][str(aid)] = d
                snapshot(s, f"{fk}|{aid}|drafted")
                did = d["r"]["id"]
                D["approve"][fk][str(aid)] = call(s, "POST", f"/api/sessions/{s}/drafts/{did}/approve", {"body": d["r"]["body"]})
                snapshot(s, f"{fk}|{aid}|sent")
                D["replies"][fk][str(aid)] = call(s, "POST", "/api/webhooks/reply",
                                                  {"text": "Sounds good, let's talk next week.", "draft_id": did})
                snapshot(s, f"{fk}|{aid}|replied")
                # accounts the re-rank brings into the top 5 can still be drafted
                for a in c.get(f"/api/sessions/{s}/accounts").json()["accounts"]:
                    if str(a["id"]) not in D["drafts"][fk]:
                        D["drafts"][fk][str(a["id"])] = call(s, "POST", f"/api/sessions/{s}/drafts", {"account_id": a["id"]})
            print(f"  {fk}: {len(top)} outreach threads", flush=True)

        # receipts: every post a card or chip cites becomes a static page
        blob = json.dumps(D)
        posts = sorted(set(re.findall(r'"/api/posts/(\d+)"', blob)), key=int)
        if DIST.exists():
            shutil.rmtree(DIST)
        shutil.copytree(WEB, DIST / "static")
        (DIST / "api" / "posts").mkdir(parents=True)
        for pid in posts:
            h = c.get(f"/api/posts/{pid}").text.replace('href="/static/', 'href="../../static/')
            (DIST / "api" / "posts" / f"{pid}.html").write_text(h)

    blob = re.sub(r'"/api/posts/(\d+)"', r'"api/posts/\1.html"', blob)
    (DIST / "static" / "demo-data.json").write_text(blob)

    # paths: the site lives under /<repo>/, so absolute /static/ links become relative
    # icon variables (--i:url(...)) resolve against styles.css, which already lives in static/
    for f in (DIST / "static").glob("*.js"):
        f.write_text(f.read_text().replace("url(/static/", "url(").replace('"/static/', '"static/')
                     .replace("`/static/", "`static/").replace("'/static/", "'static/"))
    css = DIST / "static" / "styles.css"
    css.write_text(css.read_text().replace("url(/static/", "url(").replace('url("/static/', 'url("'))
    idx = (WEB / "index.html").read_text().replace("url(/static/", "url(").replace('"/static/', '"static/')
    idx = idx.replace("<head>", '<head>\n  <script src="static/static-shim.js"></script>', 1)
    (DIST / "index.html").write_text(idx)
    (DIST / ".nojekyll").write_text("")
    mb = (DIST / "static" / "demo-data.json").stat().st_size / 1e6
    print(f"dist/ built in {time.time() - t0:.0f}s: {len(posts)} receipts, {len(D['states'])} states, demo-data {mb:.1f} MB")


def publish():
    """Commit dist/ as the whole tree of gh-pages (git plumbing, a separate index: no nested repo)."""
    git = lambda *a, **kw: subprocess.run(["git", *a], cwd=ROOT, check=True, capture_output=True, text=True, **kw).stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        env = os.environ | {"GIT_INDEX_FILE": str(Path(tmp) / "index")}
        git("--work-tree", str(DIST), "add", "-A", "--force", ".", env=env)
        tree = git("write-tree", env=env)
    parent = subprocess.run(["git", "rev-parse", "-q", "--verify", "origin/gh-pages"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    commit = git("commit-tree", tree, *(["-p", parent] if parent else []), "-m",
                 f"Static demo from {git('rev-parse', '--short', 'HEAD')}")
    git("push", "origin", f"{commit}:refs/heads/gh-pages")
    print(f"pushed {commit[:7]} to gh-pages")


if __name__ == "__main__":
    main()
    if "--publish" in sys.argv:
        publish()
