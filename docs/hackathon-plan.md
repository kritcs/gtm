# Plan: building "Intuition in a bottle" in a 24-hour hackathon

## Context
The handoff describes a GTM engine for founders. The founder talks for a few minutes. The product combines what it hears (the founder's "gut") with a graph of which companies use which tools, built from public data. It then gives a "reading": a hit, a discovery, a stranger's post, a blind spot, and one email. The repo (`verygoodbread/gtm`, branch `krit/friendly-carson-htuofc`) is empty, so this is greenfield.

Constraints: 24 hours on site, two builders (Krit and an ML teammate), Next.js plus FastAPI. We don't yet know whether pre-building is allowed. The data question is answered below: yes, the core data can come from Hugging Face and public APIs, so hour 0 doesn't start with scraping.

Guiding rule: **build backwards from the 3-minute demo.** Every card in the reading is a query against the data and the model, so data is the critical path. The UI has to look finished even when a card falls back to cached output.

## Data: where it comes from (answering "can we get it from HF or the web?")
| Need | Source | Notes |
|---|---|---|
| Company ↔ tool adoption over time (the market graph) | HF `brusic/hacker-news-who-is-hiring-posts` (MIT, every top-level comment since 2011) | Treat a tool named in a company's hiring post in month *m* as an adoption edge with timestamp *m*. This gives us time, which the backtest needs. |
| Fresh 2025–26 months (if the HF copy is stale) | HN Algolia API (`hn.algolia.com/api/v1/search?tags=comment,story_<id>`) or HF `open-index/hacker-news` | Free and needs no key. Pull the monthly "Ask HN: Who is hiring?" threads after the HF cutoff. |
| "A stranger's real post" (pain match) | HN Algolia search over Ask HN and comments, plus the hiring posts themselves | Each post comes with a date and a link, which gives every claim a receipt. |
| Company attributes for the hit (size, industry, HQ, founded) | Wikidata SPARQL; fields parsed from the posts (remote, stage, roles hired such as "RevOps") | Use LLM extraction with the source span cited. |
| Fallbacks | `dansbecker/hackernews_hiring_posts` | Same data, older. |

Pipeline (`data/`):
1. `load_hn.py`: load the HF parquet, then top it up from Algolia to the latest month.
2. `extract.py`: Claude batch extraction, one post → `{company, tools[], roles[], size_hint, pain_phrases[], evidence_span}`. Run a regex/dictionary pass for ~300 known tools first, so the LLM only resolves the rest. Cache to parquet.
3. `resolve.py`: entity resolution. Normalize names, merge on domain or URL, then fuzzy-match, and link to Wikidata QIDs where possible.
4. `embed.py`: embed posts with `bge-small` (sentence-transformers, local, free) and store them in a pgvector or FAISS index.
5. `load_neo4j.py`: load the brief's schema `(:Company)-[:ADOPTED {month, post_id}]->(:Tool)`, `(:Company)-[:POSTED]->(:Post)`, `(:Company)-[:HAS_ATTR]->(:Feature)`. Each edge keeps `post_id` as its citation.

**If pre-building is allowed:** steps 1–4 and model training happen before the event, and the 24 hours go to the reading and the GTM engine. **If not:** the ML teammate runs steps 1–2 on a 2022–2025 slice in hours 0–4 (about 50k posts), which is enough.

## Repo layout
```
apps/web/          Next.js (App Router, Tailwind, Framer Motion): the reading plus the dashboard
api/               FastAPI: interview, bottle, insight engine, GTM engine, webhooks
  bottle/          interview state, extraction, experience set E, Beta trust
  insight/         hit.py, discovery.py, pain.py, blindspot.py → Card objects with receipts
  gtm/             ranking.py, reviewer.py (Band agent), flexprice.py, outreach.py, webhook.py
ml/                LightGCN training, desc→embedding map, fold-in, backtest.py, ablation.py
data/              pipeline scripts above
.github/workflows/ Snyk + lint
docker-compose.yml Neo4j + Postgres/pgvector + api + web
```
Shared contract: every insight endpoint returns `Card {id, kind, front_text, clarity(0–1), receipts[{post_url, date, quote}], stats{lift, p_chance, n}, revisable}`. Krit builds the UI against mocked cards from hour 1. The ML teammate fills in the real ones.

## Components and the math behind them
- **Market model** (`ml/`): LightGCN in PyTorch Geometric on the bipartite company–tool graph, trained with BPR loss. Use a temporal split: train through 2024-12, test on 2025. Fallback if LightGCN slips: implicit ALS (the `implicit` library), which trains in minutes and uses the same interface.
- **Cold start** (`ml/coldstart.py`): fit a ridge map g_φ from the text embeddings of tool descriptions to the LightGCN tool embeddings, which gives e⁽⁰⁾ for the founder's product. Then fold in the founder's named customers by optimizing that one embedding with every other embedding frozen, for about 50 steps.
- **Bottle** (`api/bottle/`): an interviewer agent (Claude) asks 3–4 story questions, each building on the last answer. Live extraction pulls out pain, trigger, buyer, named companies, and the key phrases that settle on screen. E is the set of resolved named companies. d(c, E) is the minimum cosine distance in LightGCN space. Founder rules carry trust q_k ~ Beta(α, β), stored in Postgres.
- **Ranking**: s(c) = ŷ + λ₀·e^{−d(c,E)/τ}·s_founder(c), where s_founder is the cosine between the company's posts and the founder's pain and trigger.
- **Hit** (`insight/hit.py`): for each binary feature f, compute lift = P(f | customers) / P(f | market) and the binomial chance of seeing it. Keep only features with p < 0.01 that the transcript never mentioned (checked with string and embedding match). Clarity on the card comes from p.
- **Discovery**: cluster the candidate companies (k-means on embeddings). Score each segment by fit(S) · d(S, E), and keep segments with 8 or more companies that come with quotes.
- **Pain match**: max cosine between the founder's pain vector and each post. Show the top post, dated and linked.
- **Blind spot**: segments with high model score but low founder coverage and little data. Say this honestly on the card.
- **Correction loop**: "No, they're too big" becomes a constraint parsed by the LLM (a filter or a reweighting). Re-run the card and show the diff ("Removed 3 companies over 500 people").

## GTM engine (thin but working)
- Accounts list: the top 5 companies, each with evidence chips (citations are mandatory).
- **Reviewer agent (Band)**: checks that each claim's triple is supported by the cited post and rejects unsupported ones. Its feed is shown as the "open kitchen".
- **Flexprice**: a credits meter. Enrichment runs in p_i/c_i order until the balance hits zero.
- **Outreach**: Claude drafts the email, using the founder's own sent emails as few-shot examples, then runs an "AI-sounding" check. An approval screen is mandatory. Send through Resend.
- **Reply webhook**: Resend inbound (or Postmark) → `/webhooks/reply` → update the Beta trust → push over SSE so the Kanban card moves to Replied.
- **Snyk** runs in GitHub Actions from hour 1, which is a cheap sponsor checkbox.

## Frontend: the reading (`apps/web/app/reading/`)
Full-screen states: Enter → Talk → Pause → Understood → Surprised → Needed → Blind spot → Next step → Dashboard.
- Voice: use the browser Web Speech API for speech-to-text, with a text-input fallback that's always available (noisy room). Text-to-speech for the interviewer is optional; text on screen is the default.
- Key phrases fade into place with no transcript. The "pause" flickers company names while `/reading/run` streams over SSE, with a 10 s budget. If it runs over, serve the precomputed reading.
- Cards flip (front: statement, back: receipts and stats). Opacity and blur map to clarity. Editorial type (serif headline, lots of white space), no mystical styling.
- Dashboard tabs: Accounts, Bottle inspector, Outreach queue, Pipeline (Kanban), and the Band feed.

## 24-hour schedule
| Hours | ML teammate | Krit |
|---|---|---|
| 0–2 | Settle the open questions; pick the "your product" tool (one with a steep 2024→25 adoption curve in HN posts, e.g. ClickHouse, dbt or Temporal; check counts first) | Scaffold the monorepo, docker-compose, Card contract, Snyk CI, mock cards |
| 2–6 | Load HN data, regex and LLM extraction, entity resolution | Reading screens Enter → Talk, interviewer agent, live extraction |
| 6–10 | Neo4j load, embeddings, train ALS then LightGCN, temporal split | Pause, Understood and Surprised screens with card flip, against mocks |
| 10–14 | Hit, discovery, pain match and blind spot on real data. **Milestone: the full reading runs end to end on real data** | Wire the real endpoints; correction loop |
| 14–18 | Backtest Recall@50 vs an LLM baseline (post-cutoff months only); ablation of market / founder / both | GTM engine: ranking view, Reviewer agent, Flexprice meter, outreach and approval |
| 18–20 | Cold-start fold-in; tune λ₀ and τ against the ablation | Resend send, reply webhook, Kanban, SSE |
| 20–22 | **Feature freeze.** Try the hit and the discovery on 2–3 real founders; fix what gets a shrug | Polish, cache the demo reading, fallbacks |
| 22–24 | Numbers on the close slide | Rehearse 3×, record a backup video, deploy (Vercel + Railway/Fly) |

Cut order if behind (from the handoff, plus): voice output → Wikidata attributes → LightGCN (keep ALS) → cold-start fold-in (use description embedding only) → Band feed visuals. Never cut: citations, the approval step, the hit card, the discovery card, the reply moment.

## Verification
- `ml/backtest.py` prints Recall@50 for model, LLM baseline, and the ablation (market / founder / both). The combination has to beat both parts, or we drop the slogan from the close.
- An end-to-end script `scripts/demo_check.py` runs a scripted transcript through `/interview` → `/reading/run`. It asserts that every card has at least one receipt with a working URL and date, that the hit's p < 0.01 and its feature isn't in the transcript, and that the discovery has 8 or more companies.
- Webhook test: send to a teammate's address, reply from a phone, and see the Kanban card move to Replied within 10 s.
- A Playwright smoke run through all reading states at phone and laptop widths.
- Run the full demo 3 times on the venue network before judging.

## Still open (does not block starting)
- Are the pre-event build rules confirmed? This changes only whether data and model hours move before hour 0.
- Where is the judging rubric? It decides how thin the Pipeline and analytics can be.
- Whose sent emails give the demo founder their voice?
- Do we do the sponsor run on Band or Flexprice? Only if the discovery is genuinely non-obvious.
