# Intuition in a bottle

A GTM engine for founders breaking into a market. The founder talks through a short interview. The engine combines what it hears with a market graph built from public hiring posts, then gives a **reading** with checkable evidence, followed by ranked accounts and approved outreach.

> **Demo data is simulated.** There was no time to pull and train on the real HN corpus during the hackathon. Only the post *text* leaves the simulator. The extraction, the Reviewer, LightGCN, the cold start, the stats and the backtest all run for real on that text. ***We hope to update this repo as fast as possible once we finish training our dataset. We will improve and maintain this dataset overtime.*** 

## Run it

```bash
uv sync
uv run uvicorn api.main:app --port 8000      # open http://localhost:8000
uv run python scripts/demo_check.py          # end-to-end check of the scripted demo
uv run pytest -q                             # tests
uv run python -m api.ml.backtest             # backtest + ablation (writes data/cache/backtest.json)
make sim                                     # Model & proof panel: real training on the synthetic market (~18 s)
```

The engine builds in about 4 s on a laptop CPU. A reading takes about 0.3 s, well under the 10 s budget.

## The demo, in order
1. **Begin**, then answer the 4 interview questions. Speak (Web Speech API) or type, or press *use the demo founder's answer*. Key phrases settle on screen and the bottle fills.
2. **Let me look.** Company names shuffle past while the open-kitchen feed shows each agent working.
3. **The reading:**
   - **Reflection:** the founder's own words.
   - **The hit:** *all 3 customers hired RevOps in the six months before signing*. The base rate is 1.4% in matched companies, and BH q ≈ 0. The founder never said "RevOps".
   - **The one you didn't see:** *31 logistics companies describe your pain as "freight audit"*. Correct it with "No, they're too big." and the card revises in place and says what it removed.
   - **Needed:** a stranger's dated post.
   - **Blind spot:** where the engine can't see clearly.
   - **The next step:** one account.
4. **Open the engine:**
   - **Accounts:** evidence chips, where every chip cites a post and passed the Reviewer.
   - **Enrich:** Flexprice credits, spent in order of score ÷ cost.
   - **Draft email:** written in the founder's voice, with an AI-sounding check. It needs your approval before it sends.
   - **Reply:** open `/reply/<draft_id>` on a phone and tap Reply. The webhook fires and the card moves to *Replied*.
   - **Bottle:** rules with Beta trust, plus every feature tested.
   - **Band feed:** every agent step.
   - **Model & proof:** say it out loud: *"This panel is the roadmap. The training ran on synthetic data, the voice fine-tune is simulated, and there are no measured numbers yet."*

## What's inside

| Layer | Code | What it does |
|---|---|---|
| Market | `api/market/simulate.py`, `extract.py`, `graph.py` | Seeded market. Dictionary extraction with a negation filter, plus a deliberately noisy simulated "LLM pass". Only Reviewer-approved triples enter the graph. `snapshot(asof)` gives leakage-safe views. `to_cypher()` exports to Neo4j. |
| Text | `api/ml/text.py` | TF-IDF + LSA (eigendecomposition, no downloads), plus a lexical channel through a random projection: a dense + lexical hybrid, like embeddings + BM25. |
| Market model | `api/ml/lightgcn.py`, `market.py` | LightGCN + BPR in NumPy. Propagation is linear, so E = M·E0 and the gradient is exactly Mᵀ·∂L/∂E. The product is held out of training. |
| Cold start | `api/ml/coldstart.py` | A ridge map from description to embedding, then a BPR fold-in on the named customers with an L2 pull toward the prior (MAP), so 3 customers aren't memorized. |
| Ranking | `api/insight/ranking.py` | `s(c) = pct(ŷ) + λ(c)·pct(s_founder)`, with `λ(c) = λ0·exp(−d(c,E)/τ)`. Both terms are converted to percentiles because they're on different scales. |
| The hit | `api/insight/hit.py`, `stats.py` | Matched base rates (size, stage, calendar window), a Poisson-binomial tail, Benjamini–Hochberg over every feature seen, a shrunk lift, and exclusion of anything the founder already said. |
| Discovery | `api/insight/discovery.py` | k-means segments. Keep the top fifth by fit, then take the farthest from E; it needs at least 8 companies with their own quotes. Also the stranger's post and the blind spot. |
| Bottle | `api/bottle/interview.py` | Story-first questions, company resolution, key phrases, pain/trigger/buyer rules with `q ~ Beta(α, β)`. |
| GTM | `api/gtm/` | Accounts with Reviewer-checked chips, outreach in the founder's voice, an approval gate, and Resend sending. |
| Integrations | `api/integrations/band.py`, `flexprice.py` | Real API calls when keys are set; otherwise the same payloads are mirrored locally. |
| API + UI | `api/main.py`, `apps/web/` | FastAPI with SSE; a static full-screen UI with no Node build. |

## Design and open-source UI pieces

Before changing the UI, read [docs/design-guide.md](docs/design-guide.md). It holds the rules for copy, type, color, motion and the checks to run before committing.


The look follows the dark editorial style of startup sites like Monaco and Ornn:
- near-black surfaces
- a serif display face for headlines
- a clean grotesk for UI text
- pill buttons
- a floating nav bar

Those sites use commercial fonts (Season Serif, PP Neue Montreal, FK Roman), so we use open-source counterparts instead:
- **Type:** [Instrument Serif](https://github.com/Instrument/instrument-serif) for headlines, [Inter](https://github.com/rsms/inter) for the UI, and [Fragment Mono](https://github.com/weiweihuanghuang/fragment-mono) for numbers. All three are SIL OFL.
- **Icons:** [Lucide](https://github.com/lucide-icons/lucide) (ISC).
- **Front-flow backgrounds** (`apps/web/hero3d.js`, canvas 2D): the front page shows a bottleneck, where market signals funnel through one glowing neck (your instinct) and fan out to customers. The Problem page shows a fan of rays from one founder to the market. How it works shows an arc dome (`apps/web/dome3d.js`, three.js), where every streak runs from the founder's intuition to one of the 900 companies.
- **Market map:** [three.js](https://github.com/mrdoob/three.js) r170 (MIT), in `apps/web/map3d.js`. All 900 companies are plotted as a 3D point cloud laid out by the model's embeddings (PCA). Each card lights up its evidence on the map, and a scan wave runs while the engine scores companies.

All of these are vendored in `apps/web/vendor/` with their licenses, so the demo needs no network.

## Integrations (all optional; the demo runs offline)

| Env var | Effect |
|---|---|
| `BAND_API_KEY` (+ `BAND_CHAT_ID`) | Posts every agent step to a Band room as an event (`POST /api/v1/agent/chats/{id}/events`). Creates a room per session if no chat id is given. |
| `FLEXPRICE_API_KEY` (+ `FLEXPRICE_BASE_URL`, `FLEXPRICE_EVENT_NAME`) | Creates a customer and a PRE_PAID wallet per session, ingests an `account_enrichment` usage event per enrichment, and reads the real-time balance. |
| `RESEND_API_KEY` + `DEMO_SEND_TO` (+ `RESEND_FROM`) | Approved emails really send, redirected to a teammate's inbox. Without these they go to a local outbox. |

## Proof (simulated market, seed 7)

Train through 2024-12, then predict which of 724 companies adopt Ledgerlight afterwards (69 did). Recall@50, with 95% bootstrap CIs:

| Method | Recall@50 | 95% CI |
|---|---|---|
| random (expected) | 0.069 | — |
| popularity | 0.087 | 0.029 to 0.159 |
| description match (LLM stand-in) | 0.043 | 0.000 to 0.087 |
| market model alone | 0.101 | 0.043 to 0.188 |
| founder alone | 0.029 | 0.000 to 0.072 |
| **market + founder** | **0.145** | 0.072 to 0.232 |

The combination beats both parts on the point estimate, but the intervals overlap, and the UI says exactly that. Most 2025 adopters come from the logistics shift, which neither the market history nor the founder's gut anticipated. That shift is what the discovery card exists to surface.

## Model & proof panel (`make sim` → `sim/out/results.json`)

The panel says once at the top that everything is trained on synthetic data, and badges mark only the exceptions:
- **Synthetic (default, no badge):** real training code ran on the seeded synthetic market. This covers LightGCN training, `logit(founder prior) + correction` across a time split, Thompson sampling, the backtest and ablation, and the checks that planted patterns were recovered.
- **Simulated badge:** the LoRA + DPO voice fine-tune is a seeded mock-up. It was not run.
- **Measured badge:** real public data. There are none yet.

What the synthetic runs show:
- **Market model:** LightGCN only matches the popularity baseline on this market.
- **Prior + correction:** the correction learns the planted RevOps trigger and shifts weight toward logistics. Because the founder's stated beliefs are miscalibrated, the correction alone beats prior + correction.
- **Thompson sampling:** founder priors used as stated cost conversions. Rescaled to the market base rate, they finish best.

## Honest limits
- The market is simulated. Swapping in the real corpus means replacing `simulate.generate()` with a loader for `brusic/hacker-news-who-is-hiring-posts` (updated 2026-11-21). The post → triple → graph path stays the same.
- The interviewer and extractor are rule-based, so they're deterministic and run offline. A Claude-backed extractor can be added behind the same `absorb()` interface.
- The "LLM baseline" is a description-match stand-in. A real LLM baseline must only be scored on months after its training cutoff.
