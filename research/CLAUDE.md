# Project: Founder-Prior Relational Foundation Model (research)

Research project by Peem (Stanford CS). Goal: a real ML research paper proposing a new
method for customer discovery for startups and small/medium companies (B2B or B2C) with
very few labels (5–50 customers).

Full research map (thesis, literature, math, sources): `docs/research-map.md`
Live version of that doc: https://claude.ai/artifact/RhgqPzhgoEiU6nQjxfYi2t
Hackathon origin (AI GTM Engine, won): https://claude.ai/artifact/N4Kk66aGDENDDMGeu9z84k
Execution plan for the chosen direction: `PLAN.md`

## Core thesis (do not lose)

Recommendation systems learn from history; an emerging company's most predictive
information is tacit knowledge in its founders' heads. The method must use that knowledge
as a revisable prior, learn how far to trust it, and explore enough to catch it when wrong.

Technical reasons public data is not enough:
1. No history: the company is a near-zero-degree node; future customers are almost never
   within a few hops of its history (near-zero "locality").
2. Too few labels: commercial lead-scoring needs 50–1,000+ outcomes; startups have 5–50.
3. The "similar product" path is empty: in a new category almost nobody has bought anything
   similar, so the strongest public signal doesn't exist yet.
4. Biased labels: outcomes only for accounts the founder chose to contact (positive-unlabeled,
   missing-not-at-random).
5. Popularity bias: recommenders and benchmarks reward the popular head; early customers of
   new products sit in the tail.

Non-technical: decisive info is tacit (interviews, domain experience); customers in new
markets don't yet know they need the product; founders are often wrong (95% of NSF I-Corps
teams changed their business model, most often the customer segment).

Why a new method: today's choice is rules that encode founder beliefs but never learn, or
data-hungry ML that learns but can't use founder knowledge.

## Chosen direction

**Prior-conditioned relational foundation model (PC-RFM).** Train one transformer, PFN-style,
on millions of simulated markets, each with a simulated founder of random reliability. Input:
market graph + founder segment beliefs and confidence + k labelled customers. Output: P(customer)
for remaining accounts. It learns in-context, Bayes-optimally (under the simulator), how much
to trust the founder. Main research risk: simulator realism.

**CPR-LP v2** (Bayesian graph posterior centred on a calibrated founder prior, closed form) is
the principled baseline and source of theorems. See `docs/research-map.md` sections
"Proposed method", the five "Math:" sections, and "CPR-LP v2".

## The two failure modes we must prevent (top priority)

1. Benchmarks biased toward large, mature markets → priors can be right and still score badly.
   Defenses: stratify evaluation by emergence (seller age, degree, category novelty); use
   popularity-cancelling metrics (lift over popularity, within-segment hard negatives,
   "novel customer recall"); headline results on semi-synthetic data.
2. Graph and priors make results worse in every run.
   Defenses: method nests baselines (trust→0, graph weight→0); learned trust; main regime
   n = 0–20 labels; **sanity ladder** — (1) oracle prior in simulation MUST help, else stop and
   debug; (2) sweep prior quality to find break-even; (3) only then real founder priors.

## Research conventions

- Pre-register hypotheses, the expected regime, and analysis before running real-data evals.
- Never fabricate citations or numbers. Many constants in `docs/research-map.md` came through
  summarised PDF reads and some are marked "verify" — check against the source before using.
- Keep Bayesian and frequentist claims separate (Bayesian regret can't express "founder is wrong").
- Time-based splits only; point-in-time features; no leakage. Hierarchical bootstrap over
  companies, not accounts. Report every run, multiple seeds.
- Check EXNEX (Neuenschwander 2016) and amortized data-borrowing work before novelty claims.
- Licensing: G2 forbids ML benchmarking; Crunchbase forbids model training. USAspending is
  public. SAP SALT is fully anonymized (unusable for human/LLM priors).

## Working with Peem

- When teaching a concept: one concept at a time, then stop and ask whether to continue.
- He wants ambitious, novel methods with real math (theorems + proof sketches), but honest
  about risks and what is unproven.
