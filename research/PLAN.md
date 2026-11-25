# PLAN: Prior-Conditioned Relational Foundation Model (PC-RFM)

Working plan to hand to Claude Code. Everything here is a proposal to refine, not settled.
Background and citations: `docs/research-map.md`.

## 1. Research question and hypotheses

Can a model pretrained on simulated markets with simulated founders learn, in context, to
combine founder segment beliefs, a market graph and 0–50 labels into better customer
predictions than labels-only models — and to ignore founders who are wrong?

Pre-register:
- H1: With informative founders, PC-RFM beats labels-only baselines at n ≤ 20; the gap shrinks as n grows.
- H2: With confidently wrong founders, PC-RFM's loss stays within a small margin of labels-only and recovers as labels arrive (learned trust).
- H3: Gains concentrate in "emerging" strata (young sellers, low degree, novel categories) and vanish in mature strata.
- H4: Gains hold over strong baselines (CPR-LP v2, TabPFN, KumoRFM-2 in-context if accessible), not just weak ones.
- H5: Trained only on simulation, PC-RFM transfers to real backtests (USAspending, Amazon new brands) in the emerging strata.

## 2. Simulator (half the paper — build first)

### 2a. Market generator
- Accounts with interpretable attributes: industry, size band, region, age, tech stack, signal events (funding, hiring) with timestamps.
- Segments = attribute combinations; soft membership allowed.
- Heterogeneous graph: account–product adoption (bipartite, preferential attachment), competitor edges, people/employment edges, community structure (degree-corrected SBM).
- True buying propensity for seller u and account v:
  `logit P(v buys u) = η_segment(v) + r_v (graph-smooth latent, GMRF on L) + pair(u,v) + latent_need(v) + noise`
- Knobs (each a sweep axis):
  - **Emergence**: share of variance from *unobservable* latent need vs observable features; whether the "similar-product adopters" path exists (novel category).
  - Segment vs graph share of variance (when segments explain nothing, priors can't help).
  - Homophily vs heterophily of relations.
  - Base rate, market size, seller degree.

### 2b. Founder generator
- Founder sees true segment effects through noise: belief p̃_k with reliability ρ ∈ [−0.5, 1].
- Systematic bias (favours some segments), overconfidence (stated confidence vs actual error), coverage (opinion on only some segments), "no idea" answers.
- Adversarial founders (confidently wrong about the best segment). Multiple correlated co-founders.
- Output in the same format as the real elicitation protocol: per-segment (median, quartiles or counts out of 20, confidence).

### 2c. Label process
- Outreach chosen by the founder's beliefs with exposure knob α (self-confirming loop); positive-unlabeled observations; k labels ∈ {0, 1, 5, 10, 20, 50}.
- Optional: funnel stages with delays; value weights.

### 2d. Calibration to reality
- Fit generator knobs to real distributions: degree distributions and segment concentration of first customers from USAspending (vendor → agency office, first awards) and Amazon Reviews 2023 (new stores → first buyers).
- Report where real data sits in knob space.

## 3. Model

- PFN objective: sample (market, founder, labels) from the simulator; train to predict held-out candidate labels by NLL. At the optimum this approximates the Bayes posterior predictive under the simulator prior (Müller et al. 2022).
- Tokens: candidate account (features + segment memberships + local graph summary), founder-belief tokens (segment predicate embedding, belief, confidence), context-label tokens.
- Graph: start with precomputed graph features (heuristics through product/competitor paths, PPR from labels and from the prior), then a relational graph transformer over sampled subgraphs.
- Staged build:
  - v0: no graph; segments + features + founder tokens + labels (TabPFN-like). Proves learned trust.
  - v1: + graph features.
  - v2: + learned relational encoder.
- **Decision (2026-11-24): the graph side follows relational deep learning (RDL), not knowledge-graph embedding.** The market is a relational database (accounts, products, adoptions, later people/employment and events), and the encoder is an RDL-style heterogeneous GNN / relational graph transformer over sampled subgraphs (RelBench, RelGNN, KumoRFM lineage). KG-embedding methods (TransE-style triple scoring) are out of scope except as baselines if a reviewer asks.
- Prerequisite found in M0/M1: the v0 simulator's account-account graph carries almost no signal at 5–50 labels, so an RDL model has nothing to learn. Simulator v1 must add the relational structure the thesis is about: a product table, an account–product adoption table, and a seller–product similarity, with the emergence knob controlling whether the "similar-product adopters" path exists.
- Also train an ablation without founder tokens (same compute) — the cleanest test of the claim.

## 4. Baselines (equal tuning budget)

Prior-only ranking · global and segment popularity · CN/AA/RA/Katz through product paths · LightGBM and TabPFN on pair features · labels-only GNN (GraphSAGE/RelGNN) · ContextGNN or NBFNet · **CPR-LP v2** (closed-form GMRF posterior with calibrated founder prior; fixed and learned trust variants) · KumoRFM-2 in-context if accessible · PC-RFM without founder tokens.

## 5. Evaluation

- Sanity ladder (stop if a step fails): oracle founder must help → sweep ρ to find break-even → real priors.
- Surfaces: lift vs labels-only over (ρ, k, α, emergence). Report break-even ρ, worst-case loss under adversarial founders, prior washout time.
- Metrics: MAP@10/Recall@K with within-segment hard negatives; lift over popularity; novel-customer recall; calibration (ECE); per-stratum results.
- Real tiers (see research map "Evaluation and data"): USAspending backtests; Amazon new brands (B2C); archival founder priors; chronologically clean LLM priors (ChronoGPT-style); prospective founder cohort.
- Hierarchical bootstrap over sellers; ≥5 seeds; report every run.

## 6. Milestones

- M0 Repo, simulator v0 (market + founder + labels), CPR-LP v2 closed-form baseline; oracle-prior sanity check passes.
- M1 PC-RFM v0 (no graph) trained; show learned trust: helps with good founders, ignores bad ones.
- M2 Graph-aware PC-RFM; emergence-stratified sweeps; beats CPR-LP v2 somewhere predicted.
- M3 Real backtests: USAspending pipeline, Amazon new-brand pipeline; calibrate simulator.
- M4 Real founder priors (archival / prospective); write-up with theorems from CPR-LP v2.

## 7. Open decisions

- Compute budget and hardware.
- Exact simulator family (start simple; realism is the bottleneck).
- Whether to target a NeurIPS Datasets & Benchmarks–style contribution (simulator + benchmark) alongside the method.
