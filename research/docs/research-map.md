# Founder Priors for Low-Signal B2B Link Prediction — Research Map

Nov 24, 2026 · @Peem Puangmaha

## Bottom line

The claim is defensible and appears genuinely open: no documented method puts revisable founder priors into a graph link predictor, industry tools need 50–1,000+ outcomes before learning anything, and no prior-injection method learns per-segment trust or handles the bias a prior creates by deciding who gets contacted.

- **Strongest new contribution:** learning how much to trust a founder from 5–50 positive-unlabeled outcomes, pooled across segments and founders, plus an identifiability result: without some exploration, a wrong belief about a segment the founder avoids can never be corrected.
- **Algorithm:** a GNN that learns only a correction on top of a calibrated founder prior (CPR-LP), which unifies posterior regularization and residual-on-prior methods.
- **Headline evidence:** a semi-synthetic sweep on a real market graph, showing gain as a function of prior quality, label count and exposure bias. Real-prior tiers use archival founder documents, chronologically clean LLMs and a prospective founder cohort, because hindsight can't be instructed away and masking doesn't stop LLMs recognising companies.
- **Data:** USAspending.gov is the reproducible public backtest; Wayback snapshots of vendor customer pages are the closest to real startup acquisition and a possible benchmark contribution.
- **Biggest risks:** KumoRFM-2 in-context learning may already capture much of what founders know (make it the main baseline); EXNEX must be checked before claiming novelty for per-segment robust trust; and no public dataset has real outreach outcomes.
- **Scope:** small and medium companies and startups, B2B or B2C.

## Core thesis: why emerging companies lack recommendation signal

Recommendation systems learn from history, and an emerging company's most predictive information is tacit knowledge in its founders' heads. So the right method must use that knowledge as a revisable prior, learn how far to trust it, and explore enough to catch it when it is wrong.

**Technical reasons**

1. **No history.** A new company is a node with almost no edges; its future customers are almost never within a few hops of its own history, so collaborative and graph signals have nothing to follow.
2. **Too few labels.** Commercial tools need 50–1,000+ outcomes before learning; startups have 5–50.
3. **The "similar product" path is empty.** Mature markets run on "companies that bought products like yours will buy yours". In a new category almost nobody has bought anything similar, so the strongest public signal doesn't exist yet.
4. **Biased labels.** Outcomes exist only for accounts the founder chose to contact; everything else is unknown, not negative.
5. **Popularity bias.** Recommenders and the benchmarks that grade them reward predicting popular, well-connected entities; early customers of new products are often in the tail.

**Non-technical reasons**

- The decisive information (customer interviews, domain experience, product insight) is tacit and not in public data.
- In new markets customers often don't yet know they need the product, so public signals lag real demand.
- Founders are often wrong too (95% of NSF I-Corps teams changed their business model, most often the customer segment), so their beliefs can't simply be trusted.

**Why a new method.** Today's choice is binary: hand-written rules that encode founder beliefs but never learn, or data-hungry ML that learns but can't use founder knowledge and needs data the company doesn't have.

## Risks and defenses

**Fear 1: benchmarks are biased toward large, mature markets.**

- Make *emergence* an evaluation axis: stratify by seller age, seller degree and category novelty, and pre-register that priors help in emerging strata and are neutral in mature ones. The crossover is the result.
- Use popularity-cancelling metrics: lift over a popularity baseline, within-segment hard negatives, and "novel customer recall" counting only customers outside the popular head and the seller's neighbourhood.
- Build emerging-market tasks from real data (USAspending vendors entering codes with few prior vendors; Amazon new brands in young categories).
- Put the headline result in semi-synthetic data, where market structure is controlled and benchmark bias can't hide the effect.

**Fear 2: graph and priors make results worse in every run.**

- The method nests the baselines (trust → 0 is data-only; graph weight → 0 is a plain segment model). With learned trust, worst case ≈ baseline + small bounded overhead.
- Test where gains are expected: published prior gains concentrate at 1–20 labels and fade by \~100. Make n = 0–20 the main result.
- Sanity ladder, in order: (1) an oracle prior in simulation must help, or stop and debug; (2) sweep prior quality to find break-even; (3) only then use real founder priors.
- Learn the graph weight and ablate it; "the graph helps only when segments explain little" is still publishable.
- Pre-register the expected regime so positive results there count as evidence, not cherry-picking.

## Direction chosen: prior-conditioned relational foundation model

Train one transformer on millions of simulated markets, each with a simulated founder of random reliability (sharp, overconfident, or wrong). Given the market graph, the founder's beliefs and confidence, and a few labelled customers, it predicts the remaining customers, and learns in context how much to trust the founder. Prior-data fitted networks approximate the exact Bayesian posterior when the training distribution is right, so it learns Bayes-optimal founder trust with no hand-derived updates. Nothing like this exists for relational data. The training distribution can include emerging markets and unreliable founders, addressing both fears. The main research risk is the simulator: a realistic market-and-founder generator calibrated to real data is half the paper. CPR-LP v2 serves as the principled baseline.

## The problem

Small and medium companies and startups (B2B or B2C) must find customers before they have the data that recommendation systems need, so public signals alone under-determine their ideal customer profile.

- **Few labels.** A young company has roughly 5–50 customers, far too few to train a model on its own outcomes.
- **Weak public signal.** Public data (firmographics, hiring, funding, tech adoption, product categories) captures who *could* buy, not who *will*. In new or niche markets, the signals that matter may not be observable at all.
- **A cold-start node.** Structurally, the company is a new, low-degree node attached to a large public market graph. Predicting its next customers is cold-start link prediction.
- **Biased labels.** Outcomes exist only for accounts the company chose to contact or market to, so unlabeled ≠ negative.
- **Founders know things the data doesn't.** Founders hold beliefs about which segments will buy, from customer interviews, domain experience and product insight. Those beliefs are informative but sometimes wrong.

**Claim to test:** encoding founder intuition as priors over interpretable segments, and learning how much to trust them, improves customer prediction in this low-data regime while keeping what graph models do well. The gain should shrink as data accumulates, and confidently wrong priors should be detected and down-weighted.

## Injecting priors while keeping GNN behaviour

Segment-level beliefs ("segment k converts at about p\_k") are statements about outputs, so they belong in function space, not weight space. Two classic families do this well at very few labels, and both reduce to a per-segment logit offset on top of an unconstrained GNN.

**Mechanisms compared**

| Mechanism | Where the prior enters | Handles wrong priors? | Evidence |
| --- | --- | --- | --- |
| Posterior regularization / generalized expectation (Ganchev 2010; Mann & McCallum 2010; Druck 2008) | Expected outputs over unlabeled accounts in a segment | Slack or fixed weight; no learned trust | Strongest at 1–10 labels: 37.3 vs 24.9 with one label (CITE); gains vanish by \~100 labels |
| Residual on a prior model (APHYNITY, ICLR 2021; Correct & Smooth) | Prior as base logit; network learns a minimum-norm correction | Implicitly; residual size measures how wrong the prior is | Strong in physics; C&S needs dense labels |
| Logic distillation / semantic loss (Hu 2016; Xu 2018) | Teacher projection or constraint probability | Fixed confidences | +2–3 points at low data; suits hard exclusions only |
| Functional priors for BNNs (Sun 2019; Tran 2022; Rudner 2022) | Distribution over outputs on context points | Fixed KL weight | Medium; none on graphs |
| Dirichlet evidence on graphs (Graph Posterior Network, 2021) | Prior pseudo-counts + diffused evidence | Prior dominates only when evidence is low | Robust under perturbation; flat priors only |
| Commensurate / power priors (Hobbs 2011) | Hierarchical shrinkage with learned precision | Yes, explicitly; residual harm at small n | Canonical learned trust; not on graphs |
| Weak-supervision source models (Snorkel) | Learned accuracy per labeling source | Yes, per source | Strong; useful for several founders or advisors |
| Prior-conditioned PFNs (PFNs4BO, ICML 2023) | Belief and confidence fed as inputs | Amortized; qualitative evidence only | The only amortized trust-in-user-belief method found |
| Link-MoE gating (NeurIPS 2024) | Per-pair weights over predictors | Gate learns to ignore an expert | Strong on OGB link prediction |

**Where existing methods break in our setting**

1. **Cold-start source.** The startup node has no edges, so propagation methods have nothing to spread; the prior must act on the candidate side.
2. **No learned per-segment trust.** Posterior regularization, generalized expectation, logic distillation and functional priors all fix the prior's weight.
3. **Positive-unlabeled labels with unknown class prior.** The non-negative PU estimator (Kiryo 2017) needs exactly the segment base rates the founder is guessing.
4. **Selection bias created by the prior.** Founders contact the segments they believe in, so a wrong belief about a never-contacted segment can never be falsified. No paper found addresses this.
5. **What the founder's number means.** "70%" is P(buy | contacted, segment), not the marginal link probability; a contact-propensity term is needed.
6. **Fuzzy segments.** Membership comes from noisy public data, so it must be soft.

Gaps 2 and 4 are the most defensible places for a new algorithm; the proposals are in *Proposed method*.

## Few-shot link prediction and graph topology

A startup is a node with near-zero locality: its future customers are almost never within a few hops of its own history. Pair-wise GNNs therefore lose their edge, and segment-level "exploratory" scoring carries the signal — which is exactly where founder priors attach.

**What the literature shows**

- **Locality decides which model wins.** On RelBench recommendation, ID-GNN dominates entity-specific tasks (e.g. hm 2.81 vs GraphSAGE 0.80 MAP) while two-tower wins where popularity beats past visits (amazon 0.74 vs 0.10). ContextGNN measured locality at roughly 0.001–0.002 on rel-amazon and gained 42–80% over NBFNet on the lowest-locality tasks by blending pair-wise and exploratory scores per user.
- **Relational foundation models are untested in our regime.** KumoRFM-2 beats supervised RelGNN on classification (79.60 vs 78.06 AUROC) and accepts user-supplied context tables, but reports no recommendation results and nothing below \~1k context labels. OpenRFM's smallest context is 256.
- **Heuristics matter in sparse graphs.** Under hard negatives (HeaRT), Katz and resource allocation beat NCN, BUDDY and SEAL on ogbl-collab. GNNs cannot count common neighbours, so explicit heuristic features should be added.
- **Few-shot graph link prediction exists but is narrow.** GILT reaches 67.83 Hits@K at 5-shot on ogbl-collab vs 58.00 for UniLP; Meta-Graph meta-learns across ego-networks. Both are homogeneous and static.
- **Exposure bias has a direct precedent.** Gupta et al. (ICML 2021) learn a segment-level exposure-propensity matrix; MAP rose from 41.87 to 59.90 on semi-synthetic data.

**Topology to design for**

1. **Ego node, near-zero locality.** Score by startup attributes × account representation learned across the public graph.
2. **Company–product adoption backbone.** The key path is startup → similar or competing product → its adopters. Compute heuristics through product and competitor nodes, not the startup node.
3. **Buying committees as small cliques** on accounts: aggregate to account level, use degree-aware aggregation.
4. **Temporal event signals** (funding, hiring, adoption) with label windows matched to sales cycles.
5. **Exposure-driven positive-unlabeled labels:** the founder's beliefs both guide outreach and generate the labels.

**Openings this creates**

- No relational foundation model reports in-context recommendation or results below 256 labels.
- No work seeds relational in-context learning with prior-derived pseudo-examples.
- No work uses expert beliefs as the exposure propensity for de-biasing.
- A per-startup trust weight between prior and graph model is the natural analogue of ContextGNN's per-user familiar-vs-exploratory gate.

**Baselines reviewers will require:** global and segment popularity; CN/AA/RA/Katz through product paths; LightGBM and TabPFN on pair features at n = 5–50; RDL GraphSAGE, ID-GNN, RelGNN; NBFNet and ContextGNN; NCN/NCNC; KumoRFM-2 in-context; GILT or Meta-Graph; cold-start methods (NodeDup, Cold Brew); PU/IPS variants; prior-only and fixed-weight fusion.

## Eliciting priors and learning how much to trust them

Existing trust mechanisms learn trust one parameter at a time from a decent-sized dataset. With 5–50 labels spread across hundreds of segments, per-segment trust is unlearnable; trust must be pooled across segments and across founders.

**What the evidence says**

- **Experts and LLMs are overconfident.** Typical 90% intervals capture only about 40–60% of outcomes (cited in Clemen & Lichtendahl). Some LLMs gave Beta priors worth 1,000+ pseudo-observations (Selby et al.). In Agentic-Priors the LLM's stated uncertainty was too narrow to use.
- **Fixed-strength priors can hurt at scale.** A NAVER WEBTOON A/B test with \~595k users per arm found an LLM-derived Beta prior lifted CTR 9.5% on cold items but cost 5.7% overall.
- **LLM priors help most at very few labels.** AutoElicit: \~10× lower error at n = 5, parity by n = 20. Agentic-Priors: a prior worth about 6–40 surveyed users.
- **Per-source trust is unidentifiable at small n.** Commensurate priors gave a 95% posterior interval for log τ (the commensurability precision) of (1.59, 82.00) in the case study with 171 historical and 270 current observations (Hobbs et al. 2011, Table 3; corrected 2026-11-24 after reading the paper — an earlier note said "60 historical points", which is the simulation setting); power-prior exponents go bimodal at N = 100–200.
- **Calibration can be learned as a slope.** Extremizing aggregated forecasts improved Brier scores 21–29% in the Good Judgment Project (Baron et al. 2014); a founder's logit-scale slope is the same parameter.
- **Founders' stated rankings may not match their real preferences** (OPEN, 2024), so pairwise consistency checks matter.

**Elicitation protocol for founders**

1. **Ask about observables.** "Of 20 companies in this segment we contact this quarter, how many book a qualified meeting?" Name the funnel stage.
2. **Factorise segments.** Main effects (industry, size, trigger signal) plus at most 5–10 founder-nominated combinations.
3. **Use SHELF order against anchoring.** Plausible limits first, then median, then quartiles; always allow "no idea".
4. **Add 5–10 seed questions** with known answers (own reply rates, public benchmarks) to initialise each founder's calibration.
5. **Control anchoring.** Never show the LLM's or co-founder's answers first; randomise order; repeat a few questions to measure noise; add pairwise comparisons.
6. **Fit a Beta and cap its strength** at about 10–20 pseudo-observations; the trust model decides how much is really justified.
7. **Pre-register and timestamp** before any labels are seen. This is what makes the PAC-Bayes argument below legitimate.
8. **LLM priors:** many paraphrased prompts, mean from the mixture, spread from disagreement across prompts. Ignore the LLM's self-reported uncertainty.

**Two trust-learning ideas**

- **Pooled calibration.** Model each source's reported rate as logit p\_g = a\_f + b\_f · logit(m\_fg) + noise, with (a\_f, b\_f, σ\_f) shared across all segments and drawn from a population of founders. Trust information then grows with *total* labels, not per-segment labels. Ancestor: Clemen & Lichtendahl's hierarchical calibration.
- **Feature-dependent robust mixture.** Mix founder, LLM and reference components per segment with weights that depend on segment features ("knows fintech, not healthcare"), fitted by marginal likelihood. Ancestors: robust mixture priors and EXNEX (Neuenschwander 2016, not yet verified, must be checked before claiming novelty).

Both are developed into full algorithms in *Proposed method*.

## Industry practice

Commercial predictive scoring switches on only at roughly 50–1,000+ labelled outcomes. Below that, founder beliefs enter as hard rules, manual points or LLM prompts, never as priors that data can revise. No vendor documents founder priors inside a graph or link-prediction model.

**Minimum data before predictive scoring works**

| Tool | Minimum | Below the minimum |
| --- | --- | --- |
| [Salesforce Einstein](https://resources.docs.salesforce.com/242/latest/en-us/sfdc/pdf/sales_einstein_implementation_guide.pdf) | 1,000 leads, 120 conversions per segment | Global model pooled across customers |
| [MadKudu / HG Insights](https://help.hginsights.com/docs/customer-fit-training-and-validation-datasets) | 1,000 records, 200 conversions | Manual points; hard overrides "regardless of what the historical data says" |
| [Klaviyo](https://help.klaviyo.com/hc/en-us/articles/360020919731) (B2C) | 500 ordering customers, 180 days | "Does not qualify" |
| [Meta lookalikes](https://developers.facebook.com/docs/marketing-api/audiences/guides/lookalike-audiences) | 100 seed people per country | Platform model |
| [Demandbase](https://support.demandbase.com/hc/en-us/articles/360053937452) | 50 accounts, 100+ recommended | Lookalike against a hand-picked training group |
| [HubSpot AI-suggested scores](https://knowledge.hubspot.com/scoring/understand-the-lead-scoring-tool) | 50 contacts (25 converted) | Manual scoring |
| [ZoomInfo ICP Builder](https://pipeline.zoominfo.com/marketing/icp-builder) | \~100 closed-won suggested | Under 20 customers: "build a hypothesis ICP" |

**Four cold-start patterns**

1. **Pooled models** on other companies' data (Salesforce, Shopify, Meta, Google, KumoRFM): need a big platform; the seller can't express beliefs.
2. **Seed lookalikes** (Ocean.io from one URL, Demandbase, Meta): assume early customers are representative and that similarity is the right relation.
3. **Manual rules and weights** (HubSpot, Clay, Common Room, PostHog): encode intuition fully, learn nothing.
4. **LLM judgment over text** (Claygent, Pocus, Apollo): intuition goes in as a prompt; no vendor publishes calibration.

**Closest analogue: consumer ad platforms.** Google treats advertiser segments as "an informed starting point" that the model may stop serving, and uses landing pages and creative as product content. Meta treats targeting as "suggestions". Both are opaque, rely on huge pooled data, and aren't graph models of a seller's own market. Claim "no documented B2B or graph method does this", not "nobody does this".

**Founder intuition is informative but often wrong**

- 95% of 230 NSF I-Corps teams changed at least one business-model component; customer segments changed most often.
- In a randomised trial (Camuffo et al. 2020, n = 116), training founders to treat beliefs as testable hypotheses increased pivoting and revenue.
- 81% of founders rated their own success odds at 70%+, versus 39% for comparable firms (Cooper et al. 1988).
- Retool's first two ICP hypotheses were "dead wrong" (First Round Review).

This supports the paper's framing: the method operationalises the "scientific approach" to entrepreneurship by turning founder beliefs into priors that evidence can revise, with learned trust and explicit wrong-prior tests.

## Evaluation and data

The "turn back the clock" test is feasible, but only as one tier of four. Each tier controls a different bias, and the headline claim should rest on the tiers where hindsight and LLM leakage are impossible by construction.

**Why naive backtests fail**

- **Human hindsight cannot be instructed away.** Fischhoff (1975): knowing an outcome roughly doubled its perceived odds, and telling people to answer "as if you didn't know" did not remove the bias. Blinding must be informational, not instructional.
- **LLMs leak the future, and masking doesn't fix it.** GPT-4 re-identified firms from name-censored earnings calls 70% of the time (Sarkar & Vafa). Fine-tuning a modern model on old data still leaks (Time Machine GPT). Date-filtered web search leaked a strong signal for 71% of questions (El Lahib et al. 2026).
- **Entity knowledge distorts in both directions.** Anonymized headlines *outperformed* named ones in Glasserman & Lin, a "distraction effect".
- **Offline replay needs randomized logs** (Li et al. 2011). Sales logs aren't randomized, so labels are positive-unlabeled with attribute-dependent exposure.

**Four-tier protocol**

1. **Semi-synthetic sweep on a real graph (main quantitative tier).** Real market graph and features; simulated conversion surface; simulated priors with controlled correlation to truth ρ ∈ {−0.5 … 1}, miscalibration, coverage and adversarial cases; outreach driven by the prior itself with an exposure knob α. Report lift as a surface over (ρ, labels 5–50, α), the break-even ρ, worst-case loss and unlearning speed. Template: ACIC 2016 causal-inference challenge and Schnabel et al. 2016.
2. **LLM priors under time control.** Primary arm: chronologically consistent models (ChronoGPT, Time Machine GPT, DatedGPT) with cutoff ≤ T0. Frontier LLMs only anonymized, with a re-identification probe, a Min-K% contamination covariate (Gao et al. 2025), before/after-cutoff splits and a frozen T0 corpus, no live search.
3. **Human priors with blinding.** (a) Archival priors coded from founder documents dated before T0 (pitch decks, launch posts, Wayback pages): recorded foresight. (b) Blinded expert raters with recognition screening. (c) A prospective cohort: pre-register current founders' priors now, score 6–12 months later. Small N, but leakage-free.
4. **Retrospective case studies.** 3–5 narratives chosen by a pre-registered rule; illustrative only.

**Cross-tier rules:** sampling frame fixed at T0 including failures; point-in-time features only, with a "today features" leakage ablation; hierarchical bootstrap over companies, not accounts; HeaRT-style hard negatives scored both across and within segments; pre-registration of frame, segments, metrics and analysis.

**Datasets**

| Source | What it gives | Fit | Main risk |
| --- | --- | --- | --- |
| [USAspending.gov](https://api.usaspending.gov/docs/endpoints) | Timestamped agency → vendor awards since FY2001, small-business flag, NAICS codes, readable; subawards give firm-to-firm links | Best fully public, reproducible backtest | Government buying ≠ commercial sales |
| Wayback snapshots of "customers" / logo pages | Vendor → customer edges dated by first capture | Closest to real startup acquisition; a new benchmark contribution | Logos are marketing claims; capture lag; heavy LLM memorization; legal review |
| [SupplySim](https://github.com/snap-stanford/supply-chains) (extended) | Simulator with known buying rules | Ground truth for prior-quality sweeps | Toy scale (119 firms) |
| Amazon Reviews 2023 by store; Olist | Small seller → first buyers, timestamped | B2C variant | Customers anonymous, so priors act only on product side |
| rel-salt (SAP) | Real sales orders | Not usable for priors | Fully anonymized, single seller |

Ruled out for release: G2's terms forbid ML benchmarking; Crunchbase's terms forbid model training.

## Proposed method

The flagship algorithm is a graph link predictor that learns only a correction on top of a calibrated founder prior, pools trust across segments and founders, and corrects for the bias created when the prior decides who gets contacted. Working name: **Calibrated Prior-Residual Link Prediction (CPR-LP)**.

**The model, in four parts**

**1. Calibrated prior.** Each source f (founder, co-founder, LLM) reports a rate m\_fg per segment g. A pooled calibration model maps reports to beliefs:

```latex
\operatorname{logit} p_g = a_f + b_f\,\operatorname{logit}(m_{fg}) + \varepsilon_{fg},\qquad (a_f, b_f, \log\sigma_f) \sim \mathcal{N}(\mu_{\text{pop}}, \Sigma_{\text{pop}})
```

Trust parameters are shared across all segments, so they are learnable from 5–50 total labels; a new founder inherits the population prior.

**2. Residual scoring.** With soft segment membership m\_k(v) for each account v, the GNN learns only the correction, kept small unless data demands it (APHYNITY-style):

```latex
s(u,v) = \lambda\,\ell_{\text{prior}}(v) + f_\theta(u,v),\qquad \ell_{\text{prior}}(v) = \operatorname{logit}\Big(\textstyle\sum_k m_k(v)\,\tilde p_k\Big)
```

This equals the closed-form posterior-regularization teacher, so it unifies the two strongest existing families. The size of f\_θ on each segment is a built-in "how wrong was the founder" diagnostic.

**3. Positive-unlabeled loss with segment class priors.** Use the non-negative PU estimator with a segment-conditional class prior from the calibrated beliefs times a contact propensity. The founder supplies exactly the quantity PU learning normally lacks.

**4. Selection-aware outreach.** Contacts are chosen by Thompson sampling over segment posteriors with a minimum exploration rate η. Known propensities make trust identifiable, and outcomes are reweighted by inverse contact propensity.

**Claims that could become theorems**

- **Effective sample size.** For segment k with prior bias δ\_k and n\_k labels, the Beta–Binomial posterior's error is minimised at prior strength τ\* = p(1−p)/δ², giving an effective sample size of n\_k + τ\*: the gain is an explicit function of prior quality. (Standard algebra; state it as such.)
- **Bounded harm.** With a reference component of weight w\_ref in the prior mixture, the PAC-Bayes bound loses at most log(1/w\_ref)/n against the no-prior bound when the founder is wrong, and tightens when the founder is right. The prior must be elicited before data for this to hold.
- **Identifiability (the most novel result).** Without exploration, trust in a segment the founder dislikes is not identifiable, because that segment is never contacted. With exploration rate η, trust estimates are consistent at rate O(1/√(ηn)). A clean negative-plus-positive result.
- **Decision loss.** Thompson sampling with a misspecified prior loses reward bounded by the total variation to the true prior (Simchowitz et al. 2021); learned calibration shrinks that distance as labels accrue.

**Alternative, more ambitious design.** A **prior-conditioned relational foundation model**: pretrain a relational in-context learner on synthetic graphs where each task also samples a founder belief of random reliability, and feed the belief as tokens (generalising PFNs4BO). At the optimum this implements exact hierarchical-Bayes trust. Highest upside, highest compute; a good second paper.

**Hypotheses to pre-register**

1. **H1:** With good priors, CPR-LP beats labels-only models at n ≤ 20, and the gap shrinks as n grows.
2. **H2:** With confidently wrong priors, a fixed-weight prior hurts while CPR-LP's loss stays within the bounded-harm envelope and recovers.
3. **H3:** Gains hold over strong few-shot baselines (KumoRFM-2 in-context, TabPFN, ContextGNN), not just weak ones.
4. **H4:** Without exploration, trust in disliked segments never corrects; with η > 0 it does, matching the identifiability result.
5. **H5:** Real archival, chronologically clean LLM and prospective founder priors land where their measured quality predicts on the semi-synthetic map.

**Core ablations:** prior-only; GNN-only; fixed-weight fusion; posterior regularization without residual; per-segment vs pooled trust; with vs without PU loss; with vs without exploration; where the prior enters (offset, loss, pseudo-labels, input feature).

## Math: graph diffusion with the founder prior as a source

Treating the founder prior as a distributed source in a graph diffusion, and labels as point sources, gives a closed-form steady state that is exactly a Bayesian posterior under a Gaussian Markov random field, cannot oversmooth, and has explicit formulas for how far each label overrides the prior.

**Prior-sourced graph diffusion (new component).** With latent account logits x, heterogeneous Laplacian L, prior precisions B = diag(β\_v), label precisions J on contacted accounts and targets ỹ:

```latex
\frac{dx}{dt} = -\gamma L x - B\,(x - \ell - f_\theta) - J\,(x - \tilde y),\qquad x^* = M^{-1}\big[B(\ell + f_\theta) + J\tilde y\big],\quad M = \gamma L + B + J
```

- **Bayesian reading.** x\* is the posterior mean under a GMRF prior with precision γL + B centred on the founder prior, with Gaussian label likelihood; M⁻¹ is the posterior covariance.
- **Random-walk reading.** A walker from v either moves to a neighbour, is absorbed at the founder prior, or is absorbed at a label; x\*\_v is its expected payoff. This generalizes the "dongle" nodes of Zhu, Ghahramani & Lafferty (2003). The probability of absorption at the prior, ω\_prior(v), replaces the draft's single global λ with a node-adaptive trust.
- **Precedents.** It is APPNP/personalized PageRank teleporting to the founder prior instead of the seed node, and GRAND++ / Poisson learning (Thorpe et al. 2022; Calder et al. 2020) with the prior as a distributed source. BiRank (He et al. 2017) gives the bipartite analogue for the company–product backbone.
- **Why β must be positive.** With fixed labels and growing graphs, pure Laplacian regularization is ill-posed and degenerates to spikes plus a constant (Nadler et al. 2009; Calder et al.: Laplace 16.1% vs Poisson 90.2% on MNIST at one label per class).

**Segment–graph hierarchical GMRF (new component).** Write x = Φη + r: segment effects η centred on calibrated founder beliefs, plus a graph-smooth residual r. Labels correct segment beliefs globally and residuals locally, with closed-form posteriors and empirical-Bayes learning of trust.

**Candidate theorems** (agent's derivations, proof sketches in the report)

1. **Well-posedness, maximum principle, no oversmoothing.** M is a positive-definite M-matrix, so x\* is unique and a convex combination of prior and label values; the GNN residual is 1-Lipschitz through the diffusion; x\* never collapses to a constant (unlike Cai & Wang's exponential energy decay).
2. **Explicit locality.** (M⁻¹)\_vw ≤ ρ^d(v,w)/((1−ρ)s\_w) with ρ = max γd\_w/s\_w < 1. Each label overrides the prior within O(log(1/ε)/log(1/ρ)) hops; larger β shrinks the footprint.
3. **Prior influence decays with labels and distance.** ω\_prior(v) ≥ 1 − ρ'^d(v,J), and prior-induced error at v ≤ ω\_prior(v)·‖e‖\_∞.
4. **Smoothing can't fix graph-coherent errors.** A prior error that is smooth on the graph and sits in never-contacted segments passes through almost unattenuated: ‖βM⁻¹z‖ ≥ β/(β+R'(z))·‖z‖. This is the spectral form of the never-contacted-segment result.
5. **Effective label count.** Wrong-prior weight on segment k is 1/(1 + σ\_k² n\_k^eff). Labels clustered in one graph region saturate n\_eff (k labels ≈ one), a principled reason to diversify outreach. A never-contacted segment is identifiable only through the founder's pooled trust.
6. **Monotone improvement.** Each new label decreases every posterior variance; expected squared error is non-increasing in the label set.

**Pitfalls:** heterophilous relations (competitors, churned-from) need learned non-negative relation weights or a signed sheaf Laplacian (Bodnar et al. 2022), at the cost of the maximum principle. Use the full heterogeneous Laplacian (or its Kron reduction), not adjacency powers, on bipartite structure. Theorem 3's label-count bound assumes labels independent of the graph, which adaptive outreach violates. M is symmetric diagonally dominant, so preconditioned conjugate gradient scales; each Thompson sample costs one solve via perturb-and-MAP.

## Math: Gaussian processes on graphs and calibrated uncertainty

A Gaussian-process version of CPR-LP puts the founder prior in the GP *mean* and graph structure in the *kernel*. The posterior is then an explicit shrinkage from the founder's belief toward the data, trust is a regression coefficient integrated out in closed form, and Thompson sampling gets principled uncertainty.

**Model.** For seller u and candidate accounts v, with calibrated founder logits ℓ\_f(v):

```latex
f(v) = \underbrace{a + \textstyle\sum_f b_f\,\ell_f(v)}_{\text{trust-weighted founder mean}} + g(v),\qquad g \sim \mathcal{GP}(0, k_\theta),\qquad b_f = \bar b + \xi_f
```

The kernel is a sum of interpretable parts: a segment residual (Σ\_k τ\_k² m\_k(v)m\_k(v'), with τ\_k shrinking as founder confidence rises), a Matérn kernel on the account graph (Borovitskiy et al. 2021; sparse and scalable for integer smoothness), a distance-aware embedding kernel from a *frozen* or weight-free GNN (spectral normalization as in SNGP, or the infinite-width GNN-GP kernel of Niu et al. 2023), and a nugget.

**PU likelihood via Pólya-Gamma augmentation (new component).** Known customers, outreach rejections and unlabeled accounts with latent true labels are handled with Pólya-Gamma variables (Wenzel et al. 2019), giving closed-form mean-field updates. The posterior mean is then GP regression on pseudo-targets, centred on the founder prior:

```latex
\mu = m + K\,(K + \bar\Omega^{-1})^{-1}(z - m),\qquad \Sigma = K - K\,(K+\bar\Omega^{-1})^{-1}K
```

In each spectral direction j, the prior is replaced by data in proportion to s\_j = λ\_j/(λ\_j + 1/ω̄). At segment level this is James–Stein shrinkage: the founder's belief is overridden at speed n\_k ω̄ τ\_k².

- **Automatic down-weighting.** Trust coefficients have a closed-form GLS posterior. If a founder's logits carry no information, b̂\_f → 0 and the mean collapses to an intercept.
- **The prior fixes a PU identifiability gap.** In standard PU learning the labeling frequency and the class prior can't be separated from positives and unlabeled data; the founder's segment rates supply exactly the missing constraint.
- **Tempering for misspecification.** Choose a posterior temperature by SafeBayes (Grünwald & van Ommen 2017); empirical-Bayes temperature fails under misspecification, and warm-intro customers make PU labels non-random.

**Cold-start pairwise kernel (new component).** When pooling across many sellers: k((u,v),(u',v')) = k\_V(v,v')·\[ρ + (1−ρ)k\_B(b\_u, b\_u')\] + δ\_uu'·k\_V^own(v,v'), where b\_u is the seller's belief vector plus product features. A brand-new seller still gets predictions from sellers with similar beliefs; the kernel is universal if its factors are (Stock et al. 2018).

**Calibrated shortlists.** Positive-calibrated conformal recall: holding out known customers gives a threshold that captures a new true customer with probability ≥ 1−α. With 20 customers the best achievable is about α = 5%. Under adaptive Thompson sampling, use selection-weighted quantiles.

**Candidate theorems**

1. **Exact bias–variance decomposition.** Expected error = ‖(I−S)(m−f₀)‖² + variance; after marginalising trust, only the part of founder error that calibration can't absorb remains.
2. **Contraction depends on prior error.** Rate ε\_n = max(ε\_n⁰, ‖f₀−m‖\_H/√n): confidently wrong beliefs (small τ\_k, large error) are what slow learning. On a finite graph the asymptotic statement is vacuous; the useful content is the finite-sample constant.
3. **Bounded harm in cumulative log-loss.** Against any comparator, including "ignore the founder", the Bayesian mixture over trust loses at most a quadratic penalty plus O(d log n) nats; a good founder gains linearly in time.
4. **Thompson-sampling regret.** Õ(√T(‖f₀−m‖\_H√γ\_T + γ\_T)): prior quality improves regret through the RKHS-norm term (Chowdhury & Gopalan 2017; logistic constants still to verify).
5. **Reversion to the founder prior off-support.** With a bi-Lipschitz embedding, predictions far from labeled data return to the calibrated founder mean, which is minimax under Brier score (SNGP Prop. 2).

**Pitfalls:** don't train GNN weights by marginal likelihood on 5–50 labels; deep kernel learning overfits worse than a plain network (Ober et al. 2021). The seller node has no edges, so put graph structure on the account side. Pólya-Gamma mean-field underestimates variance; check against Laplace or EP before trusting Thompson draws. Cost: O(nM² + M³) with inducing points; avoid dense O(N³) graph kernels beyond \~10⁴ nodes.

## Math: bandits and guarantees that accuracy improves over time

In the sequential version the founder's weight is not a constant: it emerges from the posterior per segment and decays like 1/(labels in that segment). Regret then depends on a confidence-weighted KL between founder beliefs and the truth. Without exploration, a confident pessimistic founder causes linear regret; with exploration the method is consistent and the prior washes out at a computable time.

**Emergent, decaying prior weight.** For segment k with prior spread ω\_k and delay-corrected label count Ñ\_k(t):

```latex
m_{t,k} = \lambda_{k,t}\, b_{0,k} + (1-\lambda_{k,t})\,\bar y_k,\qquad \lambda_{k,t} = \frac{\omega_k^{-2}}{\omega_k^{-2} + \tilde N_k(t)/\sigma^2}
```

A constant λ is exactly what lets a wrong prior dominate forever, so the draft's global λ should be replaced by this. Founder trust (a\_f, g\_f) enters linearly, so it adds just 2F Gaussian dimensions (the hierarchical-bandit structure of Basu et al. 2021 and Hong et al. 2022); g\_f ≤ 0 can flag an anti-informative founder.

**Algorithm sketch (CPR-TS).** Each period: add delay-corrected pseudo-observations for each funnel stage (Vernade et al. 2017); update the posterior; draw one sample; fill a batch of k contacts by top-k of the sample, except with probability η pick a segment stratified toward under-explored ones; log every contact propensity for off-policy evaluation; refit the frozen GNN features periodically.

**Candidate theorems**

1. **Bayesian regret with batching and delay.** BR(T) ≤ 2√(2 log(NTk) · n · C\_k,G · c\_σ · 2γ\_n) + ηnΔ\_max, where the information gain γ\_n splits into segment, trust and graph parts. Confident founders make the segment part near zero, and a more confident prior also *reduces* the penalty for batching and slow retention feedback.
2. **Frequentist regret under a wrong prior.** R(T) = Õ((√γ\_n + D₀)√(n C γ\_n)) with D₀² ≈ 2Σ\_k c\_k·kl(p\*\_k ‖ p̃\_k): a *confidence-weighted KL* between founder beliefs and truth. Wrong but humble founders cost little; confident and wrong ones cost a lot. Corollary: regret is minimised when the prior's spread equals the founder's actual error, which justifies calibration.
3. **Bounded harm, stated honestly.** For prediction, a mixture of informed and uninformed predictors has log-loss ≤ min(informed, uninformed) + ln 2 on any data stream, and Fixed Share tracks a founder whose reliability drifts. For decisions, a free "minimum of both regrets" is impossible (Lattimore 2015, Pareto frontier); the provable claim is that a wrong prior inflates regret by at most a factor (1 + D₀/√γ\_n) and never makes it linear when η > 0.
4. **Lower bound without exploration (identifiability).** A segment never contacted keeps its prior variance unless coupled through graph or trust. With greedy selection and a pessimistic prior on the best segment, regret is linear. With Thompson sampling it is linear until a horizon ≈ e^(z²/2), exponential in the squared standardized prior error (consistent with Liu & Li 2016; Wang & Chen 2018 publish the same failure for combinatorial TS).
5. **Consistency and prior washout.** With stratified η-exploration, segment estimates converge at O(1/√(ηn)); the prior-bias term decays like 1/t, faster than noise. This gives an explicit *prior washout time* worth reporting. Per-account expected error is non-increasing in time, and exploration bounds IPS weights so off-policy evaluation needs no clipping.

**Pitfalls:** Bayesian regret assumes truth is drawn from the prior, so it can't express "the founder is wrong"; wrong-prior claims need the frequentist D₀ bound. Accounts are consumed once contacted, so learning flows only through shared structure. Rare conversions inflate logistic constants. Retention delays can exceed the horizon, so use early funnel stages as surrogates. At 5–50 labels every bound is vacuous in its constants: support claims with simulations of washout time, D₀ sweeps and regret with vs without exploration.

## Math: optimal transport, information geometry and exposure correction

Two results carry this section. First, an unbalanced optimal-transport layer gives trust a precise meaning: a mirror-descent step size toward the founder, with a closed form and an exact harm bound. Second, a doubly robust positive-unlabeled loss turns founder-prior errors into variance, never bias.

**Segment-marginal layer (unbalanced OT).** Let a\_v = σ(s(u,v)) be model customer mass per account, s = Mᵀa the model's expected customers per segment, and q^f\_k = p̃^f\_k · n\_k the founder's expected count (unnormalized, which is why *unbalanced* OT fits). Solve

```latex
\min_{\Gamma\ge 0}\; \varepsilon\,\mathrm{KL}\big(\Gamma \,\|\, \mathrm{diag}(a)K\big) + \sum_f \tau_f\,\mathrm{KL}\big(\Gamma^\top\mathbf 1 \,\|\, q^f\big)
```

With hard row mass and ε = 1 the solution is closed form (derived and checked numerically by the agent; Chizat et al. 2018 give the general Sinkhorn updates):

```latex
S_k = s_k^{\,1-t}\,\bar q_k^{\,t},\qquad t = \frac{T}{1+T},\quad T=\sum_f \tau_f,\qquad s'(u,v) = s(u,v) + \log\sum_k M_{vk}\Big(\tfrac{\bar q_k}{s_k}\Big)^{t}
```

- **Meaning of trust.** S lies on the log-linear path from the model's segment masses to the founders' pooled belief (a geometric mean, i.e. a logarithmic opinion pool). t is literally a mirror-descent step toward the founder. t → 0 ignores the prior; t → 1 recovers hard Sinkhorn matching.
- **It's a tempered label-shift correction.** With hard segments, s' = s + t·log(q̄/s): the classic label-shift weight shrunk by an exponent, paralleling RLLS (Azizzadenesheli et al. 2019).
- **Free wrong-prior alarm.** Lost total mass equals exp(−(1−t)·Rényi divergence) between model and prior.
- **Cost:** O(NK), one matrix-vector product, differentiable.

**Doubly robust PU loss with policy-defined propensity.** Thompson sampling plus η-exploration makes each account's contact probability p\_v known by design, with floor p\_v ≥ ηB/N. With imputation ê\_v from the founder prior and Bekker et al.'s within-contact PU loss L̃\_v:

```latex
\hat R_{\mathrm{DR}}(g) = \frac1N\sum_v\Big[\hat e_v + \frac{O_v}{p_v}\big(\tilde L_v - \hat e_v\big)\Big]
```

Apply the non-negative clamp (Kiryo et al.) to the negative-class part. Segment beliefs enter only through the imputation, not as a hard class prior.

**Gibbs-posterior unification.** The whole pipeline is one variational problem (Bissiri, Holmes & Walker 2016): a Gibbs posterior over GNN weights with a small-variance prior (the "residual kept small" device), plus the OT layer as an I-projection with KL slack, which is posterior regularization. Thompson sampling draws weights, then applies the layer.

**Candidate theorems**

1. **Stability.** log S is t-Lipschitz in the prior and (1−t)-Lipschitz in the model: segment-level GNN fluctuations shrink by 1/(1+T).
2. **Exact harm bound.** For true segment distribution p\*: KL(p\*‖g\_t) − KL(p\*‖ŝ) ≤ t·KL(p\*‖q̂). Harm from a wrong prior is at most trust × the prior's KL error.
3. **Convex trust learning.** With several founders, choosing trust weights is a convex program with a unique optimum; with a consistent estimate of p\*, learned trust on a wrong founder goes to zero.
4. **Double robustness.** With known propensities, the risk estimate is unbiased for *any* imputation, including a wrong founder prior; founder error only inflates variance. Plain nnPU, by contrast, is biased under a misspecified class prior.
5. **Consistency under adaptive outreach.** With a propensity floor decaying slower than 1/t, the estimator is consistent (martingale argument, following Hadad et al. 2021).
6. **Sensitivity to unlogged early outreach.** Bound the unknown propensity of pre-system contacts with a marginal sensitivity model and report the confounding level at which the top-k ranking flips (Kallus & Zhou 2018).

**Pitfalls:** don't count the prior three times (offset, imputation, OT target); calibrate λ and t jointly or set λ = 0 when the layer is on. Without exploration, propensities hit zero and the DR estimator is undefined. 1/p weights explode at 5–50 labels, so use self-normalization or stabilizing weights.

## Math: stability theory for the GNN

The stabilizing design is to make message passing a contraction with the founder prior as its source term, train that body only on unlabeled graph structure, and fit just a small strongly convex head on the 5–50 customers. This yields perturbation bounds with no dependence on depth or degree, a no-harm oracle inequality, and a forgetting-bounded-by-gain update.

**Why plain GNNs are unstable here.** Published stability and generalization constants grow with depth and degree: F^(L−1) for multilayer filters (Gama, Bruna & Ribeiro 2020), d^((l−1)/2) times weight-norm products for PAC-Bayes bounds (Liao et al. 2021), and the filter's top eigenvalue raised to the number of training steps (Verma & Zhang 2019). Garg et al. (2020) show the depth dependence disappears exactly when message passing is contractive.

**Prior-sourced contractive equilibrium (new component).** Node embeddings are the unique fixed point of

```latex
Z^* = \sigma\Big(\textstyle\sum_r \gamma_r \hat S_r Z^* W_r + XU + p\,b^\top\Big),\qquad \textstyle\sum_r \gamma_r\|W_r\|_2 \le \gamma < 1
```

with degree-floored, row-normalized typed operators Ŝ\_r and prior logits p as the source term. Well-posedness follows the implicit-GNN conditions (Gu et al. 2020). In the linear case this is personalized PageRank whose teleport is the founder prior, with operator stability constant γ/(1−γ)² (Kenlay et al. 2021, Prop. 2). The source term also prevents oversmoothing (Di Giovanni et al. 2023).

**Calibrated head.** Score s(v) = λp\_v + aᵀφ(v), trained with the convex unbiased-PU logistic risk plus μ(‖a‖² + c(λ−λ₀)²). Strong convexity gives a unique minimizer, so results don't depend on random seed; the prior/residual split is unique by APHYNITY's Chebyshev-set argument.

**Sequential update.** A single-anchor Laplace–Kalman recursion on the head, anchored at the latest posterior only (Huszár 2017 shows multi-anchor EWC double-counts). Online natural gradient is exactly this filter (Ollivier 2018). Thompson sampling draws from the head posterior directly.

**Candidate theorems** (derived by the agent from cited results; not yet published or checked)

1. **Joint perturbation bound.** For every account, score change ≤ (λ + ‖a‖‖b‖/(1−γ))·prior error + (‖a‖γκ'/(1−γ))·graph error + feature term. No dependence on depth, width or degree.
2. **Exponential locality.** A change at hop distance ≥ r moves v's score by at most ‖a‖γ^r‖g‖/(1−γ): new edges or prior revisions far away barely matter. Top-k outreach lists stay unchanged whenever the rank-k gap exceeds twice the bound.
3. **Uniform stability and oracle inequality.** The bottleneck is the number of customers n\_p. Excess risk scales with the norm of the minimum-norm residual needed on top of the prior over √n\_p. A good prior shrinks that norm; a wrong prior costs at most O(κ'√c·λ₀/√n\_p) versus ignoring it.
4. **Monotone uncertainty and bounded forgetting.** Without drift, posterior variance never increases, so exploration decays monotonically; for a Gaussian head the sequential result equals batch training (zero forgetting). For any convex loss, forgetting on old data is bounded by the improvement on new data.

**Pitfalls:** the cold-start seller has near-zero degree, which breaks degree-based bounds, so use a degree floor. Labels are adaptively selected, not i.i.d., so state bounds transductively or with martingale tools. Contraction limits long-range and heterophilous signals, and constants blow up as γ → 1. Fine-tuning the body on the few labels voids the exact bounds; freeze it or use anchored ensembles. Treat constants as design guidance and verify with empirical sensitivity curves.

## CPR-LP v2: the assembled algorithm

The five math threads describe one object: a graph-smoothed Bayesian posterior centred on a calibrated founder prior. The diffusion steady state, the Gaussian Markov random field posterior and the Gaussian-process posterior mean are the same estimator, because a GMRF with precision γL + B is a GP with kernel (γL + B)⁻¹. That shared spine is what makes the theorems compose.

**The model, in five layers**

**1. Calibrated, pooled founder prior.** Segment effects are centred on calibrated beliefs, with trust shared across segments and founders:

```latex
\eta_k = a_f + b_f\,\operatorname{logit}(m_{fk}) + \xi_k,\qquad \xi_k \sim \mathcal N(0, \omega_k^2),\qquad (a_f, b_f) \sim \text{population prior}
```

**2. Stable body.** A prior-sourced contractive equilibrium produces account embeddings φ(v); it is trained only on unlabeled graph structure (link reconstruction) and frozen, so the 5–50 labels never touch its weights.

**3. Scoring field.** Account logits are segment effects plus a graph-smooth residual plus a neural-linear term on φ. With prior precision B, graph Laplacian L and label precision J on contacted accounts:

```latex
x^* = \Phi\hat\eta + M^{-1}\big[B\,\phi^\top \hat w + J(\tilde y - \Phi\hat\eta)\big],\qquad M = \gamma L + B + J
```

The founder's influence at each account is the node-adaptive weight ω\_prior(v), not a global λ, and it decays with nearby labels.

**4. Likelihood that respects how labels arise.** Positive-unlabeled outcomes via Pólya-Gamma augmentation (closed-form updates); segment class priors from the calibrated beliefs (which also resolves PU's identifiability gap); contacts weighted by their known propensities in a doubly robust loss, so founder error inflates variance, not bias. Delay-corrected pseudo-observations per funnel stage, weighted by value.

**5. Decisions and updates.** Thompson sampling over the posterior (one linear solve per draw via perturb-and-MAP) with stratified η-exploration; single-anchor Laplace–Kalman updates as data arrive.

**Optional diagnostic.** The unbalanced-OT segment layer is best used as an *alarm* (lost mass = Rényi disagreement between model and founder) rather than a third injection of the prior. The prior should enter as belief once; elsewhere only as a control variate.

**Headline theorems for the paper** (pick three; all are agent derivations that still need full proofs)

1. **Bounded harm from wrong priors.** Exact identity: KL(p\*‖g\_t) − KL(p\*‖ŝ) ≤ t·KL(p\*‖q̂); and frequentist regret Õ((√γ\_n + D₀)√(nγ\_n)) with D₀² ≈ 2Σ\_k c\_k·kl(p\*\_k‖p̃\_k), a confidence-weighted KL. Corollary: regret is minimised when stated confidence matches actual error, so calibration is optimal, not cosmetic.
2. **Identifiability and prior washout.** Without exploration, a confident pessimistic belief about the best segment causes linear regret (Thompson sampling: linear until ≈ e^(z²/2)); smoothing cannot repair graph-coherent errors in never-contacted segments. With η-exploration, segment estimates are consistent at O(1/√(ηn)) and the prior washes out at an explicit time.
3. **Stability and locality.** The scoring operator is a positive-definite M-matrix: unique, a convex combination of prior and labels, no oversmoothing. Score perturbations are bounded jointly in prior error and graph error with no depth or degree dependence, and each label's influence decays as ρ^distance.

Supporting results: double robustness of the PU loss; monotone non-increasing posterior variance; forgetting bounded by gain; conformal recall guarantees for shortlists.

**What is genuinely new**

- Founder beliefs as a calibrated, pooled, *learned-trust* prior inside a graph posterior (no prior work learns per-source trust in segment-level beliefs on graphs).
- The confidence-weighted-KL regret term and the identifiability/washout pair for priors that steer their own data collection.
- The prior-sourced diffusion / contractive equilibrium as a stable, oversmoothing-free way to carry beliefs over a heterogeneous market graph.

**Before claiming anything**

- Check every constant against the source papers; several bounds came through summarised PDFs, and some steps were marked "verify".
- Keep Bayesian and frequentist claims separate: Bayesian regret can't express a wrong founder.
- At 5–50 labels all constants are vacuous; the theorems explain *shape* (when priors help, how fast they wash out), and simulations must show it.
- Check EXNEX (Neuenschwander 2016) and amortized data-borrowing work before claiming novelty for per-segment trust.

## Sources

Numbers were extracted by research agents through a summarising fetch tool; spot-check key tables before citing.

**Relational and few-shot learning:** [RelBench](https://arxiv.org/html/2407.20060) · [RelBench v2](https://arxiv.org/html/2602.12606) · [RelGNN](https://arxiv.org/html/2502.06784) · [ContextGNN](https://arxiv.org/html/2411.19513) · [KumoRFM-2](https://arxiv.org/html/2604.12596) · [OpenRFM](https://arxiv.org/html/2606.04320) · [GILT](https://arxiv.org/html/2510.04567) · [Meta-Graph](https://arxiv.org/pdf/1912.09867) · [HeaRT](https://arxiv.org/abs/2306.10453) · [Can GNNs learn link heuristics?](https://arxiv.org/html/2411.14711) · [NCN/NCNC](https://arxiv.org/html/2302.00890v4) · [TabPFN-2.5](https://arxiv.org/html/2511.08667) · [PULL](https://arxiv.org/html/2405.11911) · [Exposure bias in link recommendation](https://arxiv.org/abs/2106.07041) · [Cold Brew](https://arxiv.org/abs/2111.04840) · [NodeDup](https://arxiv.org/html/2402.09711v2) · [Link-MoE](https://arxiv.org/abs/2402.08583)

**Prior injection:** [Hu et al. 2016](https://arxiv.org/abs/1603.06318) · [Semantic loss](https://arxiv.org/abs/1711.11157) · [Posterior regularization](https://www.jmlr.org/papers/volume11/ganchev10a/ganchev10a.pdf) · [Generalized expectation](https://www.jmlr.org/papers/volume11/mann10a/mann10a.pdf) · [Labeled features](https://maroo.cs.umass.edu/pub/web/getpdf.php?id=820) · [Functional variational BNNs](https://arxiv.org/abs/1903.05779) · [Good functional prior](https://arxiv.org/abs/2011.12829) · [Function-space VI](https://arxiv.org/abs/2312.17199) · [Correct & Smooth](https://arxiv.org/abs/2010.13993) · [Graph Posterior Network](https://arxiv.org/abs/2110.14012) · [PFNs](https://arxiv.org/abs/2112.10510) · [PFNs4BO](https://arxiv.org/abs/2305.17535) · [APHYNITY](https://arxiv.org/abs/2010.04456) · [Snorkel](https://arxiv.org/abs/1711.10160) · [nnPU](https://arxiv.org/abs/1703.00593)

**Elicitation and trust:** [Mikkola et al.](https://arxiv.org/abs/2112.01380) · [AutoElicit](https://arxiv.org/html/2411.17284) · [Had enough of experts?](https://arxiv.org/html/2402.07770) · [LLM priors in trials](https://arxiv.org/html/2509.04250) · [LLM Processes](https://arxiv.org/html/2405.12856) · [Daee et al.](https://arxiv.org/html/1612.03328) · [Robust mixture priors](https://arxiv.org/html/2412.03185v2) · [Commensurate priors](https://pmc.ncbi.nlm.nih.gov/articles/PMC3134568) · [Power prior review](https://pmc.ncbi.nlm.nih.gov/articles/PMC4626399) · [Normalized power prior](https://arxiv.org/abs/2004.14912) · [Prior-data conflict](https://arxiv.org/pdf/1611.00113) · [Extremizing forecasts](https://faculty.wharton.upenn.edu/wp-content/uploads/2015/07/2015---two-reasons-to-make-aggregated-probability-forecasts_1.pdf) · [Clemen & Lichtendahl](https://people.duke.edu/~clemen/bio/BayesianCalibration.pdf) · [Misspecified priors](https://arxiv.org/pdf/2107.01509) · [Learning to defer](https://arxiv.org/html/2006.01862) · [Agentic-Priors](https://arxiv.org/pdf/2610.21938) · [LLM Beta priors at NAVER](https://arxiv.org/pdf/2608.03382) · [OPEN](https://arxiv.org/pdf/2403.05534)

**Evaluation:** [Glasserman & Lin](https://arxiv.org/abs/2309.17322) · [Sarkar & Vafa](https://conference.nber.org/conf_papers/f205530/f205530.pdf) · [Test of lookahead bias](https://arxiv.org/html/2512.23847v1) · [ChronoGPT](https://arxiv.org/html/2502.21206v3) · [Time Machine GPT](https://arxiv.org/html/2404.18543) · [Halawi et al.](https://arxiv.org/abs/2402.18563) · [ForecastBench](https://arxiv.org/abs/2409.19839) · [Forecaster pitfalls](https://arxiv.org/html/2506.00723v1) · [Date-filter leakage](https://arxiv.org/html/2602.00758v1) · [Elephants never forget](https://arxiv.org/html/2404.06209v3) · [Fischhoff 1975](https://apps.dtic.mil/sti/pdfs/ADA001807.pdf) · [Pre-registration workshop](http://proceedings.mlr.press/v148/bertinetto21a/bertinetto21a.pdf) · [Offline replay](https://arxiv.org/abs/1003.5956) · [Recommendations as treatments](https://arxiv.org/abs/1602.05352) · [ACIC 2016](https://arxiv.org/abs/1707.02641) · [PU survey](https://arxiv.org/abs/1811.04820) · [ImageNet v2](https://arxiv.org/abs/1902.10811) · [Neural nets vs boosted trees](https://arxiv.org/abs/2305.02997)

**Data:** [USAspending API](https://api.usaspending.gov/docs/endpoints) · [Wayback CDX](https://raw.githubusercontent.com/internetarchive/wayback/master/wayback-cdx-server/README.md) · [SupplySim](https://arxiv.org/pdf/2407.18772) · [SAP SALT](https://arxiv.org/html/2501.03413) · [PMGraph](https://huggingface.co/datasets/shawntzx/PMGraph) · [CompanyKG](https://arxiv.org/abs/2306.10649) · [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)

**Industry and founder evidence:** [Camuffo et al. 2020](https://gwern.net/doc/economics/2019-camuffo.pdf) · [I-Corps pivoting](https://researchwith.stevens.edu/en/publications/strategic-pivoting-in-deep-tech-an-investigation-of-nsf-i-corps-t/) · [Cooper et al. 1988](https://gwern.net/doc/psychology/cognitive-bias/1988-cooper.pdf) · [First Round on ICP](https://review.firstround.com/how-vanta-clay-retool-found-icp) · [Google optimized targeting](https://support.google.com/google-ads/answer/10537509) · [Meta targeting suggestions](https://www.jonloomer.com/targeting-suggestions/) · [MadKudu overrides](https://help.hginsights.com/docs/customer-fit-overrides) · [Kumo lead scoring](https://docs.nvidia.com/sdgm/examples/growth-lead-scoring)
