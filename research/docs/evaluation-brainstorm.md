# Evaluation brainstorm: making the claim believable without public outreach data

Date: 2026-11-24. Status: brainstorm. Nothing here is decided. This goes beyond PLAN.md §5, which already covers sim-to-sim stress tests, real outcomes with a simulated founder, the benchmark audit, USAspending backtests and a prospective cohort.

Tags for every specific source: **[V]** = I confirmed it exists this session (by web search or a live API call), with the URL. **[M]** = from memory, not checked. **[U]** = an unverified idea, meaning no specific source is claimed.

The four problems are referred to as P1 (no outreach data), P2 (benchmarks reward memorizing), P3 (the simulator is optimistic) and P4 (hindsight contamination).

---

## Ideas

### 1. SBIR/STTR abstracts as dated founder priors, scored against later federal sales
Thousands of small firms write a public, dated abstract when they win an SBIR/STTR award. NSF abstracts have a required "broader impact / commercial potential" passage that says who the firm thinks will buy. That passage is a real founder prior, written before any outcome. Outcomes come from USAspending: which agencies and sub-offices later award the same firm (UEI) non-SBIR contracts. Each firm is one task.
- Set the funding agency aside as the k=1 label. Otherwise the prior trivially points at it.
- Map the abstract to segments (agency × buyer NAICS × office type) using only taxonomy or embedding similarity, so no world knowledge enters. An LLM extractor is an ablation, not the main arm.
- Redact firm names. Drop "SBIR mills" (firms with more than N prior awards) so the cohort is genuinely early-stage.
- The firms that never sell again are the failure cases, and they are in the sampling frame by construction.

Sources:
- NSF Award API, which returns `abstractText` and was tested live: https://api.nsf.gov/services/v1/awards.json **[V]**
- SBIR.gov bulk award CSV, about 367 MB (HTTP 200): https://data.www.sbir.gov/awarddatapublic/award_data.csv **[V]**
- SBIR.gov API: https://api.www.sbir.gov/public/api/awards **[V]** exists but currently returns HTTP 403.

**Doubt removed:** P1 and P4 at once. These are real beliefs from real founders, recorded before the outcome, on thousands of companies, with true negatives (offices that never bought). It also measures where real founders sit on the simulator's reliability (ρ) axis, which calibrates the founder generator.
**Feasibility (1–3 mo):** High. Public data, US-government work, no IRB, and it reuses the USAspending pipeline. The risks are entity matching (UEI vs. the older DUNS identifier) and DoD reporting lag.
**Success:** In emerging strata, prior + labels beats labels-only at k ≤ 10, and learned trust correlates with each abstract's measured accuracy afterwards.
**Failure:** Abstract priors are no better than the agency's base rates. That is still reportable as "real founder priors are weak in this domain", which tests H2.
**Honest flag:** A reviewer will say "a grant abstract is not a sales belief, and government is not a commercial market". Present it as a public proxy, not the whole story.
**Ratings:** novelty 4, persuasiveness 4, feasibility 4.

### 2. Live prospective benchmark (ForecastBench-style) on federal first sales
Freeze predictions now for events that resolve later. Take every small vendor whose first award appears in a given month (k=1 label). Predict which agency offices give it awards 2–5 over the next 3–6 months, then publish a hash of the predictions with a timestamp (on OSF or arXiv). Resolve each month. Future events cannot leak into any model, so frontier LLMs become legitimate prior sources and baselines here, which they are not in backtests. The template is ForecastBench (Karger et al., ICLR 2025), https://arxiv.org/abs/2409.19839 **[V]**.
**Doubt removed:** P4 completely, and P3 partly, because the test distribution did not exist when the simulator was designed.
**Feasibility:** Medium. The pipeline is cheap once #1 exists. The constraint is the clock: a 3-month horizon gives only one or two resolution rounds inside the project window. DoD reporting delay (I recall about 90 days) **[M]** argues for excluding DoD or using longer horizons.
**Success / failure:** Pre-registered lift over labels-only and over a frontier LLM with no labels, on future data. Failure: no lift, published anyway (that is the point of pre-registration).
**Ratings:** novelty 4, persuasiveness 5, feasibility 3.

### 3. Negative-control priors and "prior surgery"
Run on every real tier, with five prior conditions:
- (a) The true prior.
- (b) The prior permuted across segments.
- (c) Another company's prior (cross-firm swap; this needs #1).
- (d) A flat prior with the same confidence.
- (e) The true prior with one segment flipped.

Predictions: the gain disappears under (b), (c) and (d). Under (e), learned trust on the flipped segment decays at the rate the CPR-LP posterior predicts, and the other segments are unaffected. This is the negative-control logic from epidemiology applied to the prior. It separates "the founder's knowledge helps" from "any structured regularizer helps" and from "the extra parameters help".
**Doubt removed:** It tests the trust mechanism separately from ranking quality. It also guards against a reviewer's "your gain is an artifact" reading in all four problems.
**Feasibility:** Very high. No new data is needed, only runs.
**Success:** The gain is (a) ≫ (b), (c), (d) ≈ 0, and the per-segment trust response is localized in (e).
**Failure:** The gain survives permutation, meaning the method is not using the founder's knowledge.
**Ratings:** novelty 3, persuasiveness 4, feasibility 5.

### 4. Pre-registered dose-response: "helps exactly where theory says, nowhere else"
From CPR-LP v2, derive a closed-form predicted gain for each task. It is a function of the prior's effective sample size versus k, the measured agreement between prior and labels, and a locality/emergence score (for example, the share of test positives within h hops of known positives). Pre-register three things:
- A monotone relation between predicted and observed gain across all real tasks, measured by rank correlation over tasks.
- An equivalence test (two one-sided tests, TOST, at margin ε) showing zero gain in mature or high-locality strata.
- A crossover point.

A method that helps everywhere looks like a bug. One that helps in exactly the predicted cells, and is provably neutral elsewhere, is evidence. This extends the benchmark audit from "benchmarks are biased" to "our gain lives where benchmarks don't look".
**Doubt removed:** P2. A loss on the memorizable slice becomes a predicted null, not a failure.
**Feasibility:** Very high. The risk is computing the predictor with information that leaks from the outcome. Use only training-side quantities.
**Success:** Rank correlation clearly above 0, and TOST passes in mature strata. **Failure:** Gains are uncorrelated with the predictor.
**Ratings:** novelty 3, persuasiveness 4, feasibility 5.

### 5. Selective-labels partial identification on non-randomized logs
Bank Marketing, the SBIR→USAspending data and the field cohort all have outcomes only for accounts that were contacted or bid. Report three numbers:
- The point estimate.
- Worst-case and best-case bounds on recall and precision, treating every uncontacted account as negative or as positive, Manski-style.
- A "contraction" estimate using the most permissive contactor as reference (Lakkaraju et al., KDD 2017, "The selective labels problem") **[M]**.

If PC-RFM beats the baselines even at the pessimistic bound, missing-not-at-random bias cannot explain the result.
**Doubt removed:** It is the honest answer to "your real tiers are biased by outreach", which is the gap between P1 and P3.
**Feasibility:** Very high. The bounds are often wide, so it works best on a narrow candidate set (for example, offices active in the firm's NAICS code).
**Ratings:** novelty 3, persuasiveness 3, feasibility 5.

### 6. Pharma launches: a new drug looking for its first prescribers
This is the one public domain where "new entrant, segment beliefs, a target list, true negatives, and a network" all exist.
- Seller = a newly approved drug. Approval dates come from Drugs@FDA data files: https://www.fda.gov/drugs/drug-approvals-and-databases/drugsfda-data-files **[V]**
- Customers = prescribers. Yearly counts per prescriber and drug, 2013–2024, come from CMS Part D Prescribers by Provider and Drug: https://catalog.data.gov/dataset/medicare-part-d-prescribers-by-provider-and-drug **[V]**
- Graph = CMS Physician Shared Patient Patterns, 2009–2015: https://img.nber.org/data/physician-shared-patient-patterns-data.html **[V]**
- Prior = the indication on the label at approval, mapped to specialties. It is dated and set before uptake. Off-label uptake supplies naturally "wrong founders".

Pharma sales-rep targeting is literally prior-driven outreach, and the classic diffusion study of physicians (Coleman, Katz & Menzel 1957) **[M]** gives the paper a lineage.
**Doubt removed:** P1 and P4 in a non-government domain, with a real graph.
**Caveats:**
- Annual granularity.
- Rows with fewer than 11 claims are suppressed, so "adoption" means at least 11 claims.
- Medicare only.
- The graph ends in 2015, so restrict to launches around 2013–2016.
- B2C-ish, not startup B2B.

**Feasibility:** High. US-government data, no IRB, multi-GB files.
**Ratings:** novelty 5, persuasiveness 3, feasibility 4.

### 7. Interleaved within-founder field experiment
Real founders get one blinded contact list built by team-draft interleaving of their own ranking and the model's. Interleaving comes from search evaluation: Chapelle, Joachims, Radlinski & Yue, TOIS 2012, https://doi.org/10.1145/2094072.2094078 **[V]**. Founders contact the list as usual, and the outcome is a reply, a meeting booked or a qualified lead within 14 days. Each founder is their own control, which removes most between-team variance, so about 20 teams × 30–40 contacts can be enough. Stanford's Lean LaunchPad (ENGR 245) expects 10–15 interviews per week per team (https://steveblank.com/category/lean-launchpad/ **[V]**), so the outreach happens anyway. Founders' priors are elicited first, which also feeds #9.
**Doubt removed:** The real objection, "does it help actual startups?". No simulator, no benchmark, no hindsight.
**Feasibility:** Low to medium. It needs IRB (likely minimal-risk, but contacts are third parties), a course or accelerator partner (ENGR 245, StartX, or the NSF I-Corps hubs **[U]** for access), and fast-proxy outcomes rather than sales.
**Success:** Interleaving credit favours the model, or the hybrid list, at p < 0.05 under a pre-registered test. **Failure:** Founders' own lists win, which is still informative and points to strong tacit priors.
**Honest flag:** Reply rate is not purchase, and the sample is small and Stanford-skewed.
**Ratings:** novelty 4, persuasiveness 5, feasibility 2.

### 8. Realism audit of the simulator: two-sample test and a realism–performance curve
Fit a classifier two-sample test (Lopez-Paz & Oquab, ICLR 2017, https://arxiv.org/abs/1610.06545 **[V]**) that separates real tasks (from #1, #6 and Bank Marketing) from simulated ones, using task-level summary statistics:
- degree tail
- segment concentration of positives
- locality
- prior accuracy

Then plot PC-RFM's real-world advantage over CPR-LP against how "real" each task looks to the classifier. Prediction: the advantage shrinks toward CPR-LP as tasks fall outside the simulator's support. Add a correctness check on the conjugate sub-family, where CPR-LP is the exact Bayes posterior: PC-RFM's predictions must match it, in the spirit of simulation-based calibration (Talts et al., https://arxiv.org/abs/1804.06788 **[V]**).
**Doubt removed:** It turns P3 into a measured quantity instead of a vague worry.
**Honest flag:** On raw features, C2ST nearly always reaches AUC ≈ 1. It is informative only on summary statistics.
**Ratings:** novelty 3, persuasiveness 3, feasibility 4.

### 9. Trust recovery against ex-post founder reliability (proper scoring)
Elicit priors as probabilities, for example "out of 20 such accounts, how many buy?", and score each founder afterwards with a proper scoring rule (Brier or log). This gives a ground-truth reliability per founder. The test is whether the model's learned trust after k labels ranks founders by that measured reliability. In the prospective cohort, pay a small bonus tied to the score so that honest reporting pays (incentive compatibility). In SBIR (#1), reliability is computed directly from outcomes, with no elicitation needed.
**Doubt removed:** It evaluates "learns how much to trust the founder" directly, which is the paper's core claim, rather than only through ranking metrics.
**Feasibility:** High on SBIR. Slow in the cohort, because outcomes take months.
**Ratings:** novelty 4, persuasiveness 4, feasibility 3.

### 10. Red-team founders under a realism constraint
An adversary searches for priors that maximize PC-RFM's loss. The search is gradient-based over the belief tokens or proposed by an LLM, and the priors must look real, either by staying inside the distribution of #1's real priors or by passing #8's discriminator. Report the worst-case loss against CPR-LP's analytic bound and against labels-only. PC-RFM's trust is learned only from the simulator's error types, so a reviewer will ask about errors the simulator never produced.
**Doubt removed:** The tail of P3, the safety claim behind H2.
**Feasibility:** High, compute only.
**Success:** The worst-case loss stays within ε of labels-only, or PC-RFM fails where CPR-LP is safe, which is a finding to report honestly.
**Ratings:** novelty 3, persuasiveness 3, feasibility 4.

### 11. Guarantee-backed "safe trust" switch
Wrap either model in a certificate. Use the prior only if a held-out check on the k labels certifies, with probability at least 1−δ, that the loss is no worse than labels-only + ε (conformal risk control, Angelopoulos et al. **[M]**). The claim then holds on any data, without a representative benchmark.
**Honest flag:** With k ≤ 20 the certificate is loose and will often turn trust off. Worst-case safety is guaranteed, but the gain may disappear. Present it as a safety property, not as evidence of usefulness.
**Ratings:** novelty 3, persuasiveness 3, feasibility 4.

### 12. Open-source packages finding their first dependents
Seller = a new package. Customers = the packages and organizations that adopt it. Prior = the README and keywords at first release, dated by git history. Graph = the dependency graph. The deps.dev API (https://api.deps.dev/v3/... was tested live) **[V]** and the BigQuery dataset (https://docs.deps.dev/bigquery/ **[V]**) provide the data. I did not find the licence for the BigQuery data, and upstream registries carry their own terms. Popularity bias is extreme, which makes this an ideal testbed for the memorizability audit.
**Honest flag:** Adopting a dependency is not buying, and a README is weak as a "belief".
**Ratings:** novelty 3, persuasiveness 2, feasibility 4.

### 13. Blinded human baseline: do sales and VC people beat the model?
Recruit raters with sales experience (Upwork, or an MBA pool) to rank candidate buyer offices for anonymized SBIR firms (#1), given the abstract and k known buyers. Compare four rankers: human alone, model alone, human prior → model, and model → human. Run a recognition screen first.
**Feasibility:** About $2–5k and probably IRB-exempt.
**Honest flag:** Reviewers discount non-expert crowds. It is persuasive only with genuinely experienced raters, and they are hard to recruit at scale.
**Ratings:** novelty 2, persuasiveness 3, feasibility 3.

---

## Summary table

| # | Idea | Problems addressed | Novelty | Persuasiveness | Feasibility | P×F |
|---|---|---|---|---|---|---|
| 3 | Negative-control priors and prior surgery | P3, trust mechanism | 3 | 4 | 5 | 20 |
| 4 | Pre-registered dose-response and equivalence | P2 | 3 | 4 | 5 | 20 |
| 1 | SBIR abstracts → USAspending | P1, P4 | 4 | 4 | 4 | 16 |
| 2 | Live prospective benchmark | P4, P3 | 4 | 5 | 3 | 15 |
| 5 | Selective-labels bounds | P1/P3 bias | 3 | 3 | 5 | 15 |
| 6 | Pharma first prescribers | P1, P4 | 5 | 3 | 4 | 12 |
| 8 | Two-sample realism curve and posterior check | P3 | 3 | 3 | 4 | 12 |
| 9 | Trust vs. ex-post founder reliability | trust mechanism | 4 | 4 | 3 | 12 |
| 10 | Red-team founders | P3 tail | 3 | 3 | 4 | 12 |
| 11 | Safe-trust certificate | P2, P3 | 3 | 3 | 4 | 12 |
| 7 | Interleaved field experiment | P1–P4 | 4 | 5 | 2 | 10 |
| 13 | Blinded human baseline | P1 | 2 | 3 | 3 | 9 |
| 12 | Open-source first dependents | P2 | 3 | 2 | 4 | 8 |

## (a) Top 5 by persuasiveness × feasibility

1. **Negative-control priors and prior surgery (#3)**: 3 / 4 / 5
2. **Pre-registered dose-response and equivalence (#4)**: 3 / 4 / 5
3. **SBIR abstracts as dated priors → USAspending outcomes (#1)**: 4 / 4 / 4
4. **Live prospective benchmark (#2)**: 4 / 5 / 3
5. **Selective-labels partial identification (#5)**: 3 / 3 / 5

The interleaved field experiment (#7) is the single most persuasive idea, but it falls out of the top five on feasibility. Start the IRB and partner conversations now, whatever else is chosen.

## (b) Proposed evaluation package for the paper

| Component | P1 no data | P2 benchmarks | P3 sim optimism | P4 hindsight |
|---|---|---|---|---|
| A. SBIR-abstract backtest (#1), with #9 trust-vs-reliability inside it | ✔ real beliefs + outcomes | stratified by emergence | CPR-LP vs PC-RFM gap on real data | ✔ beliefs predate outcomes |
| B. Live prospective benchmark (#2), on the same pipeline | ✔ | | ✔ unseen distribution | ✔ outcomes did not exist yet |
| C. Negative-control and prior-surgery suite (#3), run on A, Bank Marketing and Hillstrom | | | ✔ the gain comes from knowledge, not structure | |
| D. Pre-registered dose-response and equivalence (#4), run on all real tiers and the RelBench audit | | ✔ the null in mature strata is predicted | ✔ theory, not the simulator, sets expectations | |
| E (stretch). Interleaved pilot (#7), 10–20 teams | ✔ actual startups | | ✔ | ✔ |

Report selective-labels bounds (#5) next to every non-randomized result in A and E.

**Headline story.** On real founders' dated beliefs (A, B) the method helps where theory predicts (D). The help comes from the beliefs themselves, not from structure (C). It holds under pessimistic missing-label bounds (#5). It holds on outcomes that did not exist when we froze predictions (B).

**Order of work:**
1. C and D first, since they need no new data.
2. Then A.
3. Then B, frozen as soon as A's pipeline runs, so it resolves before submission.
4. E in parallel, if IRB allows.

## (c) Considered and rejected

- **Regression discontinuity or instrumental variables on accelerator admission or SBIR scoring cutoffs.** These identify the causal effect of a program, not the predictive accuracy of a ranker. Off-policy evaluation with known propensities (Hillstrom, already planned) is the right causal tool here.
- **Prediction markets on segment outcomes.** Markets would be thin, have no liquidity and raise regulatory problems. They add nothing that the proper scoring in #9 does not.
- **LLM-simulated buyers or "digital twin" customers.** Circular: they would test the method against an LLM's stereotypes. A skeptical reviewer would reject it immediately.
- **Crunchbase, G2, Product Hunt, app-store reviews or LinkedIn job posts as data.** Terms of service forbid or restrict ML use, and scraping risk is high. G2 and Crunchbase are already ruled out in CLAUDE.md. I did not check Product Hunt's terms **[U]**.
- **S-1 customer-concentration disclosures.** They cover only mature firms that reached IPO, are survivor-biased and give a handful of large customers. That is the opposite of the target regime.
- **SEC Reg CF Form C business plans.** They are dated beliefs, but there are no customer-level outcomes to score them against **[U]**.
- **First citations of new papers, a GitHub repo's first contributors, a restaurant's first Yelp reviews.** The analogy is too loose: no buying decision, and no meaningful segment prior. It would read as data-hunting.
- **CRM-vendor or VC data partnerships as a paper-v1 tier.** Months of NDA work, and nothing could be released, so it is not reproducible. Pursue it for a follow-up only.
- **A crowdsourced "wisdom of founders" tier on its own.** It is subsumed by #13 and #9. Without real stakes, crowd priors measure crowd opinion, not founder knowledge.
- **A Kaggle-style public challenge.** Good for the Datasets & Benchmarks track later, but it is not evidence for the method within 3 months.
