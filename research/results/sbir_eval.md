Real B2B tier: SBIR firms' first federal customers (docs/sbir-pipeline.md).

Task: firm f at t0 (first SBIR Phase I award, 2012–2018) ranks federal awarding sub-agencies by how
likely each is to make a real purchase (non-SBIR, non-R&D contract) within 5 years. Evaluated on
firms with >= 1 purchase. Time split: firms with t0 in 2012–2015 fit anything learned; 2016–2018 test.

"Founder belief" = keyword rules over the firm's own pre-t0 text (title, topic, abstract). No LLM
(a modern LLM may know later outcomes). The funding agency / branch is a known fact at t0, given to
every method.

Methods
  popularity          sub-agency purchase frequency among training firms
  funding             funding branch first, then other sub-agencies of the funding agency, then popularity
  keywords            abstract keyword belief, ties broken by popularity
  pooled              logistic regression fitted on training firms over
                        [same branch, same agency, log keyword hits, log popularity]
                      = calibrated trust in the keyword "founder" pooled across firms
  pooled_nokw         the same without the keyword feature (ablation: does the founder belief add?)

Pre-registered (committed before the full cohort's outcomes were available):
  SB1. On purchases from a sub-agency OUTSIDE the funding agency (the novel slice), pooled beats
       pooled_nokw on recall@3 (founder beliefs help where there is nothing to memorize).
  SB2. On purchases from the funding agency, funding ≈ pooled ≈ pooled_nokw (keywords add ~nothing).
  SB3. Overall, pooled >= max(popularity, funding, keywords) on MRR.
Bootstrap over firms (1,000 resamples) for 95% intervals of paired differences.

## Results

Train firms (2012–2015, >=1 purchase): 293; test firms (2016–2018): 218; candidate sub-agencies: 81; test purchases: 407 (56% from a new agency).

Pooled weights [intercept, same_branch, same_agency, kw, logpop]: [-6.12, 0.47, 0.99, 0.57, 1.04]

| method | recall@3 all | recall@3 funding agency | recall@3 new agency | MRR all |
|---|---|---|---|---|
| popularity | 0.339 | 0.436 | 0.263 | 0.294 |
| funding | 0.413 | 0.821 | 0.092 | 0.345 |
| keywords | 0.354 | 0.564 | 0.189 | 0.320 |
| pooled | 0.430 | 0.743 | 0.184 | 0.362 |
| pooled_nokw | 0.432 | 0.749 | 0.184 | 0.370 |

Paired differences over test firms (mean, bootstrap 95% CI, firms):

- pooled − pooled_nokw, hit3, new_agency: +0.008 [-0.014, +0.033] (n = 133)
- pooled − pooled_nokw, hit3, funding_agency: -0.007 [-0.030, +0.014] (n = 118)
- pooled − funding, hit3, funding_agency: -0.096 [-0.148, -0.047] (n = 118)
- pooled − popularity, rr, all: +0.088 [+0.055, +0.126] (n = 218)
- pooled − funding, rr, all: +0.006 [-0.017, +0.027] (n = 218)
- pooled − keywords, rr, all: +0.047 [+0.020, +0.078] (n = 218)
## Scoring (written after the run, 2026-11-25 ~13:50)

Run by Claude (Track A taken back from the Codex handoff; Codex had not started). Data: full cohort pull finished
2026-11-25 00:53 (`data/interim/sbir_stats.txt`: 511 / 5,225 firms with ≥ 1 purchase). Script unchanged since
pre-registration.

| prediction | result | detail |
|---|---|---|
| SB1 keywords help on new-agency purchases (pooled > pooled_nokw, recall@3) | **FAILS** | +0.008 [−0.014, +0.033], n = 133 firms |
| SB2 on funding-agency purchases funding ≈ pooled ≈ pooled_nokw | **FAILS (partly)** | pooled ≈ pooled_nokw (−0.007 [−0.030, +0.014]), but funding beats pooled by −0.096 [−0.148, −0.047] |
| SB3 pooled MRR ≥ max(popularity, funding, keywords) | **HOLDS (point estimate only)** | vs funding +0.006 [−0.017, +0.027]; vs keywords +0.047 [+0.020, +0.078]; vs popularity +0.088 [+0.055, +0.126] |

Readings:
- **Keyword "founder beliefs" add nothing once the funding agency and popularity are known** (pooled_nokw MRR
  0.370 ≥ pooled 0.362; learned keyword weight 0.57 is mostly redundant with same-agency). Keyword rules over a
  firm's abstract are a weak stand-in for a founder's beliefs; this tier does not test real founder priors.
- **On the novel slice (56% of test purchases come from an agency that did not fund the firm), plain popularity is
  the best method** (recall@3 0.263 vs 0.18–0.19 for keywords/pooled, 0.09 for funding). Same pattern as the
  RelBench audit: what can be memorised (funding agency) is easy, novel customers are not predicted by anything
  here except the base rate.
- SB2's failure: the pooled logistic under-weights the exact funding branch relative to the rule "funding branch
  first" (0.821 vs 0.743 recall@3 on funding-agency purchases). A calibration fit with 293 training firms over 81
  candidates is noisy.
- Implication for the paper: SBIR is reported as a B2B proxy with a **negative** result for keyword priors; the
  point of the tier becomes the slice structure (memorisable vs novel), not a founder-prior win.
