# Paper outline (working draft, 2026-11-24)

**Working title:** *Learning to Trust Founders: Revisable Priors for Customer Discovery with 5–50 Labels*

Status legend: ✅ evidence in hand · 🟡 partial / simulated only · ⬜ not yet run. Every claim links to the
result that supports it. Numbers are lift@100 unless stated.

## 1. Problem and thesis

Early-stage companies must find customers before recommendation data exists. The founder's beliefs are
the strongest signal they have, but founders are often wrong. We treat founder beliefs as a revisable
prior, learn how far to trust them from few outcomes, and explore enough to catch them when wrong.

## 2. Contributions and evidence

### C1. A Bayesian model with learned, pooled trust in founder beliefs (CPR-LP)
- Calibrated founder prior `logit p = a + b·logit(m)` with (a, b) integrated out; GP over accounts with
  segment and graph kernels; evidence-averaged segment deviation. ✅ `m0_sanity.md` (oracle gate passes)
- Learned trust b falls for shuffled beliefs (0.87 → 0.15) and turns negative for inverted founders. ✅ `placebo_bank.md`, `m0_sanity.md`
- Relational extension with learned coefficients on public product similarity (CPR-LP-rel). ✅ `rel_cprlp.md`
- Heavy-tailed (Student-t) segment prior: neutral normally, halves the error on a confidently wrong segment. ✅ `robust_ladder.md`, `explore_h4_robust.md`

### C2. Identifiability when the prior steers outreach (theory + simulation)
- Greedy targeting never corrects a wrong belief about an avoided segment (Prop. 1, proven). ✅ `theory-h4.md`, `explore_h4.md`
- Exploration restores consistency (Prop. 3, proven); Gaussian beliefs need contacts linear in the error,
  Student-t beliefs have bounded prior influence (Prop. 4). ✅ theory; 🟡 one step sketched
- Closed-form predictions match simulation (−0.59 vs −0.62 logits). ✅ `theory_check_h4.md`
- Caveat to state: our account-level Thompson sampler escapes "accidentally"; segment-level TS would not.

### C3. An amortized model that learns founder trust in context (PC-RFM)
- Founder inputs help strongly with good founders, in simulation and on real outcomes. ✅ `m1_pfn_v0_final.md`, `real_bank.md`
- Bad founders cost little once labels exist. ✅ same
- **Correction from placebo tests:** its edge over CPR-LP with bad founders is mostly better few-label
  learning, not better trust learning. ✅ `placebo_bank.md`, `diag_placebo_selection.md`
- Robust to most simulator shifts; weak when segment effects are pure interactions (fix in training prior). ✅ `stress_sim.md`; ⬜ retrained v0.1
- Founder-statement tokens with stated confidence. ⬜ implemented, training queued
- Relational versions v1 (features) / v2 (encoder). ⬜ training; pre-registered P1–P6 in `m2_preregistration.md`

### C4. Benchmarks measure memorization, not finding new customers (audit)
- A no-learning repeat + popularity heuristic reaches 93% / 88% of the best published RelBench MAP. ✅ `audit_relbench.md`
- 82–93% of test positives are novel pairs; heuristics and learned recommenders score ≈ popularity or below there. ✅ `audit_relbench.md`, `audit_models_subsample.md`
- Published ID-GNN per-slice scores. ⬜ needs Colab
- Pre-registered claim A1 ("most positives memorizable") failed — report it.

### C5. An evaluation protocol for settings without public outreach data
Ladder: simulation → sim-to-sim stress → real outcomes + simulated founder → real B2B backtest → prospective cohort.
- Real outcomes (UCI Bank Marketing), all three pre-registered predictions hold. ✅ `real_bank.md`
- Placebo priors as a standard control. ✅
- SBIR → USAspending first customers (real B2B, beliefs from firms' own pre-award text). 🟡 pipeline built, full pull running; ⬜ `sbir_eval.py` (pre-registered SB1–SB3)
- Prospective founder cohort. ⬜ drafts in `docs/cohort/`; needs sponsor + IRB

## 3. Related work positioning (from `novelty-check.md`)
- Adaptive trust in informative priors exists (EXNEX, robust MAP, power/commensurate priors): ours is a
  continuous calibration slope pooled across segments, inside a graph model, under prior-steered labels.
- PFNs4BO conditions on user priors with *stated* confidence; we infer unstated reliability.
- Selection bias from the decision policy is known in general; we give the specific identifiability statement.
- Benchmark critiques (HeaRT, ContextGNN locality) exist; ours targets the new-entity / novel-pair slice.

## 4. Biggest gaps before submission
1. Real founders (archival SBIR text is a proxy; the cohort is the real thing).
2. PC-RFM's founder-specific advantage over CPR-LP is unproven after the placebo correction.
3. Relational results (v1/v2) and the published-GNN audit slice.
4. EXNEX original not read; several theory references cited but not read in full.
5. All CPR-LP hyperparameters were set by hand on simulations; need a principled selection protocol.
