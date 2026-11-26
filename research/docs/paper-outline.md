# Paper outline (working draft, updated 2026-11-25)

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
- That edge is mostly pooling through industry/size main effects (not noise level, not the trust prior). ✅
  `diag_learner_gap.md`, `matched_prior.md`. Against a hierarchical CPR-LP the two are within 0.2 in 19/24 cells;
  PC-RFM still leads by ~+0.25 with good founders at n = 1–10, because a fixed hierarchy over-pools there. ✅
  `real_bank_hier.md`. **Refined claim: PC-RFM adapts how much to pool to the situation.**
- Against its own no-information mode, the founder benefit is large with good founders and shrinks with labels. ✅
  `bank_noise_control.md`. The "inverted founders under-discounted" cost (~0.2–0.3) appears against an
  account-scrambled null but **vanishes against a segment-shuffled null** (`bank_noise_control_tok.md`): a wrong
  founder is no worse than a random one; it is just harder to spot than obvious noise. Report both nulls.
- Training on founders down to ρ = −1 turns inverted founders into a *gain* (+0.6–0.9) once labels exist, at a
  small noisy cost with good founders; kept as an "adversarial founder" ablation, default stays ρ ≥ −0.5. ✅
  `round2_preregistration.md` (Q3)
- Robust to most simulator shifts; a stable ~−0.13 gap to CPR-LP when segments are pure interactions. Random
  segment structure in training closes ~35% of it; an explicit segment embedding closes none; it does not depend
  on training seed. ~~Likely cause: the graph~~ **Not the graph** (round 4): CPR-LP gains nothing from its graph
  term in the default simulator (graph on vs off or permuted: within ±0.015), and a graph-aware PC-RFM (v0g)
  ignores its graph features, which carry ~0.05 correlation with the truth. The gap is in segment-level
  inference. ✅ `stress_sim*.md`, `round2/3/4_preregistration.md`, `eval_v0g.md`
- **The graph helps only when it is strong and smooth** (exploratory): CPR-LP's oracle graph adds +0.43 lift at
  graph share 0.7, γ = 50, and nothing in 23/24 cells of a regime sweep (`diag_graph_regime.md`). No simulated
  result in the paper may be credited to the graph. Whether real B2B graphs are strong is an RDL-tier question.
- Founder-statement tokens: stated confidence is used in the right direction (reversing it costs up to 1.9 lift)
  and matters most for mediocre founders with ≤ 1 label; "no idea" answers did not help (T1, T3 failed as
  pre-registered). With neutral statements it matches v0. Its extra bank lift with bad founders is learner
  quality, not founder use. ✅ `m2_pfn_tok.md`, `real_bank_tok.md`, `bank_noise_control_tok.md`
- Relational v1: product relations help strongly in familiar categories (+0.8–1.9 vs v0) and v1 matches or beats
  CPR-LP / CPR-LP-rel in 66/72 cells; founder value is *not* larger in new categories (P1 fails) and a useless
  founder costs up to 0.33 at n = 5 (P3 fails). Weak spot: new categories with zero labels (−1.0 to −1.2 vs v0).
  ✅ `m2_preregistration.md`; all verdicts replicate with a second training seed ✅ `round3_preregistration.md` (S4)
- Learned encoder v2: little gain where predicted (P5 fails) but fixes v1's zero-label new-category weak spot
  (+0.7–0.8), and with v2 founder value *is* larger in new categories (+0.50 vs +0.32). ✅ `m2_pfn_v2*.md`, round 2 Q4

### C4. Benchmarks measure memorization (section draft: `paper/sec-audit.md`), not finding new customers (audit)
- A no-learning repeat + popularity heuristic reaches 93% / 88% of the best published RelBench MAP. ✅ `audit_relbench.md`
- 82–93% of test positives are novel pairs; heuristics and learned recommenders score ≈ popularity or below there. ✅ `audit_relbench.md`, `audit_models_subsample.md`
- Published ID-GNN (reproduced exactly, 2.81) gets 99.8% of its hits from repeat purchases; novel-pair MAP 0.00 (popularity 0.58), cold users 0.00. Its whole edge over the heuristic is better ordering of repeats. ✅ `idgnn_slices.md`
- Across six RelBench recommendation tasks (adds rel-trial ×2, rel-stack ×2) repeat + popularity reaches 40–107% of the best published MAP (beats ID-GNN on site-sponsor-run); heuristics ≈ 0 on novel pairs everywhere. RelBench's own Past Visit baseline understates memorisability on rel-stack (1.42 vs our 5.10). ✅ `audit_relbench_ext.md` (E1–E3 fail as pre-registered, E4 holds)
- ID-GNN on rel-stack (reproduced: 13.38, 10.89): on user-post-comment 91% of its hits are the user's own threads (posts table), and a no-learning own-thread heuristic reaches 78% of it (exploratory); on post-post-related its score is the 8% re-linked pairs (repeat slice 93.5, novel 3.28). "Neighbourhood" is schema-specific. ✅ `idgnn_ext_scoring.md`; rel-trial ID-GNN could not be reproduced (NaN training on the current stack, twice) — reported as such
- Pre-registered claim A1 ("most positives memorizable") failed — report it.

### C5. An evaluation protocol for settings without public outreach data
Ladder: simulation → sim-to-sim stress → real outcomes + simulated founder → real B2B backtest → prospective cohort.
- Real outcomes (UCI Bank Marketing), all three pre-registered predictions hold. ✅ `real_bank.md`
- Second real tier (Hillstrom e-mail test, randomized, weak segments): sim-trained PC-RFM within 0.10 lift of CPR-LP in 24/24 cells (H3 holds); founder gains at ρ = 1 are smaller and noisier (H1 3–4/5 cells), inverted-founder cost ≈ 0.2 at n = 5 (H2 fails narrowly). ✅ `real_hillstrom_v01.md`
- Prospective founder cohort: power simulation says the randomized-slice design and 40–50 companies are needed. 🟡 `cohort/analysis-plan.md`, `cohort_power.md`
- Placebo priors as a standard control. ✅
- SBIR → USAspending first customers (real B2B, beliefs from firms' own pre-award text). ✅ `sbir_eval.md`: negative for keyword priors (SB1 fails, SB2 partly, SB3 point estimate only); on new-agency purchases (56%) popularity is best. Labels hand-checked (0/50 R&D). Base rate ~9× below the sim range (`sim_calibration.md`).
- Prospective founder cohort. ⬜ drafts in `docs/cohort/`; needs sponsor + IRB

## 3. Related work positioning (from `novelty-check.md`)
- Adaptive trust in informative priors exists (EXNEX, robust MAP, power/commensurate priors): ours is a
  continuous calibration slope pooled across segments, inside a graph model, under prior-steered labels.
- PFNs4BO conditions on user priors with *stated* confidence; we infer unstated reliability.
- Selection bias from the decision policy is known in general; we give the specific identifiability statement.
- Benchmark critiques (HeaRT, ContextGNN locality) exist; ours targets the new-entity / novel-pair slice.

## 4. Biggest gaps before submission
1. Real founders (archival SBIR text is a proxy; the cohort is the real thing).
2. PC-RFM's founder-specific advantage over CPR-LP is unproven after the placebo correction (confirmed again for
   the token model). The defensible PC-RFM claims are: adaptive pooling, relational transfer, and use of stated confidence.
3. ID-GNN on rel-trial not reproduced (NaN loss). The graph-aware v0 is done: the S2 gap is not the graph (round 4), so its cause is open.
4. EXNEX original not read; several theory references cited but not read in full.
5. All CPR-LP hyperparameters were set by hand on simulations; need a principled selection protocol.
