Diagnostic for the placebo-prior failure (results/placebo_bank.md, PL2 for PC-RFM).

PC-RFM beats labels-only by ~0.3–0.45 lift even with shuffled (uninformative) founder beliefs.
Hypothesis (selection awareness): the beliefs told the model WHY the contacted accounts were
contacted, which helps it interpret biased labels. Test: outreach driven by shuffled beliefs X,
model shown beliefs X ("matched") vs an independent shuffle Y ("decoupled"), vs no founder input.
  matched ≈ decoupled  -> not selection awareness (something else about having a founder input)
  matched >> decoupled -> selection awareness
Also: outreach uniformly random (no founder steering), model shown shuffled beliefs ("random-outreach").

Lift@100 minus CPR labels-only (same contacts), 30 seeds.

| condition | n = 5 | n = 10 | n = 20 | n = 50 |
|---|---|---|---|---|
| matched | +0.22 ± 0.12 | +0.27 ± 0.12 | +0.38 ± 0.13 | +0.60 ± 0.13 |
| decoupled | +0.26 ± 0.09 | +0.16 ± 0.15 | -0.07 ± 0.17 | +0.35 ± 0.15 |
| nofounder | -0.02 ± 0.11 | -0.01 ± 0.09 | -0.01 ± 0.12 | +0.09 ± 0.12 |
| rand_outreach_shown | +0.04 ± 0.18 | +0.13 ± 0.12 | +0.50 ± 0.18 | +0.36 ± 0.14 |
| rand_outreach_nofounder | -0.17 ± 0.13 | -0.12 ± 0.12 | +0.21 ± 0.11 | -0.02 ± 0.13 |
## Follow-up: segment-ID hypothesis (run after the table above)

Same markets and outreach (driven by shuffled beliefs X), PC-RFM shown X's values scrambled across *accounts*
(no segment-ID information): +0.31 ± 0.10, +0.19 ± 0.09, +0.40 ± 0.13, +0.64 ± 0.11 at n = 5, 10, 20, 50 —
the same as per-segment shuffles. **Segment-ID hypothesis rejected. Selection-awareness hypothesis rejected** (decoupled ≈ matched).

## Interpretation

- Flat beliefs ≈ 0 gain; any *varied but wrong* beliefs ≈ +0.2–0.6; a genuinely useless founder (ρ = 0) ≈ +0.1–0.4;
  the separately trained no-founder model ≈ 0.
- Most consistent reading: varied wrong beliefs are contradicted by labels, so PC-RFM switches to relying on labels,
  and in that mode it is a **stronger few-label learner than CPR-LP's GP** (~0.3 lift on this data). Flat beliefs look
  like a confident uniform prior and suppress that switch. The separately trained no-founder checkpoint is a weaker
  learner, so "PC-RFM − no-founder" overstates the founder benefit.
- Consequences: (1) "PC-RFM beats CPR-LP with bad founders" should be reported as mostly a learner-quality effect,
  not better trust learning; (2) PC-RFM does correctly ignore useless founders (ρ = 0 ≈ its own noise-belief mode);
  (3) new standard control for every PC-RFM evaluation: the same model shown noise beliefs (per-account random).
