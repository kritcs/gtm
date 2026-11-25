Placebo priors on UCI Bank Marketing (evaluation-brainstorm.md idea #3).

If the gain over labels-only really comes from the founder's knowledge, it must disappear when the
beliefs are destroyed while keeping their format:
  true      — rho = 1 founder (segment rates from split A)
  shuffled  — the same belief values, randomly permuted across segments (same spread, no information)
  flat      — every segment believed at the overall base rate (no spread, no information)

Pre-registered (committed before the run):
  PL1. For true beliefs, CPR-LP and PC-RFM beat labels-only (replicates R1).
  PL2. For shuffled and flat beliefs at n >= 5, |method − labels-only| < 0.2 lift, i.e. the gain
       vanishes; shuffled may hurt at n = 0–1 (a confident wrong prior with no labels).
  PL3. Learned trust b (CPR-LP) is lower for shuffled than for true beliefs at every n >= 5.
Metric: lift@100 on uncontacted accounts with real outcomes; 30 seeds; same markets and
relabelling control as scripts/real_bank.py.

## Results (30 seeds)

Paired lift@100 differences vs labels-only, and CPR-LP's learned trust b.

| beliefs | n | CPR-LP − labels | PC-RFM − labels | learned b |
|---|---|---|---|---|
| true | 0 | +2.34 ± 0.08 | +2.22 ± 0.08 | 0.50 |
| true | 1 | +1.42 ± 0.13 | +1.10 ± 0.14 | 0.60 |
| true | 5 | +1.11 ± 0.17 | +1.19 ± 0.17 | 0.69 |
| true | 10 | +0.63 ± 0.10 | +0.61 ± 0.10 | 0.83 |
| true | 20 | +0.74 ± 0.10 | +0.91 ± 0.11 | 0.80 |
| true | 50 | +0.52 ± 0.09 | +0.52 ± 0.09 | 0.87 |
| shuffled | 0 | +0.03 ± 0.11 | +0.05 ± 0.12 | 0.50 |
| shuffled | 1 | +0.07 ± 0.08 | +0.16 ± 0.07 | 0.38 |
| shuffled | 5 | +0.11 ± 0.07 | +0.30 ± 0.11 | 0.26 |
| shuffled | 10 | -0.00 ± 0.10 | +0.37 ± 0.10 | 0.21 |
| shuffled | 20 | +0.05 ± 0.09 | +0.30 ± 0.10 | 0.17 |
| shuffled | 50 | -0.03 ± 0.07 | +0.46 ± 0.13 | 0.15 |
| flat | 0 | -0.04 ± 0.07 | +0.64 ± 0.17 | 0.50 |
| flat | 1 | -0.00 ± 0.05 | +0.04 ± 0.06 | 0.50 |
| flat | 5 | +0.00 ± 0.06 | -0.08 ± 0.12 | 0.50 |
| flat | 10 | +0.02 ± 0.07 | +0.10 ± 0.10 | 0.50 |
| flat | 20 | -0.01 ± 0.09 | +0.02 ± 0.11 | 0.50 |
| flat | 50 | -0.01 ± 0.04 | +0.01 ± 0.10 | 0.50 |
## Scoring (written after the run)

- **PL1 holds** (true beliefs: both methods +0.5 to +2.3 over labels-only).
- **PL2 holds for CPR-LP, fails for PC-RFM.** CPR-LP with shuffled/flat beliefs: all |Δ| ≤ 0.11. PC-RFM with
  *shuffled* beliefs: +0.30 / +0.37 / +0.30 / +0.46 at n = 5 / 10 / 20 / 50 (> 2 SE), about the same size as its
  "advantage with bad founders" over CPR-LP in results/real_bank.md. The no-founder PC-RFM is ≈ labels-only at
  ρ ≤ 0, so this is not just a better label learner. Flat beliefs at n = 0 (+0.64) is the category-preference
  effect noted in real_bank.md.
- **PL3 holds:** CPR-LP's learned b falls to 0.15–0.26 for shuffled beliefs vs 0.69–0.87 for true ones.
- **Implication:** part of PC-RFM's bad-founder advantage is not trust learning. Leading hypothesis: selection
  awareness (beliefs explain who was contacted). Diagnostic: scripts/diag_placebo_selection.py.
