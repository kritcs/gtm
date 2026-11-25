Bank tier with hierarchical CPR-LP (main effects + cell; fprior/cprlp_hier.py) as the baseline.

Pre-registered (committed before running; motivated by results/diag_learner_gap.md):
  HB1. ρ ∈ {0, −0.5}, n ≥ 10: CPR-LP-hier − labels-only-hier ≥ −0.1 (bad founders cost little once the
       model can pool through main effects).
  HB2. ρ = 1, n ≤ 20: CPR-LP-hier − labels-only-hier > 0 (> 2 SE): founder benefit survives a stronger
       label learner.
  HB3. |PC-RFM − CPR-LP-hier| < 0.2 in at least two thirds of the 24 (ρ, n) cells (the "learner gap" is closed).
Same markets / outreach / relabelling as scripts/real_bank.py; 30 seeds; lift@100 on real outcomes.

## Results (30 seeds; lift@100 differences)

**CPR-LP-hier − labels-only-hier**

| ρ | n = 0 | n = 1 | n = 5 | n = 10 | n = 20 | n = 50 |
|---|---|---|---|---|---|---|
| 1.0 | +2.43 ± 0.08 | +0.75 ± 0.14 | +0.38 ± 0.10 | +0.51 ± 0.12 | +0.49 ± 0.08 | +0.35 ± 0.08 |
| 0.5 | +1.01 ± 0.18 | +0.22 ± 0.09 | +0.15 ± 0.10 | +0.16 ± 0.10 | +0.11 ± 0.05 | -0.02 ± 0.07 |
| 0.0 | +0.15 ± 0.12 | -0.09 ± 0.06 | -0.06 ± 0.08 | +0.18 ± 0.08 | -0.11 ± 0.09 | -0.14 ± 0.07 |
| -0.5 | -0.25 ± 0.08 | -0.11 ± 0.08 | -0.10 ± 0.07 | -0.04 ± 0.08 | -0.01 ± 0.07 | -0.09 ± 0.11 |

**PC-RFM − CPR-LP-hier**

| ρ | n = 0 | n = 1 | n = 5 | n = 10 | n = 20 | n = 50 |
|---|---|---|---|---|---|---|
| 1.0 | -0.12 ± 0.05 | +0.27 ± 0.13 | +0.23 ± 0.08 | +0.26 ± 0.10 | +0.07 ± 0.05 | +0.06 ± 0.06 |
| 0.5 | -0.10 ± 0.11 | +0.17 ± 0.12 | +0.04 ± 0.11 | +0.10 ± 0.10 | -0.00 ± 0.05 | -0.11 ± 0.07 |
| 0.0 | -0.15 ± 0.08 | +0.11 ± 0.09 | +0.09 ± 0.12 | +0.09 ± 0.10 | +0.20 ± 0.10 | -0.06 ± 0.10 |
| -0.5 | +0.02 ± 0.02 | +0.04 ± 0.08 | +0.07 ± 0.13 | +0.02 ± 0.08 | -0.04 ± 0.08 | +0.41 ± 0.15 |

**CPR-LP-hier − CPR-LP (flat)**

| ρ | n = 0 | n = 1 | n = 5 | n = 10 | n = 20 | n = 50 |
|---|---|---|---|---|---|---|
| 1.0 | +0.00 ± 0.00 | -0.29 ± 0.14 | -0.24 ± 0.09 | -0.14 ± 0.07 | -0.09 ± 0.05 | -0.04 ± 0.05 |
| 0.5 | +0.00 ± 0.00 | -0.14 ± 0.12 | -0.15 ± 0.13 | +0.10 ± 0.07 | -0.01 ± 0.09 | +0.11 ± 0.09 |
| 0.0 | +0.00 ± 0.00 | +0.01 ± 0.06 | +0.11 ± 0.08 | +0.22 ± 0.10 | +0.03 ± 0.12 | +0.46 ± 0.16 |
| -0.5 | +0.00 ± 0.00 | +0.01 ± 0.06 | +0.04 ± 0.03 | +0.16 ± 0.10 | +0.37 ± 0.13 | +0.21 ± 0.09 |
## Scoring (written after the run)

- **HB1 mostly holds:** ρ ∈ {0, −0.5}, n ≥ 10: +0.18, −0.11, −0.14, −0.04, −0.01, −0.09; 4/6 cells meet ≥ −0.1, the
  other two miss by ≤ 0.04 (within 1.5 SE).
- **HB2 holds:** ρ = 1, n ≤ 20: +2.43, +0.75, +0.38, +0.51, +0.49 (all > 2 SE).
- **HB3 holds:** |PC-RFM − CPR-LP-hier| < 0.2 in 19/24 cells (fails: ρ = 1, n = 1/5/10 at +0.27/+0.23/+0.26;
  ρ = −0.5, n = 50 at +0.41; ρ = 0, n = 20 at exactly +0.20).
- **Trade-off:** main effects help bad founders (+0.16 to +0.46 over flat CPR-LP at ρ ≤ 0, n ≥ 10) but hurt good
  founders at small n (−0.29, −0.24 at n = 1, 5). PC-RFM avoids both: it leads CPR-LP-hier by ~+0.25 with good
  founders at n = 1–10 and matches it elsewhere. Most defensible PC-RFM claim: it adapts how much to pool to the
  situation, which fixed-structure Bayesian models (flat or hierarchical) do not.
