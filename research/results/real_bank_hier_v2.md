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
| 1.0 | +2.43 ± 0.08 | +0.82 ± 0.14 | +0.44 ± 0.11 | +0.56 ± 0.12 | +0.56 ± 0.09 | +0.43 ± 0.09 |
| 0.5 | +1.01 ± 0.18 | +0.22 ± 0.09 | +0.21 ± 0.12 | +0.19 ± 0.10 | +0.13 ± 0.05 | -0.03 ± 0.05 |
| 0.0 | +0.15 ± 0.12 | -0.11 ± 0.07 | -0.03 ± 0.09 | +0.11 ± 0.06 | -0.02 ± 0.09 | -0.17 ± 0.08 |
| -0.5 | -0.25 ± 0.08 | -0.13 ± 0.09 | -0.05 ± 0.05 | +0.02 ± 0.08 | -0.08 ± 0.09 | -0.02 ± 0.13 |

**PC-RFM − CPR-LP-hier**

| ρ | n = 0 | n = 1 | n = 5 | n = 10 | n = 20 | n = 50 |
|---|---|---|---|---|---|---|
| 1.0 | -0.12 ± 0.05 | +0.21 ± 0.10 | +0.16 ± 0.07 | +0.22 ± 0.09 | +0.05 ± 0.04 | +0.01 ± 0.04 |
| 0.5 | -0.10 ± 0.11 | +0.17 ± 0.11 | -0.01 ± 0.10 | +0.08 ± 0.10 | -0.02 ± 0.06 | -0.09 ± 0.08 |
| 0.0 | -0.15 ± 0.08 | +0.13 ± 0.09 | +0.13 ± 0.11 | +0.12 ± 0.09 | +0.16 ± 0.10 | -0.01 ± 0.09 |
| -0.5 | +0.02 ± 0.02 | +0.07 ± 0.09 | +0.08 ± 0.12 | -0.00 ± 0.08 | +0.04 ± 0.08 | +0.40 ± 0.15 |

**CPR-LP-hier − CPR-LP (flat)**

| ρ | n = 0 | n = 1 | n = 5 | n = 10 | n = 20 | n = 50 |
|---|---|---|---|---|---|---|
| 1.0 | +0.00 ± 0.00 | -0.22 ± 0.10 | -0.18 ± 0.07 | -0.09 ± 0.05 | -0.07 ± 0.05 | +0.02 ± 0.03 |
| 0.5 | +0.00 ± 0.00 | -0.14 ± 0.11 | -0.10 ± 0.10 | +0.12 ± 0.06 | +0.01 ± 0.08 | +0.08 ± 0.07 |
| 0.0 | +0.00 ± 0.00 | -0.01 ± 0.06 | +0.08 ± 0.07 | +0.18 ± 0.09 | +0.06 ± 0.11 | +0.41 ± 0.15 |
| -0.5 | +0.00 ± 0.00 | -0.02 ± 0.05 | +0.03 ± 0.03 | +0.19 ± 0.09 | +0.29 ± 0.12 | +0.21 ± 0.07 |

> v2 = rerun of scripts/real_bank_hier.py after adding the flat option (main-effect scale 0) to the evidence grid
> (see results/m0_hier.md). Pre-registration HB1–HB3 refers to v1 (`real_bank_hier.md`); this file is exploratory.
> Changes vs v1: good-founder over-pooling cost at n = 1 is −0.22 (v1: −0.29); |PC-RFM − CPR-LP-hier| < 0.2 in 21/24
> cells (exceptions: ρ = 1, n = 1 / 10 at +0.21 / +0.22; ρ = −0.5, n = 50 at +0.40).
