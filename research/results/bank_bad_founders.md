Bad founders on UCI Bank Marketing: Gaussian CPR-LP vs Student-t CPR-LP vs PC-RFM (vs labels-only).

Follow-up to results/bank_noise_control.md (NC3 failed: PC-RFM loses ~0.2–0.3 to its own no-information
mode with an inverted founder at n = 10–20). Same markets/outreach as scripts/real_bank.py.

Pre-registered (committed before running):
  BF1. ρ = −0.5, n ≥ 5: Student-t CPR-LP − labels-only ≥ Gaussian CPR-LP − labels-only (robust prior
       discounts a systematically wrong founder at least as well).
  BF2. ρ = 1, all n: Student-t CPR-LP within 0.1 of Gaussian CPR-LP (no cost with a good founder).
Exploratory: PC-RFM in the same cells.

## Results (30 seeds): method − labels-only, lift@100

**ρ = 1.0**

| method | n = 0 | n = 1 | n = 5 | n = 10 | n = 20 | n = 50 |
|---|---|---|---|---|---|---|
| cpr_gauss | +2.43 ± 0.08 | +1.19 ± 0.15 | +0.67 ± 0.15 | +0.79 ± 0.16 | +0.73 ± 0.11 | +0.60 ± 0.09 |
| cpr_t | +2.43 ± 0.08 | +1.19 ± 0.15 | +0.65 ± 0.14 | +0.75 ± 0.16 | +0.73 ± 0.11 | +0.57 ± 0.08 |
| pcrfm | +2.31 ± 0.08 | +1.18 ± 0.15 | +0.66 ± 0.13 | +0.92 ± 0.16 | +0.71 ± 0.11 | +0.62 ± 0.09 |
| cpr_t − cpr_gauss | +0.00 ± 0.00 | +0.00 ± 0.00 | -0.03 ± 0.04 | -0.05 ± 0.04 | +0.00 ± 0.00 | -0.03 ± 0.02 |

**ρ = 0.0**

| method | n = 0 | n = 1 | n = 5 | n = 10 | n = 20 | n = 50 |
|---|---|---|---|---|---|---|
| cpr_gauss | +0.15 ± 0.12 | +0.08 ± 0.06 | +0.02 ± 0.08 | +0.05 ± 0.09 | +0.03 ± 0.08 | -0.12 ± 0.16 |
| cpr_t | +0.15 ± 0.12 | +0.07 ± 0.07 | +0.01 ± 0.08 | +0.10 ± 0.08 | +0.02 ± 0.08 | +0.00 ± 0.13 |
| pcrfm | -0.00 ± 0.09 | +0.20 ± 0.10 | +0.22 ± 0.11 | +0.36 ± 0.11 | +0.26 ± 0.12 | +0.29 ± 0.13 |
| cpr_t − cpr_gauss | +0.00 ± 0.00 | -0.01 ± 0.02 | -0.01 ± 0.01 | +0.05 ± 0.07 | -0.01 ± 0.03 | +0.12 ± 0.09 |

**ρ = -0.5**

| method | n = 0 | n = 1 | n = 5 | n = 10 | n = 20 | n = 50 |
|---|---|---|---|---|---|---|
| cpr_gauss | -0.25 ± 0.08 | +0.01 ± 0.06 | -0.00 ± 0.09 | -0.02 ± 0.12 | -0.14 ± 0.12 | +0.15 ± 0.15 |
| cpr_t | -0.25 ± 0.08 | +0.01 ± 0.06 | -0.00 ± 0.09 | +0.02 ± 0.12 | -0.15 ± 0.12 | +0.20 ± 0.14 |
| pcrfm | -0.23 ± 0.07 | +0.06 ± 0.08 | +0.11 ± 0.10 | +0.16 ± 0.13 | +0.20 ± 0.10 | +0.77 ± 0.13 |
| cpr_t − cpr_gauss | +0.00 ± 0.00 | -0.00 ± 0.00 | +0.00 ± 0.00 | +0.04 ± 0.01 | -0.01 ± 0.02 | +0.06 ± 0.07 |
## Scoring (written after the run)

- **BF1 holds, trivially.** ρ = −0.5, n ≥ 5: Student-t − Gaussian = +0.00, +0.04, −0.01, +0.06. Both CPR-LP
  versions sit at ≈ labels-only. An inverted founder is wrong everywhere in the same direction, which the learned
  trust slope b already absorbs; the Student-t segment prior is for isolated badly-wrong segments (H4), not this.
- **BF2 holds.** ρ = 1: Student-t within 0.05 of Gaussian at every n.
- **Exploratory:** PC-RFM − labels-only with bad founders is +0.2 to +0.8 (its label-learning strength,
  `diag_placebo_selection.md`), while against its own no-information mode it loses ~0.2 (`bank_noise_control.md`).
- **Lesson:** two kinds of founder error need two mechanisms — systematic (slope b) vs isolated (heavy tails).
