Sim-to-sim stress test: PC-RFM v0 (simulator-trained) vs CPR-LP on deliberately shifted worlds.

Pre-registered (written before the full run):
  (a) Under S1 (heavy-tailed founder errors) both models lose lift vs S0, but PC-RFM loses less
      than CPR-LP (mean over rho in {1, 0.5}, n >= 5).
  (b) Under S6 (base rate 0.01 / 0.30, outside PC-RFM's training range 0.02-0.2) PC-RFM degrades
      more than CPR-LP relative to S0 (mean of PC-RFM − CPR-LP drops vs S0).
  (c) PC-RFM − CPR-LP with bad founders (rho <= 0, n >= 10) stays positive in at least 4 of the
      6 shift families (S1, S2, S3, S4, S5, S6 — S6 counts as positive only if both S6a and S6b are).
Shifts are defined in fprior/sim_shift.py. Metric: expected lift@100 on uncontacted accounts.
Lift is relative to the market's own base rate, so it is comparable across base rates only roughly.

## Summary (10 seeds; mean over cells within seed, ± SE over seeds)

PC-RFM − CPR-LP by founder quality and label count; worst cell of each model vs labels-only (over all ρ, n).

| shift | base | good founder, n≤5 | good, n≥10 | bad founder, n≤5 | bad, n≥10 | CPR-LP − labels worst cell | PC-RFM − labels worst cell |
|---|---|---|---|---|---|---|---|
| S0_default | 0.080 | -0.06 ± 0.06 | -0.11 ± 0.05 | +0.23 ± 0.09 | +0.24 ± 0.12 | -0.55 (ρ=-0.5, n=0) | -0.55 (ρ=-0.5, n=0) |

## Mean lift by shift (ρ ∈ {1, 0.5}, n ≥ 5) — for predictions (a)/(b)

| shift | labels-only | CPR-LP | PC-RFM | PC-RFM − CPR-LP |
|---|---|---|---|---|
| S0_default | 2.64 | 3.22 | 3.13 | -0.09 ± 0.06 |

## Full tables (lift@100, mean ± SE over seeds)

### S0_default

**ρ = 1.0**

| n | contacts | prior_only | labels_only | cpr_lp | pcrfm | pcrfm_nofounder | PC-RFM − CPR-LP | PC-RFM − no-founder | CPR − labels |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 3.66 ± 0.20 | 1.02 ± 0.03 | 3.67 ± 0.20 | 3.63 ± 0.19 | 1.02 ± 0.13 | -0.04 ± 0.07 | 2.61 ± 0.22 | 2.65 ± 0.21 |
| 1 | 5 | 3.67 ± 0.19 | 2.34 ± 0.36 | 3.46 ± 0.16 | 3.51 ± 0.19 | 2.62 ± 0.43 | 0.04 ± 0.04 | 0.89 ± 0.38 | 1.12 ± 0.32 |
| 5 | 22 | 3.71 ± 0.20 | 2.86 ± 0.20 | 3.73 ± 0.19 | 3.57 ± 0.20 | 2.84 ± 0.41 | -0.16 ± 0.08 | 0.73 ± 0.35 | 0.87 ± 0.23 |
| 10 | 52 | 3.66 ± 0.19 | 2.61 ± 0.25 | 3.58 ± 0.21 | 3.49 ± 0.20 | 3.14 ± 0.19 | -0.09 ± 0.08 | 0.35 ± 0.18 | 0.97 ± 0.23 |
| 20 | 98 | 3.60 ± 0.15 | 2.82 ± 0.28 | 3.46 ± 0.18 | 3.48 ± 0.18 | 3.03 ± 0.22 | 0.02 ± 0.06 | 0.45 ± 0.22 | 0.64 ± 0.25 |
| 50 | 261 | 3.54 ± 0.13 | 2.81 ± 0.22 | 3.45 ± 0.13 | 3.37 ± 0.14 | 2.61 ± 0.33 | -0.08 ± 0.06 | 0.76 ± 0.31 | 0.65 ± 0.16 |

**ρ = 0.5**

| n | contacts | prior_only | labels_only | cpr_lp | pcrfm | pcrfm_nofounder | PC-RFM − CPR-LP | PC-RFM − no-founder | CPR − labels |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 2.02 ± 0.19 | 1.02 ± 0.03 | 2.06 ± 0.18 | 1.90 ± 0.14 | 1.02 ± 0.13 | -0.16 ± 0.13 | 0.88 ± 0.20 | 1.04 ± 0.18 |
| 1 | 7 | 2.37 ± 0.37 | 1.88 ± 0.30 | 2.22 ± 0.33 | 2.05 ± 0.31 | 2.00 ± 0.31 | -0.17 ± 0.11 | 0.06 ± 0.14 | 0.34 ± 0.13 |
| 5 | 28 | 2.06 ± 0.40 | 2.44 ± 0.35 | 2.74 ± 0.38 | 2.86 ± 0.38 | 2.57 ± 0.37 | 0.12 ± 0.12 | 0.29 ± 0.16 | 0.30 ± 0.23 |
| 10 | 94 | 1.74 ± 0.26 | 2.05 ± 0.35 | 2.57 ± 0.27 | 2.38 ± 0.32 | 1.97 ± 0.36 | -0.18 ± 0.11 | 0.41 ± 0.39 | 0.51 ± 0.20 |
| 20 | 140 | 2.57 ± 0.21 | 2.22 ± 0.35 | 2.88 ± 0.25 | 2.63 ± 0.30 | 2.50 ± 0.37 | -0.25 ± 0.17 | 0.13 ± 0.26 | 0.66 ± 0.30 |
| 50 | 382 | 2.28 ± 0.31 | 3.36 ± 0.28 | 3.38 ± 0.28 | 3.28 ± 0.27 | 3.00 ± 0.27 | -0.10 ± 0.08 | 0.27 ± 0.17 | 0.01 ± 0.03 |

**ρ = 0.0**

| n | contacts | prior_only | labels_only | cpr_lp | pcrfm | pcrfm_nofounder | PC-RFM − CPR-LP | PC-RFM − no-founder | CPR − labels |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.67 ± 0.09 | 1.02 ± 0.03 | 0.67 ± 0.09 | 1.24 ± 0.37 | 1.02 ± 0.13 | 0.57 ± 0.40 | 0.22 ± 0.37 | -0.35 ± 0.08 |
| 1 | 23 | 1.13 ± 0.27 | 1.95 ± 0.31 | 2.00 ± 0.33 | 1.93 ± 0.31 | 1.76 ± 0.32 | -0.07 ± 0.21 | 0.17 ± 0.20 | 0.05 ± 0.21 |
| 5 | 77 | 1.11 ± 0.08 | 1.46 ± 0.26 | 1.30 ± 0.16 | 1.33 ± 0.24 | 1.18 ± 0.16 | 0.03 ± 0.17 | 0.14 ± 0.11 | -0.16 ± 0.19 |
| 10 | 132 | 0.99 ± 0.25 | 1.80 ± 0.29 | 1.84 ± 0.33 | 1.61 ± 0.32 | 1.93 ± 0.32 | -0.23 ± 0.27 | -0.33 ± 0.25 | 0.04 ± 0.24 |
| 20 | 255 | 1.52 ± 0.30 | 2.44 ± 0.31 | 2.44 ± 0.33 | 2.51 ± 0.28 | 2.09 ± 0.30 | 0.07 ± 0.15 | 0.42 ± 0.28 | 0.00 ± 0.06 |
| 50 | 620 | 0.96 ± 0.13 | 2.66 ± 0.27 | 2.71 ± 0.26 | 2.90 ± 0.21 | 2.62 ± 0.25 | 0.19 ± 0.15 | 0.28 ± 0.16 | 0.05 ± 0.04 |

**ρ = -0.5**

| n | contacts | prior_only | labels_only | cpr_lp | pcrfm | pcrfm_nofounder | PC-RFM − CPR-LP | PC-RFM − no-founder | CPR − labels |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.47 ± 0.11 | 1.02 ± 0.03 | 0.47 ± 0.11 | 0.46 ± 0.12 | 1.02 ± 0.13 | -0.00 ± 0.01 | -0.55 ± 0.15 | -0.55 ± 0.11 |
| 1 | 25 | 0.35 ± 0.08 | 0.75 ± 0.16 | 0.58 ± 0.11 | 1.04 ± 0.36 | 0.91 ± 0.15 | 0.47 ± 0.39 | 0.13 ± 0.32 | -0.17 ± 0.09 |
| 5 | 103 | 0.32 ± 0.07 | 1.47 ± 0.29 | 1.30 ± 0.29 | 1.70 ± 0.26 | 1.75 ± 0.22 | 0.40 ± 0.16 | -0.06 ± 0.30 | -0.16 ± 0.17 |
| 10 | 252 | 0.35 ± 0.03 | 1.65 ± 0.32 | 1.63 ± 0.26 | 2.18 ± 0.29 | 1.73 ± 0.30 | 0.55 ± 0.33 | 0.45 ± 0.25 | -0.01 ± 0.16 |
| 20 | 484 | 0.40 ± 0.06 | 2.13 ± 0.27 | 1.97 ± 0.28 | 2.54 ± 0.25 | 2.13 ± 0.26 | 0.57 ± 0.22 | 0.41 ± 0.30 | -0.16 ± 0.10 |
| 50 | 849 | 0.56 ± 0.11 | 2.37 ± 0.24 | 2.58 ± 0.19 | 2.86 ± 0.23 | 2.52 ± 0.19 | 0.28 ± 0.21 | 0.33 ± 0.21 | 0.20 ± 0.18 |
