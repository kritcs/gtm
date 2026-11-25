# Numerical checks for docs/theory-h4.md

## Part A — formulas vs simulations of the simplified model

**P1 (greedy).** K=8, B=5, T=200, ω=0.5, σ=2.0, Δ=3.0, margin γ=2.80. k* never sampled in 200/200 runs; P1 bound on the failure probability for the runner-up segment alone: 5.77e-25 (union over TB values of n). Mean per-contact regret 0.387 (≥ θ_k* − θ_runner-up = 0.200 whenever k* is not sampled; larger here because greedy exhausts the runner-up segment and moves down) → linear regret.

**P2 (Thompson escape).** One round, prior only. Segment-level TS (one draw per segment, top-1 segment): P(k* chosen) vs Φ(−z)-type bound. Account-level marginal TS (one draw per account, top-B accounts, as in fprior/explore.py): expected k* accounts in the batch vs the order-statistic formula.
- segment TS, ω=0.5, gap=1.5: simulated 1.69e-02, Φ(−z) = 1.69e-02 (z = 2.12), expected first-sampling round 1/Φ(−z) = 59
- segment TS, ω=0.5, gap=2.5: simulated 2.80e-04, Φ(−z) = 2.03e-04 (z = 3.54), expected first-sampling round 1/Φ(−z) = 4915
- segment TS, ω=1.0, gap=2.5: simulated 3.83e-02, Φ(−z) = 3.85e-02 (z = 1.77), expected first-sampling round 1/Φ(−z) = 26
- account TS, per-account sd=0.6, gap=2.5: simulated E[#k* accounts/round] = 0.000, formula = 0.000; P(≥1) simulated 0.000, Poisson approx 0.000
- account TS, per-account sd=1.3, gap=2.5: simulated E[#k* accounts/round] = 0.001, formula = 0.001; P(≥1) simulated 0.001, Poisson approx 0.001
- account TS, per-account sd=1.3, gap=1.5: simulated E[#k* accounts/round] = 0.018, formula = 0.019; P(≥1) simulated 0.018, Poisson approx 0.019

**P4 (washout).** σ²=5.0 per observation (logit scale), ω=0.5, ν=3. Expected posterior-mean error (averaged over ȳ ~ N(θ, σ²/n)) — Gaussian formula vs exact; Student-t exact vs asymptotic formula.

| Δ | n | Gaussian formula | Gaussian exact (sim) | Student-t exact | Student-t asymptotic |
|---|---|---|---|---|---|
| 1.0 | 10 | -0.667 | -0.657 | -0.543 | -1.143 |
| 1.0 | 40 | -0.333 | -0.320 | -0.221 | -0.286 |
| 1.0 | 160 | -0.111 | -0.107 | -0.057 | -0.071 |
| 3.0 | 10 | -2.000 | -1.998 | -0.720 | -0.615 |
| 3.0 | 40 | -1.000 | -0.990 | -0.136 | -0.154 |
| 3.0 | 160 | -0.333 | -0.335 | -0.012 | -0.038 |
| 6.0 | 10 | -4.000 | -4.010 | -0.433 | -0.327 |
| 6.0 | 40 | -2.000 | -1.993 | -0.124 | -0.082 |
| 6.0 | 160 | -0.667 | -0.677 | -0.042 | -0.020 |

## Part B — our H4 runs (results/explore_h4.json, explore_h4_robust.json)

**P2 in our setting (account-level marginal TS, prior at round 0).** λ₀ = expected hidden-segment accounts in the first batch from the order-statistic formula with the real CPR-LP prior means and variances; observed = first round with a hidden contact (1 = the first batch).

| seed | Δ_eff (prior error) | λ₀ | P(hit in round 0) = 1−e^{−λ₀} | observed first-contact round |
|---|---|---|---|---|
| 0 | 2.53 | 0.146 | 0.14 | 8 |
| 1 | 2.27 | 0.144 | 0.13 | 1 |
| 2 | 2.41 | 0.220 | 0.20 | 2 |
| 3 | 2.51 | 0.148 | 0.14 | 2 |
| 4 | 2.27 | 0.069 | 0.07 | 10 |
| 5 | 2.66 | 0.050 | 0.05 | 8 |
| 6 | 2.79 | 0.069 | 0.07 | 29 |
| 7 | 2.70 | 0.226 | 0.20 | 1 |
| 8 | 2.55 | 0.154 | 0.14 | 2 |
| 9 | 2.30 | 0.183 | 0.17 | 4 |

Predicted number of seeds hit in round 0: 1.3 of 10; observed 2. Predicted median first round if the per-round rate stayed λ₀ (geometric): 5; observed median 3.
Segment-level TS (one draw per segment from the ω = 0.5 segment prior) would instead need ≈ 1/Φ(−z) = 4885 rounds (z = 3.53).

**P4 in our setting (Thompson, Gaussian prior, round 39).** Gaussian formula with each seed's Δ_eff, its number of hidden contacts n, and σ² = 1/(p(1−p)) + graph² + nugget² — vs observed CPR-LP error.

| seed | n hidden contacts | observed error | Gaussian formula | Student-t (exact 1-D) |
|---|---|---|---|---|
| 0 | 53 | -0.23 | -0.81 | -0.18 |
| 1 | 22 | -1.59 | -1.12 | -0.43 |
| 2 | 81 | -0.66 | -0.52 | -0.11 |
| 3 | 56 | -0.25 | -0.77 | -0.17 |
| 4 | 2 | -2.51 | -2.14 | -2.02 |
| 5 | 26 | -0.51 | -1.25 | -0.34 |
| 6 | 6 | -1.75 | -2.19 | -1.53 |
| 7 | 82 | -0.46 | -0.58 | -0.10 |
| 8 | 5 | -2.59 | -2.13 | -1.76 |
| 9 | 9 | -1.85 | -1.75 | -1.33 |

Mean observed -1.24, Gaussian formula -1.33; correlation across seeds 0.88.

**P4, robust runs (thompson-robust, ν = 3, round 39).**

| seed | n hidden contacts | observed error (EM Student-t) | Student-t exact 1-D | Gaussian formula at same n |
|---|---|---|---|---|
| 0 | 108 | +0.33 | -0.08 | -0.47 |
| 1 | 64 | -0.42 | -0.14 | -0.57 |
| 2 | 112 | -0.36 | -0.08 | -0.40 |
| 3 | 75 | +0.27 | -0.12 | -0.63 |
| 4 | 6 | -1.80 | -1.64 | -1.91 |
| 5 | 35 | -0.09 | -0.25 | -1.06 |
| 6 | 19 | +0.12 | -0.44 | -1.49 |
| 7 | 94 | -0.16 | -0.08 | -0.52 |
| 8 | 4 | -2.50 | -1.90 | -2.20 |
| 9 | 11 | -1.60 | -1.17 | -1.67 |

Mean observed -0.62, exact Student-t -0.59, Gaussian at the same n -1.09.

**P3/P5 in our setting.** Stratified exploration samples k* at rate ηB/K per round. Contacts needed for |bias| ≤ ε = 0.5 with Δ = 2.50, σ² = 6.2, ω = 0.5:
- η = 0.05: rate 0.039 contacts/round → Gaussian needs n = 99 (2531 rounds from exploration alone); Student-t needs n = 22 (563 rounds)
- η = 0.1: rate 0.078 contacts/round → Gaussian needs n = 99 (1266 rounds from exploration alone); Student-t needs n = 22 (282 rounds)
- η = 0.2: rate 0.156 contacts/round → Gaussian needs n = 99 (633 rounds from exploration alone); Student-t needs n = 22 (141 rounds)
