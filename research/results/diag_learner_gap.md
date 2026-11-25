# What is PC-RFM's "learner gap" over CPR-LP with bad founders? (exploratory diagnostics)

Context: placebo tests (`placebo_bank.md`, `diag_placebo_selection.md`) showed PC-RFM's advantage with bad
founders is not trust learning; the matched-prior run (`matched_prior.md`) showed it is not the trust prior.
Bank tier, 15 seeds, lift@100 minus CPR labels-only (flat segment kernel). Scripts:
`scripts/diag_nugget_avg.py`, `scripts/diag_main_effects.py`.

## 1. Fixed noise level? No.
Evidence-averaging CPR-LP over nugget_sd ∈ {0.3, 0.7, 1.2, 2.0} changes every cell by ≤ 0.06.

## 2. Sharing through main effects? Mostly yes.
CPR-LP treats the 32 job × age cells as unrelated. Adding random effects for job group and age band
(one-hot extra_features with N(0, 0.5²) coefficients = kernel 0.25·([same group] + [same band])):

| ρ, n | CPR-LP | CPR-LP + main effects | labels-only + main effects | PC-RFM |
|---|---|---|---|---|
| 1, 5 | +0.52 ± 0.18 | +0.45 ± 0.17 | +0.16 ± 0.18 | +0.50 ± 0.16 |
| 1, 20 | +0.86 ± 0.17 | +0.76 ± 0.18 | +0.19 ± 0.08 | +0.86 ± 0.16 |
| 0, 10 | +0.16 ± 0.11 | +0.20 ± 0.11 | +0.10 ± 0.09 | +0.42 ± 0.16 |
| 0, 50 | +0.01 ± 0.25 | +0.36 ± 0.17 | +0.51 ± 0.18 | +0.43 ± 0.17 |
| −0.5, 10 | −0.31 ± 0.16 | −0.05 ± 0.14 | +0.01 ± 0.08 | −0.14 ± 0.17 |
| −0.5, 20 | −0.23 ± 0.22 | +0.15 ± 0.14 | +0.21 ± 0.14 | +0.15 ± 0.15 |
| −0.5, 50 | +0.19 ± 0.27 | +0.54 ± 0.23 | +0.67 ± 0.16 | +0.93 ± 0.19 |

(Full grid in the script output; all 12 cells were run.)

## Reading
- Most of the bad-founder gap is CPR-LP's flat segment kernel: it cannot pool evidence across segments that
  share a job group or age band, which PC-RFM learned from the simulator (and Bank's segment variance is ~82%
  main effects, `sim_calibration.md`). Consistent with the stress test: PC-RFM loses exactly when segments are
  pure interactions (`stress_sim.md`, S2).
- A gap remains at ρ = −0.5, n = 50 (+0.54 vs +0.93), and with good founders main effects cost a little (within noise).
- **Action:** a hierarchical kernel (main effects + cell) becomes the standard CPR-LP baseline; re-run the
  pre-registered tiers with it, and evidence-select the main-effect scale rather than fixing 0.5.
