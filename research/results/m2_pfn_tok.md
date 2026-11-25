# PC-RFM-tok: does it use stated confidence? (10 seeds, 3000 accounts)

Expected lift@100 on uncontacted accounts. Paired differences, mean ± SE over seeds.

### ρ = 0.75

| n | neutral | calibrated | inverted | noidea_worst | noidea_rand | calibrated − neutral | calibrated − inverted | noidea_worst − noidea_rand | neutral − v0 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 2.72 | 2.65 | 2.48 | 2.96 | 2.94 | -0.07 ± 0.18 | +0.17 ± 0.18 | +0.02 ± 0.28 | -0.07 ± 0.25 |
| 1 | 2.63 | 2.46 | 2.28 | 2.49 | 2.36 | -0.17 ± 0.11 | +0.18 ± 0.16 | +0.13 ± 0.23 | +0.07 ± 0.28 |
| 5 | 3.09 | 3.15 | 3.15 | 3.02 | 3.12 | +0.06 ± 0.27 | -0.00 ± 0.29 | -0.10 ± 0.18 | +0.10 ± 0.20 |
| 10 | 3.18 | 3.17 | 2.87 | 2.99 | 3.15 | -0.01 ± 0.05 | +0.30 ± 0.17 | -0.17 ± 0.15 | +0.28 ± 0.16 |
| 20 | 2.97 | 3.07 | 2.74 | 3.18 | 3.13 | +0.10 ± 0.06 | +0.33 ± 0.13 | +0.04 ± 0.15 | +0.13 ± 0.12 |

### ρ = 0.5

| n | neutral | calibrated | inverted | noidea_worst | noidea_rand | calibrated − neutral | calibrated − inverted | noidea_worst − noidea_rand | neutral − v0 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 2.08 | 2.52 | 1.66 | 2.14 | 1.91 | +0.45 ± 0.24 | +0.86 ± 0.24 | +0.23 ± 0.21 | +0.13 ± 0.16 |
| 1 | 2.11 | 2.60 | 1.96 | 2.35 | 2.06 | +0.49 ± 0.28 | +0.64 ± 0.27 | +0.29 ± 0.21 | +0.04 ± 0.12 |
| 5 | 2.77 | 3.14 | 2.65 | 2.82 | 2.69 | +0.37 ± 0.18 | +0.49 ± 0.19 | +0.12 ± 0.09 | +0.20 ± 0.16 |
| 10 | 2.58 | 2.64 | 2.27 | 2.57 | 2.63 | +0.06 ± 0.07 | +0.37 ± 0.13 | -0.06 ± 0.17 | +0.19 ± 0.13 |
| 20 | 2.85 | 2.95 | 2.88 | 2.88 | 2.59 | +0.09 ± 0.20 | +0.07 ± 0.21 | +0.28 ± 0.22 | +0.08 ± 0.11 |

### ρ = 0.25

| n | neutral | calibrated | inverted | noidea_worst | noidea_rand | calibrated − neutral | calibrated − inverted | noidea_worst − noidea_rand | neutral − v0 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.21 | 2.57 | 0.65 | 2.01 | 1.34 | +1.36 ± 0.25 | +1.92 ± 0.30 | +0.67 ± 0.32 | +0.08 ± 0.05 |
| 1 | 1.65 | 2.78 | 1.38 | 1.75 | 2.12 | +1.14 ± 0.33 | +1.40 ± 0.27 | -0.36 ± 0.30 | -0.14 ± 0.19 |
| 5 | 2.46 | 2.43 | 2.24 | 2.88 | 2.43 | -0.02 ± 0.35 | +0.19 ± 0.38 | +0.45 ± 0.29 | -0.11 ± 0.24 |
| 10 | 2.71 | 2.90 | 2.38 | 2.74 | 2.75 | +0.19 ± 0.09 | +0.51 ± 0.25 | -0.01 ± 0.06 | +0.01 ± 0.02 |
| 20 | 2.61 | 3.14 | 2.27 | 2.42 | 2.35 | +0.54 ± 0.31 | +0.87 ± 0.35 | +0.07 ± 0.13 | +0.11 ± 0.10 |

## Mechanical scoring

- FAILS — T1 stated confidence used: 3/9 cells > 2 SE
- HOLDS — T2 used in the right direction: 5/9 cells > 2 SE
- FAILS — T3 'no idea' treated as missing: 1/9 cells > 2 SE
- HOLDS — T4 advantage shrinks with labels: gap n∈{0,1} +0.53 vs n=20 +0.24
- HOLDS — T5 no regression vs v0: worst cell -0.14 (margin −0.3)

## Scoring (written after the run)

Model `runs/pfn_tok` (round r3, commit `f505a52`, `--random-seg --tokens --seed 1`, 20k steps). 10 seeds.

- **T1 fails** by the rule (3/9 cells > 2 SE). The effect is strongly ρ-dependent: calibrated − neutral is
  +1.36 ± 0.25 / +1.14 ± 0.33 at ρ = 0.25, n = 0 / 1 and +0.45 / +0.49 (≈ 2 SE) at ρ = 0.5, but ≈ 0 at ρ = 0.75,
  where per-segment errors are small and stated sds carry little information. The prediction lumped
  all three ρ together; the honest reading is "stated confidence is used, and matters when the founder is
  mediocre and labels are few" — a post-hoc description, not a held prediction.
- **T2 holds** (5/9): reversing which segments the founder is confident about costs up to 1.9 lift
  (ρ = 0.25, n = 0), so the model uses the stated sds in the right direction.
- **T3 fails** (1/9): "no idea" on the worst segments beats "no idea" on random segments only at ρ = 0.25,
  n = 0 (+0.67 ± 0.32) and is noise elsewhere. Possible reasons (untested): 25% blanked segments is mild; in
  training "no idea" was assigned at random, never preferentially to wrong segments, so the model has no reason
  to treat it as more than missing information.
- **T4 holds:** the T1 gap falls from +0.53 (n ≤ 1) to +0.24 (n = 20), though n = 20 at ρ = 0.25 is still +0.54.
- **T5 holds:** with neutral statements the token model is within −0.14 of v0 everywhere (it never needs
  stated confidence to work), despite a different training prior (`--random-seg`).

Implication for the cohort study: asking founders "how sure are you, per segment" is worth it for the
middle of the reliability range; "no idea" answers did not demonstrably help here.
