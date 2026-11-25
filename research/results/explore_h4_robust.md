H4 with a heavy-tailed segment prior: Student-t(nu = 3) segment deviations (EM scale mixture,
fprior.cprlp.robust_segment_scales) vs the Gaussian prior of scripts/explore_h4.py.

Pre-registered (written before the full run; only a 1-market timing check had been run):
  RB1. Once the hidden (wrongly believed, truly best) segment is contacted, its posterior error shrinks
       much faster with the robust prior: error at rounds 20 and 39 is closer to 0 than for the
       Gaussian version of the same policy, for greedy, thompson and thompson eta = 0.1.
  RB2. Washout (model ranks the hidden segment in its top 3) happens earlier and in more runs.
  RB3. No loss in customers found vs the Gaussian version of the same policy at round 40 (within
       2 SE); a gain is plausible because the hidden segment is the best one.
  Known limitation stated in advance: greedy never contacts the hidden segment, so a prior that only
  changes how evidence is weighed cannot help greedy (expect greedy-robust ~ greedy).

Same markets, founder and RNG seeds as scripts/explore_h4.py (10 seeds, 40 rounds x 25 contacts).
Gaussian policies are rerun here so every comparison is paired within this script.

## Results (10 seeds, 40 rounds × 25 contacts)

| policy | customers / oracle, round 20 | round 40 | hidden contacts | error r10 | error r20 | error r39 | washout (median) | runs washed out |
|---|---|---|---|---|---|---|---|---|
| greedy | +0.63 ± 0.02 | +0.74 ± 0.01 | 0.0 | -2.68 ± 0.02 | -2.65 ± 0.08 | -2.50 ± 0.06 | — | 0/10 |
| greedy-robust | +0.63 ± 0.02 | +0.74 ± 0.01 | 0.0 | -2.67 ± 0.03 | -2.68 ± 0.07 | -2.50 ± 0.06 | — | 0/10 |
| thompson | +0.51 ± 0.02 | +0.69 ± 0.01 | 36.0 | -2.07 ± 0.15 | -1.89 ± 0.21 | -1.24 ± 0.29 | 35 | 5/10 |
| thompson-robust | +0.54 ± 0.02 | +0.71 ± 0.02 | 53.9 | -1.94 ± 0.29 | -1.29 ± 0.39 | -0.62 ± 0.31 | 15 | 7/10 |
| thompson η=0.1 | +0.49 ± 0.02 | +0.66 ± 0.01 | 24.1 | -2.18 ± 0.22 | -2.08 ± 0.20 | -1.57 ± 0.23 | 3 | 1/10 |
| thompson-robust η=0.1 | +0.50 ± 0.02 | +0.69 ± 0.01 | 44.9 | -2.15 ± 0.28 | -1.58 ± 0.33 | -0.83 ± 0.36 | 19 | 7/10 |

### Paired differences, robust − Gaussian (same policy, same seed)

| policy | customers r40 (count) | error r20 | error r39 |
|---|---|---|---|
| greedy | +1.10 ± 1.05 | -0.02 ± 0.02 | +0.00 ± 0.04 |
| thompson | +3.70 ± 2.19 | +0.59 ± 0.33 | +0.62 ± 0.17 |
| thompson η=0.1 | +5.60 ± 2.84 | +0.50 ± 0.23 | +0.73 ± 0.29 |