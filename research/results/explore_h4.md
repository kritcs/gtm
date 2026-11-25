**Pre-registered** (written before the full run):
  E1. greedy and Thompson with eta = 0 contact the hidden best segment ~never, and its posterior
      error stays near the founder's -3 logits through round 40.
  E2. with stratified exploration eta in {0.05, 0.1, 0.2} the hidden segment gets contacted, its error
      shrinks toward 0, and the washout time (first round the model ranks it in its top 3 segments)
      is finite and decreases as eta grows.
  E3. exploration costs customers early and the cost is recovered (or not) by round 40: report
      cumulative customers vs oracle per round; no prediction on the sign at round 40.

10 seeds, 40 rounds × 25 contacts. Mean ± SE over seeds.

## Customers found, as a share of the oracle's

| policy | round 5 | round 10 | round 20 | round 30 | round 40 |
|---|---|---|---|---|---|
| oracle | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |
| founder | 0.52 ± 0.03 | 0.57 ± 0.02 | 0.64 ± 0.01 | 0.69 ± 0.01 | 0.75 ± 0.02 |
| greedy | 0.51 ± 0.03 | 0.58 ± 0.02 | 0.63 ± 0.02 | 0.69 ± 0.02 | 0.74 ± 0.01 |
| greedy η=0.1 | 0.49 ± 0.03 | 0.52 ± 0.03 | 0.61 ± 0.02 | 0.67 ± 0.02 | 0.74 ± 0.02 |
| thompson | 0.38 ± 0.02 | 0.41 ± 0.02 | 0.51 ± 0.02 | 0.61 ± 0.02 | 0.69 ± 0.01 |
| thompson η=0.05 | 0.37 ± 0.03 | 0.45 ± 0.02 | 0.54 ± 0.03 | 0.61 ± 0.02 | 0.68 ± 0.02 |
| thompson η=0.1 | 0.33 ± 0.02 | 0.41 ± 0.02 | 0.49 ± 0.02 | 0.58 ± 0.01 | 0.66 ± 0.01 |
| thompson η=0.2 | 0.34 ± 0.02 | 0.40 ± 0.03 | 0.49 ± 0.02 | 0.58 ± 0.02 | 0.66 ± 0.01 |

## The wrongly believed (truly best) segment

Error = model's mean logit for that segment minus the truth (founder starts at −3).
Washout = first round the model ranks it in its top 3 segments (— = never within the horizon).

| policy | contacts by round 40 | error round 10 | error round 20 | error round 39 | washout (median) | runs washed out | learned b, round 39 |
|---|---|---|---|---|---|---|---|
| greedy | 0.0 ± 0.0 | -2.68 ± 0.02 | -2.65 ± 0.08 | -2.50 ± 0.06 | — | 0/10 | 0.84 |
| greedy η=0.1 | 24.8 ± 11.0 | -2.29 ± 0.27 | -2.18 ± 0.26 | -1.74 ± 0.32 | 19 | 2/10 | 0.77 |
| thompson | 36.0 ± 10.2 | -2.07 ± 0.15 | -1.89 ± 0.21 | -1.24 ± 0.29 | 35 | 5/10 | 0.68 |
| thompson η=0.05 | 26.5 ± 8.2 | -2.33 ± 0.17 | -2.20 ± 0.19 | -1.55 ± 0.28 | 28 | 2/10 | 0.73 |
| thompson η=0.1 | 24.1 ± 6.1 | -2.18 ± 0.22 | -2.08 ± 0.20 | -1.57 ± 0.23 | 3 | 1/10 | 0.72 |
| thompson η=0.2 | 32.3 ± 7.0 | -2.12 ± 0.20 | -2.01 ± 0.23 | -1.23 ± 0.19 | 20 | 2/10 | 0.73 |