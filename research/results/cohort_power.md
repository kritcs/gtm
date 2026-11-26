# Cohort power analysis (simulation)

600 simulated companies (fprior.sim; rho ~ U(−0.25, 1); 60–300 founder-chosen contacts). Metric: AUC of realized outcomes among contacts made after the first k. Script: `scripts/cohort_power.py`.

## Company-level effect sizes (mean ± sd over companies)

| k | CPR-LP − labels only | CPR-LP − prior only | same, oracle AUC on all accounts (CPR − labels) |
|---|---|---|---|
| 5 | +0.038 ± 0.176 | +0.005 ± 0.034 | +0.095 ± 0.118 |
| 10 | +0.020 ± 0.161 | +0.011 ± 0.052 | +0.082 ± 0.117 |
| 20 | +0.009 ± 0.146 | +0.021 ± 0.072 | +0.072 ± 0.115 |

## Power: share of 1,000 resampled cohorts whose 95% bootstrap CI excludes 0 (predicted sign > 0)

| companies | CPR−labels k=5 | CPR−labels k=10 | CPR−labels k=20 | CPR−prior k=5 | CPR−prior k=10 | CPR−prior k=20 |
|---|---|---|---|---|---|---|
| 10 | 0.23 | 0.15 | 0.12 | 0.12 | 0.16 | 0.26 |
| 15 | 0.22 | 0.14 | 0.11 | 0.16 | 0.17 | 0.31 |
| 20 | 0.22 | 0.15 | 0.09 | 0.16 | 0.22 | 0.41 |
| 30 | 0.26 | 0.15 | 0.11 | 0.16 | 0.26 | 0.52 |
| 50 | 0.36 | 0.17 | 0.09 | 0.22 | 0.36 | 0.69 |

## Design B: evaluation on a random slice of 40 accounts per company

| k | CPR-LP − labels only | CPR-LP − prior only |
|---|---|---|
| 5 | +0.099 ± 0.250 | -0.000 ± 0.016 |
| 10 | +0.099 ± 0.271 | +0.001 ± 0.030 |
| 20 | +0.094 ± 0.283 | +0.004 ± 0.057 |

| companies | CPR−labels k=5 | CPR−labels k=10 | CPR−labels k=20 | CPR−prior k=5 | CPR−prior k=10 | CPR−prior k=20 |
|---|---|---|---|---|---|---|
| 10 | 0.32 | 0.31 | 0.25 | 0.02 | 0.03 | 0.09 |
| 15 | 0.43 | 0.34 | 0.30 | 0.02 | 0.05 | 0.10 |
| 20 | 0.47 | 0.46 | 0.36 | 0.04 | 0.06 | 0.09 |
| 30 | 0.61 | 0.56 | 0.51 | 0.04 | 0.04 | 0.09 |
| 50 | 0.83 | 0.76 | 0.68 | 0.05 | 0.06 | 0.11 |
