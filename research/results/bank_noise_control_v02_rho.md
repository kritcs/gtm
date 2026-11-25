Founder benefit for PC-RFM measured against its OWN no-information mode (UCI Bank Marketing).

After the placebo tests (results/diag_placebo_selection.md), the fair founder-benefit comparison for
PC-RFM is the same network shown (a) the founder's beliefs vs (b) the same values scrambled across
accounts (no information). Same markets, outreach and relabelling as scripts/real_bank.py.

Pre-registered (committed before running):
  NC1. ρ = 1: true − noise > 0 at every n in {0, 1, 5, 10, 20} (> 2 SE).
  NC2. ρ = 0.5: true − noise > 0 at n ≤ 5.
  NC3. ρ ∈ {0, −0.5}, n ≥ 5: |true − noise| < 0.2 (a useless founder adds nothing beyond the model's
       own label learning; an inverted one may cost a little at small n).

## Results (30 seeds): runs/pfn_v02_rho/model.pt, noise = account; true beliefs − noise beliefs (lift@100)

| ρ | n = 0 | n = 1 | n = 5 | n = 10 | n = 20 | n = 50 |
|---|---|---|---|---|---|---|
| 1.0 | +2.33 ± 0.09 | +1.59 ± 0.19 | +1.31 ± 0.14 | +0.98 ± 0.11 | +0.62 ± 0.09 | +0.47 ± 0.08 |
| 0.5 | +0.77 ± 0.16 | +0.69 ± 0.14 | +0.18 ± 0.13 | +0.39 ± 0.11 | -0.02 ± 0.11 | -0.04 ± 0.10 |
| 0.0 | +0.05 ± 0.13 | -0.01 ± 0.11 | +0.06 ± 0.17 | -0.01 ± 0.12 | -0.20 ± 0.11 | -0.20 ± 0.10 |
| -0.5 | -0.14 ± 0.12 | +0.43 ± 0.15 | +0.63 ± 0.19 | +0.87 ± 0.16 | +0.22 ± 0.14 | +0.13 ± 0.20 |