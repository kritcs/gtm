Founder benefit for PC-RFM measured against its OWN no-information mode (UCI Bank Marketing).

After the placebo tests (results/diag_placebo_selection.md), the fair founder-benefit comparison for
PC-RFM is the same network shown (a) the founder's beliefs vs (b) the same values scrambled across
accounts (no information). Same markets, outreach and relabelling as scripts/real_bank.py.

Pre-registered (committed before running):
  NC1. ρ = 1: true − noise > 0 at every n in {0, 1, 5, 10, 20} (> 2 SE).
  NC2. ρ = 0.5: true − noise > 0 at n ≤ 5.
  NC3. ρ ∈ {0, −0.5}, n ≥ 5: |true − noise| < 0.2 (a useless founder adds nothing beyond the model's
       own label learning; an inverted one may cost a little at small n).

## Results (30 seeds): runs/pfn_v02_rho/model.pt, noise = segment; true beliefs − noise beliefs (lift@100)

| ρ | n = 0 | n = 1 | n = 5 | n = 10 | n = 20 | n = 50 |
|---|---|---|---|---|---|---|
| 1.0 | +2.26 ± 0.12 | +1.15 ± 0.18 | +0.94 ± 0.15 | +0.97 ± 0.14 | +0.87 ± 0.12 | +0.49 ± 0.09 |
| 0.5 | +0.60 ± 0.21 | +0.45 ± 0.12 | -0.04 ± 0.18 | +0.25 ± 0.22 | +0.14 ± 0.08 | +0.10 ± 0.12 |
| 0.0 | -0.11 ± 0.17 | -0.28 ± 0.15 | +0.04 ± 0.15 | -0.07 ± 0.18 | -0.13 ± 0.16 | +0.08 ± 0.12 |
| -0.5 | -0.24 ± 0.16 | +0.37 ± 0.13 | +0.57 ± 0.16 | +0.79 ± 0.19 | +0.25 ± 0.15 | +0.39 ± 0.20 |