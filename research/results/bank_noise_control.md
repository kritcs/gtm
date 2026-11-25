Founder benefit for PC-RFM measured against its OWN no-information mode (UCI Bank Marketing).

After the placebo tests (results/diag_placebo_selection.md), the fair founder-benefit comparison for
PC-RFM is the same network shown (a) the founder's beliefs vs (b) the same values scrambled across
accounts (no information). Same markets, outreach and relabelling as scripts/real_bank.py.

Pre-registered (committed before running):
  NC1. ρ = 1: true − noise > 0 at every n in {0, 1, 5, 10, 20} (> 2 SE).
  NC2. ρ = 0.5: true − noise > 0 at n ≤ 5.
  NC3. ρ ∈ {0, −0.5}, n ≥ 5: |true − noise| < 0.2 (a useless founder adds nothing beyond the model's
       own label learning; an inverted one may cost a little at small n).

## Results (30 seeds): PC-RFM v0, true beliefs − noise beliefs (lift@100)

| ρ | n = 0 | n = 1 | n = 5 | n = 10 | n = 20 | n = 50 |
|---|---|---|---|---|---|---|
| 1.0 | +2.38 ± 0.08 | +2.14 ± 0.10 | +1.53 ± 0.12 | +0.83 ± 0.11 | +0.61 ± 0.09 | +0.54 ± 0.08 |
| 0.5 | +0.84 ± 0.17 | +1.02 ± 0.14 | +0.19 ± 0.10 | +0.20 ± 0.12 | +0.00 ± 0.09 | +0.10 ± 0.07 |
| 0.0 | -0.02 ± 0.12 | -0.01 ± 0.11 | -0.08 ± 0.14 | -0.02 ± 0.11 | -0.09 ± 0.11 | -0.24 ± 0.10 |
| -0.5 | -0.23 ± 0.08 | -0.02 ± 0.10 | -0.13 ± 0.10 | -0.29 ± 0.12 | -0.23 ± 0.12 | -0.16 ± 0.08 |
## Scoring (written after the run)

- **NC1 holds.** ρ = 1: +2.38, +2.14, +1.53, +0.83, +0.61 at n = 0–20 (all > 2 SE), shrinking with n — the
  "gap shrinks as labels accrue" half of H1, now visible against the right control.
- **NC2 mostly holds.** ρ = 0.5: +0.84, +1.02 at n = 0, 1 (> 2 SE); +0.19 ± 0.10 at n = 5 (≈ 1.9 SE, borderline).
- **NC3 fails for inverted founders.** ρ = −0.5, n ≥ 5: −0.13, −0.29, −0.23, −0.16 (n = 10, 20 exceed the 0.2
  margin). ρ = 0: within 0.1 except −0.24 ± 0.10 at n = 50.
- **Reading:** PC-RFM's trust learning is real but incomplete for systematically wrong founders. Candidate
  causes: inverted founders are a small share of training (ρ ~ U[−0.5, 1]); context sizes in training rarely
  give enough contradicting evidence. Next: widen/reweight the training ρ distribution and compare with the
  Student-t CPR-LP on the same cells.
