# Round 3 pre-registration: explicit segment embedding; training-seed replicates

Written and committed 2026-11-25 ~09:00, before any round-3 model existed.

## Why

- Round 2 Q1: `--random-seg` closed ~35% of PC-RFM's gap to CPR-LP on pure-interaction segments (S2), not all.
  Two explanations: (i) **representation** — PC-RFM sees industry and size as separate one-hots and must build
  the 32 conjunctions itself; CPR-LP has a per-segment deviation; (ii) **training noise / budget** — the gap is
  within what another training seed or more steps would change. Round 3 tests (i) and measures (ii).
- Every PC-RFM comparison so far uses one training seed. Seed-to-seed spread is unknown.

## Models (20k steps, L4, commit of this file)

| name | flags | role |
|---|---|---|
| `pfn_v03` | `--random-seg --seg-cross --seed 1` | explicit industry×size embedding added to each token |
| `pfn_v01_s2` | `--random-seg --seed 2` | v0.1 training-seed replicate |
| `pfn_v1_s2` | `--relational --seed 2` | v1 training-seed replicate (M2) |

## Predictions

Rule unless stated: paired mean over market seeds, sign as predicted, beyond 2 SE.

- **S1 (representation):** `stress_sim.py --ckpt runs/pfn_v03/model.pt --ckpt-nf runs/pfn_v01_nf/model.pt`:
  in S2_interaction, mean over ρ ∈ {1, 0.5}, n ≥ 5, v0.3 − CPR-LP > −0.10 (v0: −0.22, v0.1: −0.14).
- **S2 (no cost):** same run, S0_default: |v0.3 − CPR-LP| ≤ 0.10.
- **S3 (seed noise is small):** v0.1 seed 2 on the same stress run: |S2 metric(seed 2) − S2 metric(seed 1)| < 0.05.
  If S3 fails, a difference between v0.3 and v0.1 smaller than the seed-to-seed spread is not credited to S1.
- **S4 (M2 replicates):** `eval_pfn_rel.py --ckpt runs/pfn_v1_s2/model.pt --ckpt-nf runs/pfn_v1_nf/model.pt
  --ckpt-v0 runs/pfn_v0/model.pt`, scored by `score_m2.py`: P2, P4 and P6 hold again; P1 and P3 fail again
  (the same verdicts as seed 1). Any verdict that flips is reported as seed-sensitive.
- **S5 (real tier):** `real_bank.py --ckpt runs/pfn_v03/model.pt --ckpt-nf runs/pfn_v01_nf/model.pt`: R1–R3 hold.

Note: `--ckpt-nf` for v0.3 is the v0.1 no-founder model (no seg-cross ablation trained); "PC-RFM − no-founder"
columns therefore mix two architectures and are not used for any prediction above.

## Scoring (written after the run, 2026-11-25 ~10:10)

Trained on an A100 at commit `5c7ab98` (≈ 55 min for three 20k-step models; 7.45 units). Files:
`stress_sim_v03.md`, `stress_sim_v01_s2.md`, `m2_pfn_v1_s2.{md,json}`, `real_bank_v03.{md,json}`.

| prediction | result | detail |
|---|---|---|
| S1 representation: S2 v0.3 − CPR-LP > −0.10 | **FAILS** | −0.13 ± 0.05 (v0.1 seed 1: −0.14; seed 2: −0.13) |
| S2 no cost on S0 | **HOLDS** | −0.06 ± 0.05 |
| S3 seed noise < 0.05 on the S2 metric | **HOLDS** | seed 1 −0.14, seed 2 −0.13 |
| S4 M2 verdicts replicate with training seed 2 | **HOLDS** | P1 fails (e = 1 +0.32 vs e = 0 +0.38), P2 holds (20/20; 17/20), P3 fails (worst −0.34), P4 holds (5/6), P6 holds (68/72) — same verdicts as seed 1 |
| S5 bank R1–R3 for v0.3 | **HOLDS** | R1 all n > 2 SE; R2 worst +0.04; R3 22/24 (worst −0.22) |

Readings:
- **The explicit segment embedding does nothing** for the S2 gap (−0.13 vs −0.14). Representation is not the
  cause, and S3 says it is not training noise either: the ~−0.13 S2 gap is a stable property of PC-RFM v0.x.
  Remaining explanation (untested): CPR-LP uses the account graph covariance and PC-RFM v0.x does not see the
  graph at all; S0 has the same sign (−0.06/−0.07). A graph-aware v0 is the test; low priority, since the gap is small.
- **The "worst cell" statistic is seed-sensitive** (S2 worst cell −0.54 seed 1, −0.78 seed 2, −0.48 v0.3) —
  stop using single worst cells as pass/fail criteria; use cell means with SEs.
- **M2 conclusions are robust to training seed**, including both failures (P1, P3), which strengthens them.
- Training on an A100 is ~2.7× faster than the L4 per model-step at ~3.5× the price.
