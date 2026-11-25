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
