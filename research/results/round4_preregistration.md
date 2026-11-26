# Round 4 pre-registration: graph-aware PC-RFM (v0g)

Written and committed 2026-11-25 ~17:30, before any round-4 model existed (a 60-step CPU smoke run checked the
code only).

## Why

Rounds 2–3 left a stable gap of about −0.13 lift between PC-RFM v0.x and CPR-LP on pure-interaction segments (S2)
and −0.06/−0.07 on the default simulator (S0). Random segment structure closed ~35% of it, an explicit segment
embedding none, a second training seed none. Remaining explanation: CPR-LP uses the account-graph covariance, and
PC-RFM v0.x never sees the graph (in the default simulator 30% of the non-intercept logit variance is
graph-smooth).

## Model change (`--graph`, commit of this file)

- Prior: each training market gets a degree-corrected SBM graph (avg degree U(4, 16), within-community share
  U(0.5, 0.95), community = industry with prob U(0.3, 1)) and a graph-smooth field r = (I + γL)⁻¹ε (γ ~ U(1, 10)),
  standardised. The non-segment variance is split w_graph·r + (1 − w_graph)·need with w_graph ~ U(0, 0.8).
- Input: four neighbour-label features per account (log degree; log #neighbours contacted & bought;
  log #contacted & not bought; smoothed bought share among contacted neighbours). Only visible labels are used.
- Everything else as v0.1 (`--random-seg`), 20k steps, seed 1.

| name | flags |
|---|---|
| `pfn_v0g` | `--random-seg --graph --seed 1` |
| `pfn_v0g_nf` | `--random-seg --graph --no-founder --seed 1` |
| `pfn_v0g_s2` | `--random-seg --graph --seed 2` (training-seed replicate) |

## Predictions

Rule unless stated: paired mean over market seeds, predicted sign, beyond 2 SE.

- **G1 (closes S2):** `stress_sim.py --ckpt runs/pfn_v0g/model.pt --ckpt-nf runs/pfn_v0g_nf/model.pt`: S2_interaction,
  mean over ρ ∈ {1, 0.5}, n ≥ 5: v0g − CPR-LP > −0.05 (v0.1: −0.14, seed 2: −0.13).
- **G2 (closes S0):** same run, S0_default, same averaging: v0g − CPR-LP ≥ −0.02 (v0.1: −0.07).
- **G3 (beats v0.1 directly):** `eval_v0g.py` (20 seeds): v0g − v0.1 > 2 SE on S0, all ρ, n ≥ 5.
- **G4 (uses structure, not degree):** `eval_v0g.py`: v0g with the true graph − v0g with a node-permuted graph
  > 2 SE on S0 (all ρ, n ≥ 5). (Also reported for S2 as G4b; not a prediction.)
- **G5 (founder value kept):** stress_sim S0, ρ = 1, n ∈ {0, 5}: v0g − v0g_nf > 2 SE in both cells.
- **G6 (real tier, no graph):** `real_bank.py --ckpt runs/pfn_v0g/model.pt --ckpt-nf runs/pfn_v0g_nf/model.pt`:
  R1–R3 hold. Bank has no graph, so v0g sees degree 0 everywhere — outside its training range (degree ≥ ~4);
  a failure here means graph-less deployment needs a graph-dropout prior, and is reported as such.
- **G7 (seed replicate):** G1 metric for `pfn_v0g_s2` within 0.05 of `pfn_v0g`.


## What would change the plan

- G1 and G4 hold → the S2 gap was the missing graph; v0g becomes the headline PC-RFM for simulated graphs.
- G1 fails, G4 holds → the graph is used but the S2 gap has another cause (e.g. CPR-LP's exact per-segment
  posterior); stop chasing it and report the gap.
- G4 fails → the neighbour features are ignored; check feature scaling before any further claim.

## Deviations (written before any round-4 model finished)

- 2026-11-25 ~23:00: on a Colab T4, `torch.linalg.solve` in the `--graph` sampler crashed inside MAGMA's batched LU
  ("illegal memory access"). `train_pfn.py` now selects the cuSOLVER backend on CUDA. This changes only the
  numerical library, not the prior. Colab models are trained at the commit that adds this line; the local MPS
  model (`runs/local_v0g`, seed 1) at `b99ddd1`. If both seed-1 runs finish, the Colab one is used and the local
  one is reported as a replicate.
