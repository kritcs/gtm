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
- 2026-11-26 ~02:30: Colab session `pfn-4d` (T4, `pfn_v0g_nf` and `pfn_v0g_s2`) was reclaimed at step 8,500 of
  20,000 while the Mac slept. Checkpoints then held weights only, so both runs were **resumed from the step-8,500
  weights** (`train_pfn.py --resume`, commit `dbf5a1d`) on a new T4 session: fresh AdamW moments, LR schedule
  fast-forwarded to step 8,500, and a different batch stream from that point. The models are therefore not
  bit-identical to an uninterrupted run. This is reported with every result that uses them (G1, G2, G5, G7).
  Checkpoints now also store optimizer, scheduler and RNG state, so later resumes are exact. Colab is not training
  a seed-1 founder model in this round, so `runs/local_v0g` (MPS, `b99ddd1`) is the main v0g model.

## Scoring, part 1 (written after the run, 2026-11-26 ~03:30)

Main model `runs/local_v0g` (seed 1, MPS, commit `b99ddd1`; final val excess 0.061). `eval_v0g.py`, 20 seeds →
`results/eval_v0g.{md,json}`. G1, G2, G5, G6 and G7 need the Colab models and are scored in part 2.

| prediction | result | detail (seed-level mean ± SE, lift@100) |
|---|---|---|
| G3 v0g beats v0.1 on S0 (n ≥ 5) | **FAILS** | +0.024 ± 0.015 |
| G4 true graph beats permuted graph on S0 (n ≥ 5) | **FAILS** | +0.004 ± 0.014 (S2, not a prediction: −0.001 ± 0.016) |

Per the plan, a G4 failure means the neighbour features are ignored, and that must be checked before any further
claim. Diagnostics (exploratory, `scripts/diag/`):

1. **v0g ignores the graph even on its own training prior.** On 256 prior tasks, shuffling the 4 graph features
   across accounts (equivalent to a permuted graph) or zeroing them changes the excess loss by about 0.0001
   (0.0580 vs 0.0581), at every w_graph level, including w_graph > 0.5.
2. **The cause is not feature scaling. The features carry almost no signal.** Among unlabelled accounts with at
   least one labelled neighbour (about 45% of them), the neighbour buy share correlates only **+0.045** with the
   true logit, and still +0.045 when w_graph > 0.5.
3. **Why so weak.** Three factors multiply:
   - The field r is only moderately smooth: corr(r, neighbour mean of r) ≈ 0.45–0.53 for γ ∈ [1, 100].
   - The field makes up only part of the logit variance: √(w_graph·(1 − w_seg)).
   - Each labelled neighbour is a single noisy Bernoulli outcome at a base rate near 10%, and a typical
     account has 1–2 of them.
   Ignoring these features is close to the Bayes-optimal response under this prior.

**Reading.** In this design, one-hop neighbour-label counts are too weak a channel for the graph. G1 is therefore
expected to fail too. That would place the S2 gap with CPR-LP in its exact per-segment posterior or in
multi-hop propagation, not in missing one-hop features. CPR-LP propagates over the whole graph. A graph-aware
PC-RFM needs multi-hop information, for example:
- propagated label features: a personalised-PageRank or (I + γL)⁻¹ smoothing of the visible labels, given as
  input;
- or graph-masked attention.
That would be a new, separately pre-registered round; v0g is not adopted.

**Addendum: CPR-LP's graph term does not help either** (exploratory, `scripts/diag/diag_cprlp_graph.py`; the same 20
seeds, markets, founders and outreach as `eval_v0g.py`). Lift@100, seed-level mean ± SE:

| shift | n | CPR-LP − CPR-LP without graph (graph_sd ≈ 0) | CPR-LP − CPR-LP with node-permuted graph |
|---|---|---|---|
| S0 | 0 | −0.015 ± 0.014 | −0.011 ± 0.019 |
| S0 | 5–20 | −0.007 ± 0.009 | +0.008 ± 0.017 |
| S2 | 0 | −0.006 ± 0.014 | −0.015 ± 0.015 |
| S2 | 5–20 | +0.000 ± 0.014 | +0.007 ± 0.011 |

In the simulator's default regime (n ≤ 20 labels, w_graph = 0.3), the graph is worth nothing even to the exact
Bayesian model that knows the true graph covariance. So:
- (a) Round 4's premise, that the S2 gap between PC-RFM and CPR-LP comes from PC-RFM lacking the graph, was
  wrong before the run. The gap is in the segment-level inference.
- (b) "CPR-LP" in all simulation results so far is in effect a segment-level Bayesian model with a founder prior.
  The paper must not credit any simulated result to the graph.
- (c) With 5–20 labels, a graph can help only if neighbourhoods are very predictive, or if the founder's knowledge
  is itself relational (e.g. "customers of X buy"). In this simulator neither holds. Whether real B2B graphs do
  is an empirical question for the RDL tier, not something the simulator can show.

**Addendum 2: where the graph helps** (exploratory, `results/diag_graph_regime.md`). Sweep over graph share,
smoothness, degree and n (24 cells, 8 seeds): CPR-LP − no-graph is within 2 SE of 0 in 23 / 24 cells. The exception
is the strongest corner (w_graph 0.6, γ = 50, avg degree 10, about 430 contacts): +0.23 ± 0.09. A bug check with
w_graph 0.7, γ = 50 gives +0.43 ± 0.16 at about 180 contacts. **The implementation works; the graph pays off only
when it is strong and smooth.** The default simulator (w_graph 0.3, γ = 5) and v0g's training prior
(γ ∈ [1, 10], w_graph ≤ 0.8 of the non-segment variance) both sit in the weak-graph regime. So did eval S0/S2.

Implication for any later graph round (not pre-registered here):
- Train over a prior that covers strong, smooth graphs (γ up to about 100).
- Give the model multi-hop propagated label features.
- Evaluate on a strong-graph shift, where CPR-LP's graph is known to help (+0.4). A graph-aware PC-RFM should
  match that gain.

Whether real B2B graphs are in the strong regime is the empirical question.

**Addendum 3: would multi-hop features fix it?** (exploratory, prior only, `scripts/diag/diag_prop2.py`). A candidate
v0g2 prior (γ log-uniform on [1, 100], w_graph ≤ 0.9) with personalised-PageRank label features
(`fprior.pfn.data.prop_features`, off by default). Correlation of each feature with the true logit, by the
graph's share of the logit variance:

| graph share | 1-hop | PPR α = 0.5 | PPR α = 0.9 |
|---|---|---|---|
| < 0.2 | +0.03 | +0.03 | +0.04 |
| 0.2–0.5 | +0.05 | +0.07 | +0.09 |
| ≥ 0.5 | +0.07 | +0.07 | +0.07 |

Multi-hop roughly doubles the signal but it stays weak. A round 5 is therefore not launched automatically; whether
to pursue the graph in simulation at all is a direction decision (see status.md).

## Scoring, part 2 (written after the run, 2026-11-26 ~06:45)

Models: main `runs/local_v0g` (seed 1); `runs/pfn_v0g_nf` and `runs/pfn_v0g_s2` (Colab T4, resumed from step 8,500,
see Deviations; final val excess 0.067 and 0.064). Runs: `stress_sim_v0g.{md,json}` (v0g + v0g_nf),
`stress_sim_v0g_s2.{md,json}` (seed 2), `real_bank_v0g.{md,json}`. Scored by `scripts/score_round4.py`: seed-level
paired means over identical markets; thresholds on the point estimate, as in rounds 2–3.

| prediction | result | detail |
|---|---|---|
| G1 S2: v0g − CPR-LP > −0.05 (ρ ∈ {1, 0.5}, n ≥ 5) | **FAILS** | −0.119 ± 0.046 (v0.1: −0.14; seed 2: −0.13) |
| G2 S0: v0g − CPR-LP ≥ −0.02 | **FAILS** | −0.080 ± 0.059 (v0.1: −0.07) |
| G5 founder value kept: v0g − v0g_nf > 2 SE at S0, ρ = 1, n = 0 and 5 | **FAILS** (1 of 2 cells) | n = 0: +2.12 ± 0.33; n = 5: +0.21 ± 0.15 (1.4 SE) |
| G6 Bank R1–R3 hold for v0g | **HOLDS** | R1 all 10 cells > 2 SE; R2 no cell below −0.2; R3 23/24 cells (worst −0.22 at ρ = 0.5, n = 0) |
| G7 seed replicate within 0.05 on the G1 metric | **HOLDS** | seed 2: −0.119 ± 0.032; difference 0.001 |

Exploratory, Bank, identical markets:
- v0g − v0.1 overall: +0.044 ± 0.017.
- The no-founder model at n = 0 reaches lift 1.62 ± 0.09, vs 1.77 ± 0.09 for v0.1-nf, so the unexplained
  no-label Bank lift persists across training runs.

**Round 4 reading.** 2 of 7 predictions hold, and both are robustness checks: real-tier safety (G6) and seed
stability (G7). Every prediction about the graph's benefit fails (G1–G4). With part 1 and its addenda, the cause
is established: in this simulator's default regime the graph carries almost no learnable signal, even for the
exact Bayesian model. The S2 gap (about −0.12, stable across v0.1, v0g and seeds) is therefore not the graph.
G5's n = 5 cell is weaker than in v0.1, which is within noise but noted. Per "What would change the plan", v0g is
not adopted; v0.1 stays the headline PC-RFM. The graph question moves to the real relational tier.
