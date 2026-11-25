# Round 2 pre-registration: random-seg training, wider founder range, v2 ablation

Written and committed 2026-11-25 ~02:45, before any of these models existed (round 2 not yet launched; the
founder-token model from round r3 was at step ~7,500 and is scored separately by `scripts/eval_pfn_tok.py`).

## Models (all 20k steps, seed 1, L4, commit `2dea623` or later with the same sampler)

| name | flags | question |
|---|---|---|
| `pfn_v01` | `--random-seg` | does randomising segment structure in training fix the S2 (pure-interaction) weakness? |
| `pfn_v01_nf` | `--random-seg --no-founder` | matching no-founder ablation |
| `pfn_v02_rho` | `--random-seg --rho-min -1` | does training on founders down to ρ = −1 remove the inverted-founder cost (NC3)? |
| `pfn_v2_nf` | `--encoder --no-founder` | completes M2 P1 for v2 (founder value with a learned encoder) |
| `pfn_tok` (round r3) | `--random-seg --tokens` | founder tokens; on these tests it is fed neutral statements (stated sd 0.7) |

## Predictions

Rule unless stated: paired mean difference over seeds has the predicted sign and exceeds 2 SE.

**Q1. Random-seg fixes S2.** `stress_sim.py --ckpt runs/pfn_v01/model.pt --ckpt-nf runs/pfn_v01_nf/model.pt`.
- Q1a: in S2_interaction, mean over ρ ∈ {1, 0.5}, n ≥ 5, PC-RFM v0.1 − CPR-LP > −0.10 (v0 was −0.22 ± 0.06).
- Q1b: the S2 "PC-RFM − labels worst cell" improves from −0.76 to ≥ −0.5.
- Q1c (no cost): in S0_default, same averaging, |v0.1 − CPR-LP| ≤ 0.10 (v0: −0.07 ± 0.05).

**Q2. Real tier unchanged or better.** `real_bank.py --ckpt runs/pfn_v01/model.pt --ckpt-nf runs/pfn_v01_nf/model.pt`.
- Q2a: R1 and R2 of `real_bank.py` still hold for v0.1.
- Q2b: R3 (within 0.2 of CPR-LP in ≥ 2/3 of cells) still holds for v0.1. Same two for `pfn_tok`.

**Q3. Wider founder range removes the inverted-founder cost.**
`bank_noise_control.py 30 runs/pfn_v02_rho/model.pt results/bank_noise_control_v02_rho`.
- Q3a: NC3 holds for v0.2-rho (|true − noise| < 0.2 at ρ ∈ {0, −0.5}, n ≥ 5); it failed for v0.
- Q3b (cost): NC1 still holds (ρ = 1, true − noise > 2 SE at every n ≤ 20).
- Q3c (sim): `eval_pfn.py --ckpt runs/pfn_v02_rho/model.pt` vs v0.1 at ρ ≥ 0.5: v0.2-rho − v0.1 ≥ −0.10 in every n
  (computed from paired runs on the same markets).

**Q4. v2 founder value.** `eval_pfn_rel.py --ckpt runs/pfn_v2/model.pt --ckpt-nf runs/pfn_v2_nf/model.pt ...`, then
`score_m2.py` P1 applied to the v2 file: holds by the P1 rule (v2 − v2-nf > 2 SE in ≥ half of the cells at
ρ ∈ {1, 0.5}, and the gap at e = 1 larger than at e = 0).

## What would change the plan

- Q1 fails → segment structure is not the cause of the S2 weakness; look at the attention mask / feature set.
- Q3a fails → the inverted-founder cost is not a training-range problem; the Student-t CPR-LP (BF1) remains
  the recommendation for adversarial founders.
- Any regression in Q1c / Q2 → keep v0 as the headline PC-RFM and report v0.1 as an ablation.
