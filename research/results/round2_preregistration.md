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

## Scoring (written after the run, 2026-11-25 ~08:45)

All four models trained at commit `8919ff2`, 20k steps, seed 1. Result files: `stress_sim_v01.md`,
`real_bank_v01.md`, `bank_noise_control_v02_rho{,_segnoise}.md`, `m1_pfn_v01.md`, `m1_pfn_v02_rho.md`,
`m2_pfn_v2_full.md`. Q2 for `pfn_tok` is in `real_bank_tok.md` (holds).

| prediction | result | detail |
|---|---|---|
| Q1a S2: v0.1 − CPR-LP > −0.10 | **FAILS** | −0.14 ± 0.05 (v0: −0.22 ± 0.06) — improved, not enough |
| Q1b S2 worst cell ≥ −0.5 | **FAILS** | −0.54 (v0: −0.76) — improved, narrowly short |
| Q1c S0 \|v0.1 − CPR-LP\| ≤ 0.10 | **HOLDS** | −0.07 ± 0.04 |
| Q2a/b bank R1, R2, R3 for v0.1 | **HOLDS** | R1 all n > 2 SE (+2.37 … +0.73); R2 worst +0.07; R3 23/24 cells not worse than CPR-LP by > 0.2 (worst −0.22) |
| Q3a NC3 for v0.2-rho | **FAILS (opposite direction)** | ρ = −0.5: true − noise = +0.63 / +0.87 / +0.22 at n = 5 / 10 / 20 (account noise); similar with segment noise |
| Q3b NC1 for v0.2-rho | **HOLDS** | +2.33 … +0.62 at n = 0 … 20, all > 2 SE |
| Q3c v0.2-rho − v0.1 ≥ −0.10 at ρ ≥ 0.5, every n | **FAILS** | −0.18 ± 0.17, −0.19 ± 0.17 (ρ = 0.75, n = 1 / 5), −0.22 ± 0.15 (ρ = 0.5, n = 50); all within ~1.3 SE of zero |
| Q4 P1 rule applied to v2 | **HOLDS** | 16/30 cells > 2 SE; gap e = 1 +0.50 vs e = 0 +0.32 (driven by n = 0: +2.73 at e = 1, ρ = 1) |

Readings:
- **Random-seg (Q1) moves S2 in the right direction but does not close it.** The S2 gap to CPR-LP shrinks by
  ~35%; per the plan, segment structure is only part of the cause. Next suspect: CPR-LP's per-segment deviations
  capture interactions directly, PC-RFM sees industry and size as separate one-hots.
- **Q3a fails in an informative way.** The cost of an inverted founder did not just disappear — it became a
  *gain*: trained on ρ down to −1, the model learns to read an inverted founder's beliefs upside down once it
  has a few labels (+0.87 at n = 10). Bayes-correct under the prior, but a strong behavioural assumption for
  real founders (systematically inverted beliefs are rare; ρ ≈ 0 is the realistic bad case). The price shows in
  Q3c: small, noisy losses with good founders. **Keep `rho_min = −0.5` as default**; report v0.2-rho as the
  "adversarial founder" ablation.
- **Q4: with a learned encoder, founder value is larger in new categories,** which is what M2 P1 predicted for
  v1. The e = 1, n = 0 cell (+2.73) is where v1 failed (`m2_preregistration.md`).
