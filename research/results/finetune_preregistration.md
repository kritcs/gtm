# Pre-registration: second real tier (Hillstrom) and fine-tuning PC-RFM on real outcomes

Written and committed 2026-11-25 ~18:00, before any Hillstrom evaluation was run and before any fine-tuned model
existed (a 20-step fine-tuning smoke run checked the code only).

## Data

- **Bank** (tier 1, as before): `fprior.real_bank.load_bank`.
- **Hillstrom** (tier 2, new): MineThatData e-mail test, the two-thirds of customers who were *randomly* e-mailed;
  outcome = site visit (16.7%). Segments 8 × 4: recency quartile × newbie flag, by past-year spend band
  (`load_hillstrom`; 28 of 32 cells populated). Measured before writing this: account-weighted segment logit sd
  0.36 (Bank 0.47), inside the PC-RFM training range. Licence: none stated on the source page; raw file not
  redistributed.
- Both tiers use the same A / B split (`split_pool`, seed 12345): founder beliefs from A, evaluation markets from B.

## Fine-tuning (`scripts/finetune_pfn.py`)

Starts from `pfn_v01` / `pfn_v01_nf`; 2,000 steps, batch 8 real + 8 simulated tasks, lr 5e-5, seed 0, CPU.
Real tasks use only split A (A1 founder beliefs, A2 markets and outcomes); **evaluation split B is never seen.**

| model | init | fine-tuned on |
|---|---|---|
| `pfn_v01_ftbank`, `pfn_v01_nf_ftbank` | v0.1, v0.1-nf | Bank A |
| `pfn_v01_fthill`, `pfn_v01_nf_fthill` | v0.1, v0.1-nf | Hillstrom A |

## Predictions

Rule unless stated: seed-level paired mean (30 seeds, identical markets) with the predicted sign beyond 2 SE,
computed with `scripts/compare_real.py`. "Overall" = mean over all (ρ, n) cells of `real_bank.py`.

**Tier 2, base model** (`real_bank.py --dataset hillstrom --ckpt runs/pfn_v01/model.pt --ckpt-nf runs/pfn_v01_nf/model.pt`):
- **H1–H3:** the three Bank predictions R1–R3 (see `real_bank.py` docstring) hold on Hillstrom for v0.1.
  Stated risk: the segment signal is weak (sd 0.36), so R1 at large n may fail because labels alone catch up.

**Fine-tuning:**
- **F1 (in-domain gain):** Bank B, v01_ftbank − v01 overall > 0 beyond 2 SE.
- **F2 (in-domain gain):** Hillstrom B, v01_fthill − v01 overall > 0 beyond 2 SE.
- **F3 (cross-domain, no harm):** v01_ftbank on Hillstrom and v01_fthill on Bank: − v01 overall > −2 SE
  (fine-tuning on one real dataset does not hurt on another).
- **F4 (cross-domain gain):** the same two differences > 0 beyond 2 SE. *Expected to fail* (two datasets of very
  different businesses); stated so a positive result is not over-read.
- **F5 (no forgetting):** `stress_sim.py --shifts S0_default` (10 seeds) with each fine-tuned v0.1:
  pcrfm − v01's pcrfm (from `stress_sim_v01.json`, identical markets) ≥ −0.10 overall.
- **F6 (founder use survives):** Bank R1 (ρ = 1, every n: pcrfm − labels-only > 2 SE) still holds for v01_ftbank.

## Reading rules

- F1/F2 hold and F3 holds → a real-outcome fine-tuning stage is cheap and safe; report the in-domain gain.
- F1/F2 hold but F3 fails → fine-tuning specialises to one business; it must use the target company's own
  (or a closely matched) data, which a startup does not have.
- F1/F2 fail → the simulator prior is already as good as these real tasks can teach (consistent with R3 on Bank).

## Scoring (written after the run, 2026-11-25 ~21:30)

Fine-tuned 2,000 steps each on CPU at commit `9bd70d7` (runs/pfn_v01{,_nf}_ft{bank,hill}). Evaluations:
`real_bank_v01_ft{bank,hill}.{md,json}`, `real_hillstrom_v01_ft{bank,hill}.{md,json}`, `stress_sim_v01_ft{bank,hill}.*`;
paired comparisons with `compare_real.py` (30 seeds, identical markets verified).

| prediction | result | detail (seed-level mean ± SE of lift difference) |
|---|---|---|
| F1 Bank in-domain gain | **HOLDS (very large)** | ftbank − v01 on Bank +0.610 ± 0.045; ftbank − CPR-LP +0.697 ± 0.036 |
| F2 Hillstrom in-domain gain | **HOLDS** | fthill − v01 on Hillstrom +0.041 ± 0.011 |
| F3 cross-domain no harm (> −2 SE) | **HOLDS** | ftbank on Hillstrom +0.013 ± 0.014; fthill on Bank −0.025 ± 0.020 |
| F4 cross-domain gain (expected to fail) | **FAILS, as expected** | same two numbers; neither > 2 SE |
| F5 no forgetting on simulator S0 (≥ −0.10) | **HOLDS** | ftbank +0.037 ± 0.016; fthill −0.019 ± 0.019 |
| F6 Bank R1 still holds for ftbank | **HOLDS** | ρ = 1: +2.46, +1.21, +0.78, +0.94, +0.72 at n = 0, 1, 5, 10, 20 (all > 4 SE) |

### What the in-domain gain is (exploratory diagnosis, not pre-registered)

The Bank gain is concentrated where the founder is useless: at ρ ∈ {0, −0.5} with n ≥ 5, fine-tuned PC-RFM reaches
lift ≈ 2.8–3.2, versus 1.1–2.3 before fine-tuning and 1.0–1.8 for CPR-LP. The fine-tuned **no-founder** model
jumps the same way (e.g. ρ = 0, n = 5: 2.47 vs 1.15). So the gain is not about the founder. The model has learned a
regularity of this population that it can use even though category labels are shuffled per market:
- **Bank:** smaller segments buy more (students, retirees, the 60+ band); across segments in split B,
  Spearman(size, rate) = −0.36 (p = 0.04).
- **Hillstrom:** the sign is reversed (+0.37, p = 0.05).

That is why fine-tuning helps in-domain (same population in A and B) and does nothing across datasets (F3/F4).
No evaluation outcomes leaked (split B is never used in fine-tuning). Note that the base v0.1 no-founder model
already reaches lift 1.77 at n = 0 on Bank without labels, a smaller version of the same effect learned in simulation.

### Reading

Per the reading rules: F1/F2 and F3 hold, so a real-outcome fine-tuning stage is safe (no forgetting, no
cross-domain harm). But its gain comes from **population-specific structure that does not transfer between
businesses**: Bank and Hillstrom disagree on the sign. A startup has no such in-domain data. The honest claim is:
- PC-RFM can absorb a company's own historical outcomes when they exist, with large gains;
- fine-tuning on *other* companies' outcomes buys nothing on these two datasets.

The headline sim-to-real result stays the simulator-only model's (within 0.10–0.2 lift of CPR-LP on both real
tiers). The in-domain Bank number must not be presented as a founder-prior result.
