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
