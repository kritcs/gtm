# Section draft: Real outcomes with a simulated founder

*Draft v0.1, 2026-11-25. Sources: `results/real_bank.md`, `results/real_hillstrom_v01.md`, `results/placebo_bank.md`,
`results/bank_noise_control*.md`, `results/finetune_preregistration.md`, `results/sim_calibration.md`. All
predictions were registered before the corresponding run; failures are reported as failures.*

## Why this tier

Simulation results are optimistic, because the simulator and both models share assumptions. No public dataset
records what founders believed before outreach, so this tier keeps the **outcomes real** and simulates only the
founder. A founder of reliability ρ holds beliefs about segment rates that correlate ρ with rates estimated on a
disjoint half of the data ("experience"). The other half supplies the markets and their real outcomes. Outreach is
founder-steered, as it would be in practice. Category indices are relabelled at random per market, so no model can
score by matching our numbering.

## Data

| | UCI Bank Marketing | Hillstrom e-mail test |
|---|---|---|
| Business | telemarketing for term deposits | retail e-mail campaign (randomized) |
| Outcome (rate) | subscribed (11%) | site visit after e-mail (16.7%) |
| Segments (8 × 4) | job group × age band | recency × new-customer flag, by spend band |
| Segment signal (logit sd) | 0.47 | 0.34–0.36 |
| Where it sits in the simulator's range | inside | inside, near the low end |

## Pre-registered predictions and results

The same three predictions were registered for both datasets. Metric: lift@100 on uncontacted accounts using
real outcomes, 30 seeds.

| prediction | Bank | Hillstrom |
|---|---|---|
| R1 a ρ = 1 founder helps both models beat labels-only at n ∈ {0, 1, 5, 10, 20} | **holds** (+0.7 to +2.5) | **fails** in 1–2 of 5 cells (n = 1 for both; n = 20 CPR-LP; n = 10 PC-RFM at exactly 2 SE) |
| R2 a wrong founder (ρ ≤ 0) costs at most 0.2 lift for n ≥ 5 | **holds** | **fails narrowly** (−0.20 CPR-LP, −0.21 PC-RFM at ρ = −0.5, n = 5; gone by n = 20) |
| R3 PC-RFM, trained only on simulations, is within 0.2 lift of CPR-LP in ≥ ⅔ of cells | **holds** (worst −0.19) | **holds** (24 / 24 cells, worst gap 0.10) |

**The sim-to-real result replicates.** On two real datasets from different businesses, a model trained only on
simulated markets performs as well as a closed-form Bayesian model that needs no simulator.

**The founder value depends on how much the segments matter.** On Hillstrom, where segments carry little signal,
even a perfect founder is worth 0.1–0.4 lift, and real-outcome noise often hides it (on AUC the ρ = 1 gain is
+0.06 to +0.09 at every n). We had flagged this risk in advance.

## Controls that changed our claims

1. **Placebo beliefs** (shuffled or flat beliefs at the same confidence).
   - CPR-LP passes: no gain (|Δ| ≤ 0.11), and its learned trust drops from 0.69–0.87 to 0.15–0.26.
   - PC-RFM gains +0.3 to +0.46 lift even with shuffled beliefs. Its apparent advantage with bad founders is
     therefore mostly not trust learning.
   - We now evaluate PC-RFM's founder use against its own noise-belief mode, not against CPR-LP.
2. **Choice of null.** Whether an inverted founder "costs" PC-RFM depends on what it is compared with.
   - Against account-scrambled beliefs, which are easy to recognise as noise, it costs −0.29 at n = 10.
   - Against segment-shuffled beliefs, which are structured but wrong, it costs nothing.
   - We report both nulls and state which claim each supports.
3. **Fine-tuning on real outcomes** (2,000 steps on the experience half only; the evaluation half is never used).
   - Fine-tuning on Bank adds +0.61 lift on Bank, mostly where the founder is useless.
   - Across datasets it does nothing: +0.01 and −0.03, neither significant.
   - Diagnosis (exploratory): the models learn how segment size relates to buying rate, and that relation has
     opposite signs in the two datasets (−0.36 Bank, +0.37 Hillstrom).
   - So fine-tuning absorbs a company's *own* history when it exists, but other companies' outcomes do not
     transfer. The founder-prior claims rest on the simulator-only model.

## Limitations

- **Simulated founders.** The founders are simulated, with beliefs built from the same population's rates. Real
  founders are the subject of the prospective cohort (`docs/cohort/`).
- **Few segments.** Two datasets, 32 segments each; no graph in this tier, so CPR-LP runs without its graph
  term.
- **Bank's outreach was not randomized.** Bank's calls were targeted; every account has an outcome, but the
  population is the bank's own selection. Hillstrom's e-mails were randomized.
- **Unresolved effect.** The no-founder PC-RFM reaches lift 1.77 on Bank with no labels. We have not explained
  this.
