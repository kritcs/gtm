# M2 pre-registration: PC-RFM v1 / v2 on the relational simulator

Written and committed **before** any v1 or v2 model was trained to completion or evaluated.
The only results seen so far: M0 (CPR-LP ladder), M1 interim (v0 at step 8,500 on M0 markets), and
the baseline sweep on `sim_rel` (`results/rel_baselines.md`), which contains no PC-RFM numbers.

## Setup (fixed)

- Markets: `fprior.sim_rel.make_rel_market`, defaults (3,000 accounts), novelty e ∈ {0, 0.5, 1}.
- Founders and outreach: `fprior.sim` with the sanity-ladder RNG conventions; ρ ∈ {1, 0.5, 0, −0.5};
  n ∈ {0, 1, 5, 10, 20, 50} customers; 10 seeds.
- Metric: expected lift@100 on uncontacted accounts. Differences are paired over seeds.
- Models (all 20,000 steps, same architecture size, seed 0, trained on the randomized relational prior):
  v1 (features), v1-nf (v1 without founder inputs), v2 (learned account–product encoder),
  v2-nf. v0 (no graph, trained on the v0 prior) is evaluated as an ablation.
- Baselines: prior-only, similar-adopters, z(prior) + z(adopters) fixed fusion, labels-only,
  CPR-LP, and CPR-LP-rel (CPR-LP with a learned coefficient on similar-adopters) if it is ready.
- Script: `scripts/eval_pfn_rel.py` (plus the CPR-LP-rel and fusion columns if added). No changes to
  the models, simulator, or grid after seeing results; any later change is reported as exploratory.

## Predictions (primary comparisons)

A prediction **holds** if the paired mean difference has the predicted sign and exceeds 2 SE in at
least half of the cells named; it **fails** otherwise. Every result is reported either way.

- **P1, founder value:** v1 − v1-nf > 0 for ρ ∈ {1, 0.5} at n ∈ {0, 1, 5, 10, 20}, and the gap at
  e = 1 is larger than at e = 0 (founder beliefs matter most in new categories).
- **P2, relational value:** v1 − v0 > 0 at e = 0 for every ρ and n ≥ 1; |v1 − v0| < 2 SE at e = 1.
- **P3, bounded harm from bad founders:** for ρ ∈ {0, −0.5} and n ≥ 5, v1 − v1-nf ≥ −0.2 lift
  (no meaningful loss once labels exist). At n = 0 harm is expected and not counted against P3.
- **P4, beats the fixed fusion where it breaks:** at e = 1, v1 − fusion > 0 for ρ = 1 (the fusion
  is hurt by a noise feature; v1 should learn to ignore it).
- **P5, learned encoder:** v2 − v1 > 0 at e = 0 for n ∈ {10, 20, 50} (labels can flow through
  products); |v2 − v1| < 2 SE at e = 1.
- **P6, against the principled baseline:** v1 is within 0.2 lift of, or better than, the best of
  {CPR-LP, CPR-LP-rel} in at least two thirds of all (e, ρ, n) cells.

## Exploratory (not pre-registered)

Per-cell tables, AUC, v2-nf, behaviour at n = 50, and anything about why a prediction failed.
