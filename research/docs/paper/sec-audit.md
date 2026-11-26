# Section draft: Relational recommendation benchmarks reward memorisation, not finding new customers

*Draft v0.1, 2026-11-25. Sources for every number: `results/audit_relbench.md`, `results/audit_relbench_ext.md`,
`results/audit_models_subsample.md`, `results/idgnn_slices.md`, `results/idgnn_ext_scoring.md`. All predictions
were committed before the corresponding numbers were computed; failed predictions are reported as failures.*

## Motivation

A company with few customers needs to find accounts that have never bought from it. Public relational benchmarks
are the natural place to test whether graph models can do this. We ask a narrower question first: how much of a
leaderboard score on these benchmarks comes from predicting links the model could have looked up?

## Setup

We use the six recommendation (link-prediction) tasks of RelBench (Robinson et al., 2024) for which test labels
and published baselines are available: rel-hm user-item-purchase, rel-avito user-ad-visit, rel-trial
condition-sponsor-run and site-sponsor-run, rel-stack user-post-comment and post-post-related. Metrics are the
official ones (MAP@k with the task's k); our implementation reproduces the published global-popularity
baseline and the published repeat shares.

Each test positive (u, i) is assigned to one slice, using only data up to the test timestamp:
- **repeat**: (u, i) already occurred in the database;
- **cf-neighbour**: not a repeat, but i is among the 20 items most co-interacted with something u interacted with;
- **novel**: neither.

We compare three things on each slice:
- a **no-learning heuristic**: u's own past items, most recent first, then recently popular items;
- the **published leaderboard** numbers;
- the strongest published model, **ID-GNN**, which we retrain with RelBench's own example script and default
  hyperparameters, then score per slice.

## Findings

**1. A no-learning heuristic reaches most of the leaderboard.**

| task | repeat + popularity, % of best published MAP | novel share of test positives |
|---|---|---|
| rel-hm purchase | 93% | 82% |
| rel-avito visit | 88% | 93% |
| rel-trial site-sponsor | **107%** (beats ID-GNN, 20.2 vs 19.0) | 68% |
| rel-stack post-related | 76% | 90% |
| rel-trial condition-sponsor | 63% | 45% |
| rel-stack user-comment | 40% (78% with an "own threads" rule, exploratory) | 84% |

The heuristic's score comes almost entirely from the repeat slice (MAP 19–92 there). It scores approximately
zero on novel pairs in all six tasks.

RelBench's own "Past Visit" baseline understates how much of rel-stack is memorisable: 1.42 versus 5.10 for
our heuristic on user-post-comment, and 1.74 versus 8.24 on post-post-related. We believe, without having checked
the code, that Past Visit only sees links from earlier training-table windows, while all database
interactions up to the test time are legal inputs.

**2. The best published model wins by re-ranking what the entity already did.** We reproduced ID-GNN exactly
on rel-hm (2.81) and closely on rel-stack (13.38 vs 12.72; 10.89 vs 10.83).

- **rel-hm:** 99.8% of its correct top-12 predictions are repeat purchases (6,764 of 6,777). Its MAP on novel
  pairs is 0.00, below recent popularity (0.58), and also 0.00 for customers with no history. Its entire lead
  over the heuristic (+0.20 MAP) comes from ordering repeats better (50.8 vs 42.8 on that slice).
- **rel-stack user-post-comment:** ID-GNN does score on pairs that are new by comment history (9.5). But in an
  exploratory follow-up (not pre-registered), 91% of its hits are posts in the user's own threads: posts they
  wrote, answers to their questions, or questions they answered. A lookup rule over that relation reaches 78% of
  ID-GNN. The model is still finding the target in the entity's immediate neighbourhood, through a relation the
  comment-history definition did not include.
- **rel-stack post-post-related:** the 8% of positives that re-link an existing pair carry the score (slice MAP
  93.5); novel-pair MAP is 3.28.
- **rel-trial:** we could not train ID-GNN. The loss was NaN from the first epoch, twice, including with a fresh
  cache, on the same software stack that trained the other tasks normally. We report this as a failed
  reproduction.

**3. Learned recommenders without the graph do no better on novel pairs.** On a 20,000-user subsample of
rel-hm, item-CF, BPR-MF and LightGCN reach 0.10–0.26 MAP on novel pairs, against 0.60 for recent popularity.

## What we predicted and got wrong

- **A1 (pre-registered): "most positives are memorisable" failed.** 82–93% of test positives are novel under
  our definition. The critique is not that the tasks are mostly repeats. It is that scores are earned mostly on
  the minority of positives that are repeats, while the novel majority is not predicted by anything we tested.
- **Contrast-task predictions (E1–E3):** we expected the heuristic to reach at least 75% of the leaderboard on
  both rel-trial tasks and under 25% on rel-stack. Both failed. Memorisation explains more of rel-stack than
  predicted (40% and 76%) and less of condition-sponsor (63%).
- **"Neighbourhood" depends on the schema.** A fixed definition (past links plus co-interaction) undercounts it.
  The own-thread finding on rel-stack shows a slice definition has to be schema-aware.

## Implication

Leaderboard MAP on these tasks is dominated by a slice that does not exist for a new company: it has no past
links to repeat. On the slice that matches the cold-start problem (novel pairs, users with no history), the best
published model is at or below popularity on rel-hm and well below its own overall score on rel-stack. We
therefore use these benchmarks only as a competence check. Headline results come from slice-aware evaluation:
the novel and cold-entity slices, alongside simulation with known ground truth and real-outcome tiers.

**Recommendation for benchmark designers:** report MAP separately on repeat, neighbourhood and novel positives,
and include a "repeat then popularity" baseline that uses all legal history.

## Limitations

- **One run per model.** ID-GNN was trained once per task with default hyperparameters; published variance is
  small (±0.01–0.22) and our runs match the published means.
- **Ad hoc neighbourhood definition.** The cf-neighbour definition (top 20 co-interactions, recent window) is
  a choice.
- **Exploratory own-thread analysis.** It was not pre-registered.
- **rel-trial gap.** ID-GNN could not be reproduced on rel-trial.
- **Scope.** Six tasks from one benchmark suite. Other suites (e.g. Amazon reviews in RelBench v2) were not
  audited.
