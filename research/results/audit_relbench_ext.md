# Benchmark audit, contrast tasks (rel-trial, rel-stack)

MAP@k x100 (k = task eval_k). Heuristics computed here; published numbers from RelBench Table 5 (test).

| task | k | repeat | cf-nbr | novel | repeat+pop | best published | share of best |
|---|---|---|---|---|---|---|---|
| rel-trial/condition-sponsor-run | 10 | 38.5% | 16.3% | 45.2% | 7.15 | 11.36 | 63% |
| rel-trial/site-sponsor-run | 10 | 23.3% | 8.2% | 68.5% | 20.24 | 19.00 | 107% |
| rel-stack/user-post-comment | 100 | 14.7% | 1.4% | 83.9% | 5.10 | 12.72 | 40% |
| rel-stack/post-post-related | 100 | 8.1% | 1.7% | 90.3% | 8.24 | 10.83 | 76% |

Per slice (MAP x100):

| task | slice | users | global_popularity | recent_popularity | repeat_then_recent_pop |
|---|---|---|---|---|---|
| rel-trial/condition-sponsor-run | overall | 2057 | 2.26 | 3.03 | 7.15 |
| | repeat | 1202 | 4.05 | 5.08 | 18.80 |
| | cf_nbr | 1105 | 1.28 | 2.69 | 0.11 |
| | novel | 1721 | 0.05 | 0.05 | 0.03 |
| | cold_user | 32 | 1.15 | 1.52 | 1.52 |
| | new_item | 980 | 0.00 | 0.00 | 0.00 |
| | head10 | 1843 | 2.76 | 3.79 | 7.20 |
| rel-trial/site-sponsor-run | overall | 27428 | 2.43 | 4.63 | 20.24 |
| | repeat | 8002 | 4.99 | 7.67 | 73.54 |
| | cf_nbr | 3171 | 4.96 | 9.98 | 2.28 |
| | novel | 20866 | 1.19 | 2.80 | 2.74 |
| | cold_user | 14169 | 1.64 | 3.95 | 3.95 |
| | new_item | 4489 | 0.00 | 0.00 | 0.00 |
| | head10 | 21470 | 3.23 | 6.16 | 23.43 |
| rel-stack/user-post-comment | overall | 758 | 0.01 | 0.02 | 5.10 |
| | repeat | 132 | 0.01 | 0.10 | 35.24 |
| | cf_nbr | 16 | 0.00 | 0.00 | 0.00 |
| | novel | 681 | 0.01 | 0.00 | 0.00 |
| | cold_user | 138 | 0.00 | 0.00 | 0.00 |
| | new_item | 354 | 0.00 | 0.00 | 0.00 |
| | head10 | 94 | 0.09 | 0.15 | 6.40 |
| rel-stack/post-post-related | overall | 258 | 0.60 | 0.58 | 8.24 |
| | repeat | 23 | 1.75 | 0.49 | 92.39 |
| | cf_nbr | 6 | 4.43 | 6.25 | 3.57 |
| | novel | 234 | 0.45 | 0.46 | 0.46 |
| | cold_user | 166 | 0.46 | 0.62 | 0.62 |
| | new_item | 78 | 0.00 | 0.00 | 0.00 |
| | head10 | 97 | 1.83 | 2.14 | 16.46 |

## Mechanical scoring

- FAILS — E1 rel-trial repeat+pop >= 75% of best: 63%, 107%
- FAILS — E2 rel-stack repeat+pop < 25% of best: 40%, 76%
- FAILS — E3 novel share <= 60% (trial), >= 80% (stack): 45%, 68%, 84%, 90%
- HOLDS — E4 best heuristic novel MAP < half its overall, every task: condition-sponsor-run: repeat_then_recent_pop novel 0.03 vs overall 7.15; site-sponsor-run: repeat_then_recent_pop novel 2.74 vs overall 20.24; user-post-comment: repeat_then_recent_pop novel 0.00 vs overall 5.10; post-post-related: repeat_then_recent_pop novel 0.46 vs overall 8.24

## Scoring (written after the run, 2026-11-25 ~13:55)

| prediction | result | detail |
|---|---|---|
| E1 rel-trial: repeat+pop ≥ 75% of best published | **FAILS (mixed)** | site-sponsor-run 107% (the heuristic *beats* ID-GNN, 20.24 vs 19.00); condition-sponsor-run 63% |
| E2 rel-stack: repeat+pop < 25% of best published | **FAILS (critique transfers more than predicted)** | 40% (user-post-comment), 76% (post-post-related) |
| E3 novel share ≤ 60% (trial), ≥ 80% (stack) | **FAILS (one task)** | 45%, 68% (site-sponsor-run), 84%, 90% |
| E4 best heuristic's novel-slice MAP < ½ its overall, every task | **HOLDS** | novel-slice MAP 0.03, 2.74, 0.00, 0.46 vs overall 7.15, 20.24, 5.10, 8.24 |

Readings:
- **Across six RelBench recommendation tasks, a no-learning repeat + popularity heuristic reaches 40–107% of
  the best published MAP** (rel-hm 93%, rel-avito 88%, site-sponsor 107%, post-post-related 76%,
  condition-sponsor 63%, user-post-comment 40%). Its score comes almost entirely from the repeat slice
  (MAP 19–92 there) and it is ~0 on novel pairs (45–90% of positives).
- **Our repeat heuristic is much stronger than RelBench's own "Past Visit" baseline on rel-stack** (5.10 vs 1.42;
  8.24 vs 1.74). Likely reason (not verified in RelBench's code): Past Visit uses links from earlier *training-table*
  label windows; our heuristic uses every interaction in the database up to T (all past comments; all existing
  post links, including pairs re-linked with another link type). Both are legal at time T. Our "repeat" share
  (14.7%, 8.1%) is correspondingly larger than the paper's "%Repeated links" (4.09, 1.39), which counts
  training-table links. So the published baseline row understates how much of these tasks is memorisable.
- What is learned beyond memorisation: condition-sponsor-run and user-post-comment keep a large gap
  (ID-GNN 11.36 vs 7.15; 12.72 vs 5.10). Whether that gap comes from novel pairs or from ranking repeats better
  needs ID-GNN's per-slice predictions on these tasks (as done for rel-hm in `idgnn_slices.md`).
- Settings fixed before running (windows, K = task eval_k); 2-hop shares in `audit_relbench_ext.json`.
