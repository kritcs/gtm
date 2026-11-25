# ID-GNN per-slice audit, rel-hm/user-item-purchase

MAP@12 x100.

| slice | users | ID-GNN | recent popularity |
|---|---|---|---|
| overall | 67144 | 2.81 | 0.92 |
| repeat | 6196 | 50.81 | 0.89 |
| cf_nbr | 18376 | 0.00 | 2.39 |
| novel | 61251 | 0.00 | 0.58 |
| cold_user | 7186 | 0.00 | 0.86 |
| new_item | 5798 | 0.00 | 0.00 |
| head10 | 44364 | 3.13 | 2.01 |

## Mechanical scoring

- HOLDS — I1 reproduction: MAP 2.81 vs published 2.81
- HOLDS — I2 novel ≤ 2× recent popularity: novel MAP 0.00 vs recent-pop 0.58
- HOLDS — I3 hits mostly repeat/cf-nbr: 6764 repeat, 1 cf-nbr, 12 novel hits

## Scoring (written after the run, 2026-11-25 ~14:05)

Run: RelBench v3.1.0 `examples/idgnn_recommendation.py`, default hyperparameters (20 epochs, ≤ 2,000 steps/epoch,
batch 512, 128 channels), seed 42, one Colab L4 (~3.5 h incl. graph materialisation; ~9 units). Test predictions
from RelBench's `write_prediction_table`; scored by this script at commit `f7ff24b`.

| prediction | result | detail |
|---|---|---|
| I1 reproduction within 0.5 of 2.81 | **HOLDS** | 2.81 (RelBench's own evaluator printed 0.02806) — we label it "published ID-GNN" |
| I2 novel-slice MAP ≤ 2× recent popularity (1.16) | **HOLDS** | 0.00 vs 0.58 — ID-GNN is *below* popularity on novel pairs |
| I3 > 50% of hits are repeat / cf-nbr | **HOLDS (extreme)** | 6,764 repeat + 1 cf-nbr of 6,777 hits = 99.8%; 12 novel hits in 67,144 users |

Comparison with the no-learning heuristic (`audit_relbench.json`): repeat_then_recent_pop overall 2.61, repeat
slice 42.8, novel 0.16, cold users 0.86. ID-GNN: overall 2.81, repeat 50.8, novel 0.00, cold users 0.00.

Reading: **the strongest published model on rel-hm is a better re-ranker of items the customer already bought.**
All of its advantage over the heuristic (+0.20 MAP) comes from ordering repeats (50.8 vs 42.8 on that slice);
it scores nothing on the 82% of test positives that are new (customer, item) pairs, and nothing for customers
with no history, where even popularity gets 0.86. This is the benchmark-side version of the paper's thesis:
current relational benchmarks reward memorisation, and a model can top them while being useless for finding
new customers.
