# ID-GNN per-slice audit on rel-trial / rel-stack — scoring (written after the run)

Pre-registration: `scripts/idgnn_slices.py` docstring, "EXTENSION" (commit `276f94e`, before training). Run: RelBench
v3.1.0 `examples/idgnn_recommendation.py`, default hyperparameters, Colab L4 high-mem. Per-task tables:
`idgnn_slices_<task>.md`. The four tasks were first launched concurrently; both rel-trial jobs were killed
(out of RAM, exit 137) during text embedding and were rerun one at a time.

## rel-stack (scored 2026-11-25 ~14:45)

| prediction | task | result | detail |
|---|---|---|---|
| J1 reproduction within 1.0 | user-post-comment | **HOLDS** | 13.38 vs 12.72 |
| J1 reproduction within 1.0 | post-post-related | **HOLDS** | 10.89 vs 10.83 |
| J3 novel MAP ≥ 1.0 and > 2× best heuristic | user-post-comment | **HOLDS** | novel 9.50 vs 0.00 |
| J4 novel MAP < ½ overall | user-post-comment | **FAILS** | novel 9.50 vs overall 13.38 |
| J4 novel MAP < ½ overall | post-post-related | **HOLDS** | novel 3.28 vs overall 10.89 (repeat slice 93.5) |

### Exploratory follow-up (NOT pre-registered; `scripts/audit_stack_ownthread.py`)

ID-GNN's high novel-slice and cold-user scores on user-post-comment (9.50; 17.69 for users with no comment
history) pointed to a relation the comment-history heuristic ignores. Users comment in their **own threads**
(posts they wrote, answers to their questions, questions they answered, sibling answers — all from the posts
table, created ≤ T):
- 25% of test positives are own-thread posts.
- **91% of ID-GNN's correct hits (245 / 269) are own-thread posts.**
- A no-learning heuristic "own-thread posts newest first, then comment repeats, then recent popularity" scores
  **10.41 MAP@100 = 78% of ID-GNN** (the comment-repeat heuristic alone: 5.10 = 40%).

Reading: J3 "holds" only because our pre-registered "novel" slice (defined from comment history) is too narrow
for this schema; ID-GNN's success on it is again graph-local lookup — through post ownership instead of past
comments. On post-post-related almost all of ID-GNN's score is the 8% of positives that are re-links of an
existing pair (repeat slice 93.5). The general statement after rel-hm and rel-stack: **the best published model
wins by finding the target in the entity's immediate neighbourhood (its own past links or own content), and is
near popularity on pairs that are genuinely new.** What "neighbourhood" means is schema-specific, which is why a
fixed "repeat" definition under-counts it.
