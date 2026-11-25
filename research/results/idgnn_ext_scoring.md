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

## rel-trial, first attempt (2026-11-25 ~15:30) — INVALID RUN, not scored

Both rel-trial runs trained with `Train loss: nan` from epoch 1 and constant validation MAP (test MAP 0.33 and
0.14 vs published 11.36 and 19.00; logs kept locally in `runs/idgnn_ext/`). They loaded materialised tensor
frames from the cache written by the first, OOM-killed concurrent attempt; a partially written cache is the
suspected cause (not verified). Rerun on a fresh session with an empty cache, one task at a time; predictions
unchanged. If the rerun also gives NaN, J1/J2 are reported as "could not reproduce" and not scored on slices.

## rel-trial, rerun (2026-11-25 ~16:30) — could not reproduce

Fresh L4 session, empty cache, one task at a time: `Train loss: nan` again from epoch 1 (condition-sponsor-run,
stopped at epoch 8 to save units; site-sponsor-run not started). So the cache was not the cause. Ruled out from the
data: no inf / constant numeric columns in any rel-trial table; no empty or null target lists in train / val.
Stack: RelBench v3.1.0 example, torch 2.14.1+cu130, torch_geometric 2.8.1, pytorch_frame (latest), pyg_lib
built for torch 2.14.0. rel-hm and rel-stack trained normally on the same stack.

| prediction | task | result |
|---|---|---|
| J1 reproduction within 1.0 | condition-sponsor-run, site-sponsor-run | **FAILS — could not reproduce (NaN training)** |
| J2 rel-trial hits mostly repeat/cf-nbr | both | **not scored** (no valid model) |
| J4 novel MAP < ½ overall | both | **not scored** |

Not pursued further: locating the NaN needs debugging inside RelBench / PyG on GPU time, and the heuristic audit
already answers the main rel-trial question (repeat + popularity = 107% of published ID-GNN on site-sponsor-run,
63% on condition-sponsor-run; `audit_relbench_ext.md`).

## Summary across tasks

| task | ID-GNN (ours / published) | hits from immediate neighbourhood | ID-GNN novel-slice MAP | no-learning heuristic / ID-GNN |
|---|---|---|---|---|
| rel-hm user-item-purchase | 2.81 / 2.81 | 99.8% repeat purchases | 0.00 (popularity 0.58) | 93% |
| rel-stack user-post-comment | 13.38 / 12.72 | 91% own threads (exploratory) | 9.50 by comment history; own-thread explains it | 78% (own-thread heuristic, exploratory) |
| rel-stack post-post-related | 10.89 / 10.83 | repeat slice MAP 93.5 | 3.28 | 76% |
| rel-trial (×2) | NaN training | — | — | 63% / 107% of published |
