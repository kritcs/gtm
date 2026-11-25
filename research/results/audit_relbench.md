# Benchmark audit, first cut: RelBench recommendation tasks

Script: `scripts/audit_relbench.py` (run `python scripts/audit_relbench.py 300`). Raw numbers: `audit_relbench.json`.

## Pre-registration (verbatim from the script, written before any slice was computed)

```
Benchmark audit, first cut: how memorizable are RelBench recommendation test answers?

PRE-REGISTERED (written before any slice was computed; see note on what was already known):
  A1. Most test positives are "memorizable": repeats of a past (user, item) interaction or within 2 hops
      of the user in the training interaction graph.
  A2. A repeat-past-interactions heuristic backfilled with popularity reaches >= 50% of the best MAP
      published in the RelBench paper for the same task.
  A3. On the novel-pair slice (neither repeat nor graph-neighbourhood) all heuristics are near zero.
  Known before writing this (RelBench paper, arXiv 2407.20060, Table 15): repeated links are only 3.76%
  of rel-hm and 4.73% of rel-avito test links, so the "repeat" half of A1 was already known to be false;
  the 2-hop half and A2/A3 were not known.

Definitions (history = all interactions with time <= the test timestamp T; label window is (T, T + delta]):
  repeat       (u, i) occurred in history
  cf-nbr       not repeat; i is among the top-20 co-interacted items (same-user co-occurrence in a recent
               window) of some item u interacted with — a tight "close in the graph" definition
  2-hop        i was interacted by some user who shares any item with u (literal 2-hop; measured on a
               random sample of test users because it is expensive and close to "everything" in dense graphs)
  novel        neither repeat nor cf-nbr
  cold-user    u has no history;  new-item: i has no history
  head-1/10%   i in the top 1% / 10% of items by history count
Metric: RelBench MAP@12 (relbench.metrics.map), reported x100 like the paper.
Data: RelBench v1 on the Hugging Face Hub (stanford-star/relbench-v1), CDLA-Permissive-2.0 for task
definitions; upstream licences per database.
```

Published numbers: RelBench paper, arXiv 2407.20060, Table 5 / Table 8 (test MAP x100) — https://arxiv.org/html/2407.20060

## Overall test MAP@12 (x100)

| task | global popularity | recent popularity | **repeat → recent-pop (ours)** | published Past Visit | published GraphSAGE | published best (ID-GNN) | ours / best |
|---|---|---|---|---|---|---|---|
| rel-hm/user-item-purchase | 0.318 | 0.917 | **2.612** | 0.89 | 0.80 | 2.81 | 93% |
| rel-avito/user-ad-visit | 0.001 | 0.001 | **3.236** | 1.95 | 0.02 | 3.66 | 88% |

Check: our global popularity (0.318 / 0.001) matches the published 0.30 / 0.00, and our rel-hm repeat share (3.76%) matches the paper's Table 15, so the pipeline and metric agree with RelBench.

## Where the test positives are

| task | positives | repeat | cf-nbr (top-20 co-items) | novel | literal 2-hop, not repeat (sample of 300 users) | catalogue reachable in 2 hops (median) | cold user | new item | head 1% | head 10% |
|---|---|---|---|---|---|---|---|---|---|---|
| rel-hm/user-item-purchase | 207,996 | 3.8% | 13.7% | 82.5% | 86.2% | 64.0% | 10.3% | 4.1% | 12.6% | 41.8% |
| rel-avito/user-ad-visit | 712,985 | 6.6% | 0.9% | 92.5% | 7.1% | 0.2% | 18.3% | 71.4% | 1.9% | 8.3% |

## MAP@12 (x100) of each heuristic on each slice of positives

(Targets restricted to the slice; users with no positive in the slice dropped.)

**rel-hm/user-item-purchase**

| slice | users | global pop | recent pop | repeat → recent-pop |
|---|---|---|---|---|
| repeat | 6,196 | 1.623 | 0.885 | 42.846 |
| cf_nbr | 18,376 | 0.924 | 2.391 | 0.068 |
| novel | 61,251 | 0.153 | 0.580 | 0.163 |
| cold_user | 7,186 | 0.363 | 0.859 | 0.859 |
| new_item | 5,798 | 0.000 | 0.000 | 0.000 |
| head10 | 44,364 | 0.690 | 2.014 | 2.942 |

**rel-avito/user-ad-visit**

| slice | users | global pop | recent pop | repeat → recent-pop |
|---|---|---|---|---|
| repeat | 10,481 | 0.002 | 0.005 | 24.943 |
| cf_nbr | 2,701 | 0.000 | 0.000 | 0.000 |
| novel | 35,315 | 0.000 | 0.000 | 0.000 |
| cold_user | 11,046 | 0.000 | 0.000 | 0.000 |
| new_item | 34,109 | 0.000 | 0.000 | 0.000 |
| head10 | 13,926 | 0.010 | 0.007 | 8.558 |

## Pre-registered claims

- **A1 (most positives memorizable): fails as stated.** Repeats are 3.8% (rel-hm) and 6.6% (rel-avito) of positives; the tight co-item neighbourhood adds 13.8% / 0.9%, so 82% / 93% of positives are *novel* by that definition. Literal 2-hop covers 86% of rel-hm positives, but only because 2 hops reach a median 64% of the whole catalogue there, so it is not a meaningful notion of "memorizable"; on rel-avito literal 2-hop covers 7%.
- **A2 (repeat + popularity heuristic ≥ 50% of best published MAP): holds strongly.** 93% (rel-hm, 2.61 vs 2.81) and 88% (rel-avito, 3.24 vs 3.66). Both beat the paper's own Past Visit baseline (0.89 / 1.95) and GraphSAGE (0.80 / 0.02) by a wide margin.
- **A3 (near zero on the novel slice): holds.** Novel-slice MAP: rel-hm ≤ 0.58 (recent pop) vs 2.6 overall; rel-avito 0.000 for all heuristics. New items (71% of rel-avito positives) score exactly 0.

## What this means (and does not)

- The leaderboard score is carried by a small, memorizable slice: on rel-hm the repeat heuristic scores 42.8 MAP on the 3.8% of positives that are repeats and ~0.16 on the 82% that are novel. A recency-ordered repeat list with popularity backfill — no learning — gets within 7–12% of the best published GNN on both tasks.
- The precise version of our critique is therefore not "most answers are memorizable" (false) but "**most of the score comes from the memorizable minority**; the majority of answers (new pairs, new items) is barely scored by anything we tested". Whether published GNNs do better on the novel slice is **unknown** from this audit: it needs their per-user predictions (next step: train RelBench's GraphSAGE / ID-GNN baselines, or a cheap item-CF / two-tower model, and score per slice).
- The paper's trivial baselines are weak (Past Visit 0.89 vs our 2.61 on rel-hm), so reported relative gains of GNNs over "simple baselines" (e.g. +214% on rel-hm) overstate the margin over a properly tuned heuristic.

## Caveats

- Our repeat share for rel-avito (6.6%) differs from the paper's 4.73%; we count repeats against all history up to T, the paper likely against the task's training-table links.
- The co-item neighbourhood uses a recent window (28 / 20 days) and caps users at 200 items; other definitions will shift cf-nbr vs novel.
- Literal 2-hop is measured on 300 sampled test users per task.
- History includes interactions on day T; labels are strictly after T (as in the task SQL).
- Peak memory ~11 GB (rel-avito co-occurrence over 1.8M ads); fine on this laptop, not on small machines.

## Feasibility of extending

- rel-amazon (7.7 GB) and rel-stack (1.1 GB) are feasible with the same code but need more memory; rel-trial site-sponsor-run / condition-sponsor-run (683 MB) are small and have high repeat rates (23% / 38% per the paper's Table 15), a useful contrast.
- The decisive next step is per-slice scores for a learned model (published predictions are not released).
