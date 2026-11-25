# Benchmark audit — notes

First cut (rel-hm user-item-purchase, rel-avito user-ad-visit): `results/audit_relbench.md`.

Headline: a no-learning "repeat past interactions, then recent popularity" list reaches **93% / 88%** of the
best MAP published in the RelBench paper, while only **3.8% / 6.6%** of test positives are repeats. On the
**82% / 93%** of positives that are novel pairs, every heuristic scores near zero (rel-avito: 71% of positives
are items never seen before T). So the leaderboard score is carried by a memorizable minority — the part a new
startup does not have. The original claim "most answers are memorizable" is false and should not be made.

Open: per-slice performance of learned models (GraphSAGE / ID-GNN / ContextGNN); more tasks (rel-trial,
rel-stack, rel-amazon); a startup-relevant slice definition (new seller / new item) as the reported metric.

## Part 2: learned recommenders per slice (rel-hm, 20k-user subsample)

Script: `scripts/audit_models.py` (pre-registered LM1–LM3 in its docstring). Results:
`results/audit_models_subsample.md` (BPR/LightGCN 10 epochs, L2 1e-4) and `results/audit_models_subsample_reg.md`
(3 epochs, L2 1e-3). Same slices and MAP@12 as Part 1, on 20,000 random test users (82.6% of positives novel).
Models: item-item cosine CF, BPR matrix factorization, LightGCN (2 layers), ids only, trained on the last 90 days
up to T (the 20k users + 100k other active users), plus hybrids that list the user's past items first.

MAP@12 × 100, run with 10 epochs:

| slice | recent pop | repeat + pop | item-CF | BPR-MF | LightGCN | repeat + LightGCN |
|---|---|---|---|---|---|---|
| overall | 0.93 | **2.67** | 0.44 | 1.15 | 0.93 | 2.65 |
| repeat | 0.93 | **42.3** | 3.4 | 13.3 | 7.1 | 42.3 |
| co-purchase nbr | **2.35** | 0.08 | 0.91 | 0.59 | 1.79 | 0.17 |
| novel | **0.60** | 0.15 | 0.10 | 0.22 | 0.26 | 0.12 |
| new item | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

- **LM1 holds** (in the "not at all" form): no learned model beats repeat + popularity overall; hybrids tie it (2.62–2.65).
- **LM2 holds:** every learned model scores 0.10–0.26 on the novel slice, below plain recent popularity (0.60).
- **LM3 is vacuous** here: there is no overall gain to attribute.
- **Caveat (important):** these models are lightly tuned (ids only, no item attributes, no recency features,
  simple training window). The best published model (ID-GNN, 2.81 full test) is stronger than ours (≤ 1.15),
  so this does not show that *published* GNNs score ≈ 0 on novel pairs. What it shows: standard learned
  recommenders of the kind a startup could run do not find novel buyers better than popularity, and nearly all
  of the achievable score sits in the 3.8% repeat slice. Settling the published-model question needs RelBench's
  ID-GNN scored per slice (requires torch_geometric; Colab).
- Runtime: 130 s (10 epochs) / 53 s (3 epochs) on CPU, peak ~2.9 GB.
- Full run (all 67,144 test users, all users in the window) on Colab GPU, estimated ~10–20 min:
  `python research/scripts/audit_models.py --max-users 0 --extra-users 0 --device cuda --epochs 10 --out results/audit_models_full`
  (needs `research/data/relbench/rel-hm` present on the VM; LightGCN cost grows with the full user graph).
