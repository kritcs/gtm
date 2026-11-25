# Benchmark audit — notes

First cut (rel-hm user-item-purchase, rel-avito user-ad-visit): `results/audit_relbench.md`.

Headline: a no-learning "repeat past interactions, then recent popularity" list reaches **93% / 88%** of the
best MAP published in the RelBench paper, while only **3.8% / 6.6%** of test positives are repeats. On the
**82% / 93%** of positives that are novel pairs, every heuristic scores near zero (rel-avito: 71% of positives
are items never seen before T). So the leaderboard score is carried by a memorizable minority — the part a new
startup does not have. The original claim "most answers are memorizable" is false and should not be made.

Open: per-slice performance of learned models (GraphSAGE / ID-GNN / ContextGNN); more tasks (rel-trial,
rel-stack, rel-amazon); a startup-relevant slice definition (new seller / new item) as the reported metric.
