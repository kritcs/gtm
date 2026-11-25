Benchmark audit, part 2: how do LEARNED recommenders score on each slice of rel-hm?

PRE-REGISTERED (written before any model in this script was trained or scored):
  LM1. Learned models (item-CF, BPR matrix factorization, LightGCN) beat the repeat + recent-popularity
       heuristic overall only modestly (or not at all).
  LM2. They score near zero on the novel-pair slice (positives that are neither repeats nor among the
       top-20 co-purchased items of something the user bought) — same order as the heuristics there.
  LM3. Any overall gain over the heuristics comes mostly from the repeat slice.
  If LM2 fails (a learned model scores well on novel pairs), that refutes the audit's critique and must be
  reported prominently.

Setup (rel-hm user-item-purchase, RelBench test timestamp T = 2020-09-14, label window 7 days):
  - Test users: a random subsample of --max-users of the 67,144 test users with >= 1 target (seeded).
  - Training data for every model: interactions with t <= T in the last --window-days days, for the sampled
    test users plus --extra-users other active users (all users with --max-users 0). Nothing after T is used.
    RelBench's own GNNs instead train on the train/val task tables; this is a simpler, leakage-free setup.
  - Item attributes are not used (article table not downloaded): models use ids only.
  - Hybrids repeat+<model>: the user's past items first (as in the repeat heuristic), then the model.
  - Every model's top-12 is backfilled with recent popularity (last 7 days) when it has < 12 items; users with
    no training interactions get recent popularity (same fallback as the heuristics).
  - Slices and MAP@12 are exactly as in scripts/audit_relbench.py (repeat, cf_nbr = top-20 co-purchased in
    the last 28 days, novel, cold_user, new_item, head10). Slice MAP: users with >= 1 positive in the slice,
    targets restricted to that slice, predictions unchanged.

Usage:
  local subsample:  .venv/bin/python scripts/audit_models.py --max-users 20000 --device cpu
  full (Colab GPU): python research/scripts/audit_models.py --max-users 0 --extra-users 0 --device cuda --epochs 20

## Results: 20,000 test users (seed 0), 61,601 positives, runtime 53s on cpu

Slice shares of positives: repeat 3.8%, cf_nbr 13.5%, novel 82.6%, cold_user 10.3%, new_item 4.2%, head10 41.7%

MAP@12 × 100 (RelBench metric). Published full-test numbers for reference (arXiv 2407.20060): Global Popularity 0.3, Past Visit 0.89, LightGBM 0.38, RDL (GraphSAGE) 0.8, RDL (ID-GNN) 2.81.

| slice (users) | recent_popularity | repeat_then_recent_pop | item_cf | bpr_mf | lightgcn | repeat+item_cf | repeat+bpr_mf | repeat+lightgcn |
|---|---|---|---|---|---|---|---|---|
| **overall** (20,000) | 0.93 | 2.67 | 0.44 | 0.77 | 0.57 | 2.63 | 2.62 | 2.64 |
| repeat (1,863) | 0.93 | 42.29 | 3.39 | 8.14 | 2.87 | 42.29 | 42.29 | 42.29 |
| cf_nbr (5,374) | 2.35 | 0.08 | 0.91 | 0.54 | 1.45 | 0.16 | 0.04 | 0.10 |
| novel (18,241) | 0.60 | 0.15 | 0.10 | 0.22 | 0.26 | 0.09 | 0.11 | 0.12 |
| cold_user (2,133) | 0.70 | 0.70 | 0.70 | 0.70 | 0.70 | 0.70 | 0.70 | 0.70 |
| new_item (1,795) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| head10 (13,160) | 2.05 | 3.02 | 0.51 | 1.41 | 1.21 | 2.92 | 2.94 | 2.98 |
