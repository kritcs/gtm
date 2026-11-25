H4 with segment-level Thompson sampling (the textbook version), vs our account-level sampler.

Pre-registered (committed before running; from docs/theory-h4.md P2a/P2b):
  TS1. Segment-level Thompson with η = 0 (Gaussian prior) contacts the hidden best segment in at most 2 of
       10 seeds within 40 rounds, and its error at round 39 stays below −2.0 logits (stuck, as P2a predicts),
       unlike account-level Thompson (5/10 washed out, error −1.24, results/explore_h4.md).
  TS2. Adding stratified η = 0.1 and the Student-t prior to segment-level Thompson gets the hidden segment
       contacted in more seeds than η = 0 Gaussian.
Same markets, founder and seeds as scripts/explore_h4.py.

## Results (10 seeds, 40 rounds)

| policy | seeds that ever contact hidden segment | hidden contacts by r40 | error r39 | washed out (top 3) |
|---|---|---|---|---|
| thompsonseg η=0.0 | 5/10 | 21.7 ± 14.0 | -1.96 ± 0.27 | 2/10 |
| thompsonseg η=0.1 | 10/10 | 19.7 ± 8.9 | -1.87 ± 0.27 | 2/10 |
| thompsonseg-robust η=0.0 | 5/10 | 23.0 ± 13.9 | -1.90 ± 0.31 | 2/10 |
| thompsonseg-robust η=0.1 | 10/10 | 21.4 ± 9.0 | -1.74 ± 0.39 | 3/10 |
## Scoring (written after the run)

- **TS1 fails.** Segment-level Thompson (η = 0, Gaussian) reached the hidden segment in 5/10 seeds (predicted ≤ 2)
  and its round-39 error is −1.96 ± 0.27 (predicted < −2.0). Theory P2a's escape time (≈ 4,900 rounds) assumes one
  segment chosen per round against the best competitor; here each round contacts 25 accounts spread over several
  segments, so the hidden segment only needs to beat roughly the 3rd–5th segment. Batched selection is itself a
  source of exploration that P2a does not model.
- **TS2 holds.** With η = 0.1 every seed reaches the hidden segment (10/10 vs 5/10).
- Within 40 rounds no variant corrects much (−1.7 to −2.0). The earlier caveat that our account-level sampler escapes
  "accidentally" is only half right: batching matters too. `docs/theory-h4.md` P2a needs a batched version.
