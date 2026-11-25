# Where real data sits in simulator knob space (PLAN.md §2d)

## UCI Bank Marketing (job group × age band, 32 segments)

| Property | Bank | PC-RFM v0 training range |
|---|---|---|
| Base rate | 0.113 | 0.02–0.20 (log-uniform) |
| Segment-effect sd (logit, account-weighted) | 0.47 (split B); 0.46 signal sd from the A/B covariance | logit_sd·√w_seg ∈ [0.18, 2.24] |
| Share of segment variance from interactions | 0.18 | default recipe ≈ 0.21; `--random-seg` covers 0–1 |

Bank is **in-distribution** on all three measured axes, which is consistent with PC-RFM transferring well
(`real_bank.md`) — and means Bank is not a hard test of the sim-to-real gap. Not measurable from these data:
the individual-level (within-segment) variance and the founder population. SBIR (segments = federal
agencies, sparse labels) is the harder test; add its row when `sbir_eval.py` runs.

Computed with an inline script on 2026-11-24 from `fprior.real_bank` (`experience_logits` on splits A and B).

## SBIR first federal purchases (firm × awarding sub-agency, 81 sub-agencies) — added 2026-11-25

| Property | SBIR | PC-RFM v0 training range |
|---|---|---|
| Base rate per (firm, sub-agency), all 5,225 firms | 0.0023 | 0.02–0.20 → **below range (~9×)** |
| Base rate per (firm, sub-agency), 511 buying firms only | 0.024 | inside, at the bottom |
| Sub-agency effect sd (logit of purchase rate, 81 sub-agencies with ≥ 1 purchase) | 1.35 (1.33 purchase-weighted); lower bound, zero-purchase sub-agencies excluded | logit_sd·√w_seg ∈ [0.18, 2.24] → inside, upper half |
| Concentration of first purchases (top 1 / 5 / 10 sub-agencies) | 16% / 55% / 77% | not a training knob |
| Positive labels per buying firm | 1.9 (median 1, max 13) | n = 0–50 contacts with outcomes — SBIR gives ~1–2 positives and no observed negatives |

SBIR is **out of distribution** on the base rate (unless one conditions on firms that buy at all, as
`sbir_eval.py` does) and on label structure (positive-only, ~2 per firm, no contacted-but-declined accounts).
Segment effects are strong and inside the range. This is why SBIR was evaluated as a ranking task with
`sbir_eval.py` rather than with PC-RFM directly. Computed inline from `data/interim/sbir_labels.csv.gz`
(outcome = purchase, distinct firm × sub-agency).
