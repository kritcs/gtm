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
