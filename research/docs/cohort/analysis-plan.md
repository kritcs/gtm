# Founder cohort: analysis plan and power (draft v0.1, 2026-11-25)

*To be registered publicly (e.g. OSF) before the first questionnaire is collected. Power numbers come from
`scripts/cohort_power.py` → `results/cohort_power.md` (simulation; assumptions listed below).*

## Design change recommended by the power analysis

The v0.1 study description planned to evaluate methods on the accounts each founder contacts **after** the first
k outcomes. Because founders choose whom to contact, that evaluation set is concentrated in the segments the
founder already likes. Ranking within it hides most of the difference between methods. In simulation:
- the company-level gain of CPR-LP over labels only, at k = 10, is +0.020 AUC on founder-chosen later contacts,
  against +0.082 on all accounts (unobservable in the cohort);
- with 30 companies, the chance of detecting it is about 15%.

**Recommendation: add a randomized slice.** Each founder also contacts 40 accounts drawn uniformly at random from
their own target list, mixed into normal outreach. Methods are scored on that slice. In simulation this raises the
measured gain to +0.099 AUC and the power at k = 10 to:

| companies | 15 | 20 | 30 | 50 |
|---|---|---|---|---|
| founder-chosen later contacts (v0.1 design) | 0.14 | 0.15 | 0.15 | 0.17 |
| **randomized slice of 40 (recommended)** | 0.34 | 0.46 | 0.56 | 0.76 |

So the target is **40–50 companies** with a 40-account randomized slice each. With 15–30 companies the study is
exploratory. Possible further gains, not yet simulated: a larger slice (60–80 accounts), and a likelihood-based
score instead of AUC.

The randomized slice costs the founder about 40 contacts outside their preferred segments, i.e. 13–67% of the
60–300 contacts planned. It is also a benefit we can offer participants: it is the exploration that catches wrong
beliefs (H4 in our simulations: the greedy founder never corrects a wrong belief about a segment it avoids).

## Hypotheses (pre-registered)

All comparisons are paired within company. The unit of analysis is the company. Intervals come from a bootstrap
over companies (10,000 resamples).

- **C1 (primary):** CPR-LP using the frozen beliefs and the first k = 10 outcomes ranks the randomized slice better
  than CPR-LP without beliefs (labels only). Mean company-level ΔAUC > 0, one-sided 95% bootstrap interval.
- **C2:** the same comparison at k = 5 and k = 20 (secondary; reported regardless of C1).
- **C3 (secondary, expected to be underpowered):** CPR-LP beats the founder's beliefs alone at k = 20. Simulation:
  +0.004 AUC, power ≤ 0.11 even at 50 companies. Reported as an estimate with its interval, not as a test.
- **C4 (learned trust tracks accuracy):** across companies, CPR-LP's fitted trust in the founder at k = 20 correlates
  positively (Spearman) with the accuracy of the founder's beliefs, measured as the rank correlation between stated
  segment rates and rates observed on the randomized slice.
- **C5 (PC-RFM):** PC-RFM (the version current at registration; checkpoint hash recorded) is non-inferior to CPR-LP on
  C1's metric (ΔAUC > −0.02).
- **C6 (stated confidence):** among founders who give plausible ranges, the PC-RFM founder-token model's gain from
  their stated confidence (stated vs neutral sd) is > 0. Exploratory.

## Data handling rules (fixed now)

- **Beliefs** are frozen and hashed before outreach (`irb-packet.md` §3).
- **"Success"** is the stage the founder chose in Q5. Accounts with no recorded stage after 30 days count as
  "none".
- **Exclusions:** companies with < 3 successes or < 3 non-successes on the randomized slice (AUC undefined) are
  excluded from C1. The count of excluded companies is reported, along with a sensitivity analysis that scores
  them with log-loss instead.
- **Ordering:** the first k outcomes are the first k contacts by date, including randomized-slice contacts made
  in that window, which are then dropped from the evaluation slice.
- **Skipped slice accounts:** founders may skip randomized-slice accounts. Skips and stated reasons are recorded; if
  more than 20% of a company's slice is skipped, that company is reported separately (the slice is no longer random).
- **Segments** come only from the attributes declared on the form. No attribute is added after outcomes are seen.

## Simulation assumptions behind the power numbers

`fprior.sim` markets with 1,500 accounts and base rate U(0.03, 0.15). Founder reliability ρ ~ U(−0.25, 1).
Founder-steered outreach (α = 2, 10% exploration) for 60–300 contacts. CPR-LP without a graph. Real founders may be
less reliable than this, and real segment effects may be weaker: Bank and Hillstrom segment sd is 0.36–0.47, inside
but at the low end of the simulator's range. Both would lower power. This plan is revised (and the revision
registered) if a pilot with 3–5 companies shows effect sizes or variances far from these.
