# Founder belief elicitation form (draft v0.2, 2026-11-25)

*For the prospective founder cohort. Completed once, before outreach outcomes are known; timestamped and
frozen. About 25 minutes. Answers are confidential and analysed only in aggregate.*

**Instructions for the facilitator:** do not show any model output, co-founder answers or benchmark rates
before the founder finishes. Ask about the stated funnel stage only. "No idea" is always an acceptable answer.

---

## Part A — About the company (2 min)

1. Company name (kept private; replaced by an ID in all analysis): ______
2. One sentence on what you sell and to whom: ______
3. Stage: ☐ pre-revenue ☐ 1–5 paying customers ☐ 6–20 ☐ 21–50 ☐ 50+
4. Today's date (auto-filled) and the date you expect to start the outreach we will track: ______

## Part B — Define the funnel stage (2 min)

5. For this study, a "success" is: ☐ a reply ☐ a booked meeting ☐ a qualified opportunity ☐ a paid pilot ☐ a paid contract.
   Pick the **earliest stage you will reliably record** for every account you contact.
6. Roughly how many accounts will you contact in the next 3 months? ______

## Part C — Segments (5 min)

7. List the **dimensions** you use to describe target customers (max 3), e.g. industry, company size, role of buyer, trigger event (just raised, hiring for X, uses tool Y).
8. For each dimension, list its values (max 6 each). The form builds main-effect segments from these.
9. Optionally name up to **5 specific combinations** you believe behave differently from the main effects (e.g. "fintech AND 50–200 employees").

## Part D — Beliefs about each segment (10 min)

For each segment, answer in this order (SHELF order — limits first, then the middle):

> *"Of 20 companies in this segment that you contact this quarter, how many will reach the success stage from Q5?"*

| Segment | Lowest plausible (out of 20) | Highest plausible | Most likely | 25%/75% range (optional) | Confidence: no idea / low / medium / high |
|---|---|---|---|---|---|
| … | | | | | |

Rules shown to the founder: answer about **observable counts**, not percentages; "no idea" is fine and useful;
don't consult anyone or any data while answering.

### Part D2 — Contact plan (2 min)

| Segment | Share of next quarter's outreach you plan to spend here (%) |
|---|---|
| … | |

Why we ask: the founder decides whom to contact, so outcomes are only observed where the founder looks. The plan
lets the analysis model that selection (it is the outreach-steering strength α in our simulator) instead of
mistaking it for a property of the market.

## Part E — Calibration questions with known answers (3 min)

10. 5–8 questions whose answers the research team knows but the founder does not see yet, e.g. your own reply
    rate last month (from your CRM), published reply-rate benchmarks for cold email in your sector, base rates
    from public data. Same format as Part D. Used only to estimate each founder's calibration.

## Part F — Consistency checks (3 min)

11. Five pairwise comparisons, randomly ordered: *"Which converts better: segment X or segment Y?"*
12. Two Part D questions repeated (rephrased) to measure answer noise.

## Part G — Context (optional)

13. Where do these beliefs come from? ☐ customer interviews (how many?) ☐ prior job in the industry ☐ advisors ☐ competitors ☐ intuition
14. Have you already contacted accounts in any of these segments? Which, and roughly how many?

---

### How answers become model inputs (for the research team; not shown to the founder)

| Form answer | Model input |
|---|---|
| Part D "most likely" count m out of 20 | belief logit = logit((m + 0.5) / 21), per segment |
| Part D lowest / highest plausible (and 25/75 if given) | stated sd on the logit scale: (logit(hi) − logit(lo)) / 3.29 for a 90% range, or the 25/75 range / 1.35 |
| Part D confidence "no idea" | "no idea" flag; belief set to the founder's overall level (PC-RFM-tok input) |
| Part C combinations (Q9) | interaction segments; all other segments are built from main effects |
| Part D2 plan | outreach steering strength, fitted to the plan (sensitivity analysis only) |
| Part E calibration questions | per-founder calibration: stated-sd scale vs observed error (secondary analysis) |
| Part F repeats | answer noise; Part F pairwise choices: ordinal consistency check |

Evidence behind this design (simulation, `results/m2_pfn_tok.md`): stated confidence is used in the right direction
(reversing it costs up to 1.9 lift), it matters most for mediocre founders with ≤ 1 label, and "no idea" answers did
*not* help. So the most likely value and the plausible range are the essential questions; "no idea" stays
allowed but is not a priority.

### Outreach log format (what the founder shares; produced by `scripts/cohort_scrub.py`)

One row per contacted account: `account_key` (salted hash; the salt stays with the founder), the segment
attributes named in Part C, `contacted_on` (date), `stage_reached` (none / reply / meeting / opportunity /
pilot / contract), `stage_date`. No names, e-mails, phone numbers, URLs or free text leave the founder's machine.

---

### What happens next (shown to the founder)
- Your answers are frozen and timestamped now.
- Over the next 3–6 months you share an export of your outreach log (account attributes, date contacted,
  stage reached), with personal data removed. We provide a script that strips names and emails.
- You receive a private report: which of your beliefs held up, your calibration, and where the model would
  have pointed you differently.
