# Study description (draft v0.1): Do founders' beliefs about customers predict who buys?

**Investigator:** Peem Puangmaha, Stanford University (Computer Science). *Faculty sponsor: TBD.*
**Status:** draft for discussion with a faculty sponsor and the Stanford IRB. Not yet approved; do not recruit.

## Purpose

Early-stage companies must decide whom to sell to before they have enough customers for data-driven tools
to work. Founders hold beliefs about which customer segments will buy, which are often informative and
sometimes wrong. We are studying a method that treats those beliefs as a starting point and learns, from
the company's first outcomes, how much to trust them. This study tests that method on real startups by
recording founders' beliefs **before** outcomes are known and comparing them with what actually happens.

## Who can take part

Founders of early-stage companies (pre-revenue to ~50 customers) who will do outbound or targeted outreach
in the next 3–6 months, for example through Stanford programs (StartX, Lean LaunchPad / ENGR 245, NSF
I-Corps teams). Target: 15–30 companies.

## What participants do

1. **Once, about 25 minutes:** complete a belief questionnaire about customer segments (see
   `elicitation-form.md`). Answers are timestamped and frozen.
2. **Over 3–6 months:** share an outreach log export (account attributes such as industry and size, contact
   date, stage reached), with names, emails and other personal data removed by a provided script.
3. **At the end:** receive a private report on their own beliefs and outcomes.

No outreach is changed or directed by the study (observational). An optional later arm may compare a
model-suggested list with the founder's own list on a blinded basis; that would be a separate protocol.

## Data and privacy

- Company identities are replaced by random IDs; the key is stored separately and encrypted.
- No personal data about the founders' prospects is collected — only account-level attributes and outcomes.
- Data is used for research only, reported in aggregate, and not shared outside the research team.
  Participants may withdraw at any time and have their data deleted.

## Risks and benefits

Risks are minimal: time (~25 min plus exports) and the confidentiality of commercial information, mitigated
as above. Participants benefit from a structured review of their go-to-market assumptions and a personal
calibration report.

## Analysis (pre-registered before any outcome data is seen)

- Primary: does a model that combines founder beliefs with early outcomes rank later outcomes better than
  (a) outcomes alone and (b) the founder's beliefs alone? Metrics: ranking quality on accounts contacted
  after the first k outcomes, with k = 5, 10, 20.
- Secondary: founder calibration (stated vs observed rates), and whether learned trust tracks it.
- Hypotheses and analysis code are registered and timestamped before data collection ends.

## Open items

Faculty sponsor; IRB determination (likely minimal-risk / exempt category, to confirm); data-use agreement
template; incentive (e.g., the personal report, or a small gift card); recruitment channels.
