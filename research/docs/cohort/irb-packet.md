# Founder cohort: IRB preparation packet (draft v0.1, 2026-11-25)

*For the faculty sponsor and the Stanford IRB intake conversation. Not approved; do not recruit. Companion
documents: `study-description.md` (lay summary), `elicitation-form.md` (instrument), `analysis-plan.md`
(pre-registered analysis and power), `scripts/cohort_scrub.py` (de-identification tool).*

## 1. Classification questions to settle with the IRB

| Question | Our reading | To confirm |
|---|---|---|
| Who are the human subjects? | The founders (questionnaire about their beliefs). | — |
| Are the founders' prospects subjects? | No, as long as only **account-level business attributes** and outcomes are collected, de-identified on the founder's machine. B2C startups whose "accounts" are individual consumers are **excluded** in v1, because their logs would be data about identifiable people. | IRB |
| Risk level | Minimal risk: a questionnaire plus de-identified business records. Main risk is disclosure of commercially sensitive information. | IRB |
| Likely review category | Exempt (survey / interview procedures with adults; data recorded so that subjects are not readily identifiable, or disclosure would not reasonably place them at risk). | IRB determination |
| Deception / intervention | No deception. One small intervention: founders include ~40 randomly drawn accounts from their own target list in their outreach (the "randomized slice", `analysis-plan.md`). The founder sees the list and may skip any account. | IRB |
| Compensation | A personal calibration report; optionally a small gift card. | Sponsor and budget |

## 2. Consent form (draft text)

> **Study:** Do founders' beliefs about customers predict who buys?
> **Researcher:** Peem Puangmaha, Computer Science, Stanford University. **Faculty sponsor:** [TBD].
>
> **What we ask.** About 25 minutes once, to write down what you expect about which customer segments will
> respond to your outreach. Then, over 3–6 months, a monthly export of your outreach log, produced by a script
> we give you. The script removes names, e-mails, phone numbers and web addresses, and replaces company
> identifiers with codes only you can reverse.
>
> **One request about your outreach.** We ask you to include about 40 accounts drawn at random (by our script)
> from your own target list. You see the list first and may skip any account.
>
> **What we do not do.** We do not contact your prospects or share your answers with
> anyone outside the research team, including program staff, investors or other participants.
>
> **Risks.** Your beliefs and results are commercially sensitive. We keep them under a random ID and store the
> ID key separately and encrypted. We report only aggregates across companies, or anonymised examples you have
> approved in writing.
>
> **Benefits.** A private report comparing your expectations with your outcomes, and showing where a model
> that learns from your early results would have pointed you.
>
> **Voluntary.** You can stop at any time and ask us to delete your data, without giving a reason. Taking part
> or not has no effect on your standing in any Stanford program.
>
> **Questions.** [researcher e-mail]; IRB contact [Stanford IRB office details].
>
> ☐ I agree to take part. ☐ I agree to be contacted about a possible follow-up study.

## 3. Data management plan

- **Collected:** questionnaire answers; de-identified outreach logs (`account_key`, segment attributes,
  contact date, stage reached, stage date). Nothing else.
- **Not collected:** names or contact details of prospects; free-text notes; CRM credentials. We never access
  participants' systems; founders run `cohort_scrub.py` themselves, and it refuses to write a file if a kept
  column looks like an e-mail, phone number or URL.
- **Storage:** Stanford-approved encrypted storage. The company-to-ID key is held separately by the
  investigator. Nothing goes into the public research repository (`research/data/` is git-ignored).
- **Retention:** de-identified data for 3 years after publication; the ID key is deleted at the end of the study.
- **Sharing:** aggregate results only. Any per-company example needs written approval from that founder.
- **Freezing beliefs:** each questionnaire is hashed (SHA-256) and the hash is timestamped before the founder's
  outreach period starts, so we can show the beliefs were recorded before any outcomes.

## 4. Eligibility

Include founders of companies selling to businesses (B2B) with 0–50 customers that plan at least 60 outbound
or targeted contacts in the next 3 months, and that record a funnel stage for every contacted account.

Exclude consumer-facing companies whose accounts are individuals (v1); companies that cannot export
per-account outcomes; and anyone under 18.

## 5. Recruitment channels (to confirm with program leads)

StartX, Lean LaunchPad / ENGR 245, NSF I-Corps teams, the Stanford GSB startup garage. Recruitment e-mails go
through program staff. No cold contact of founders by the researcher.

## 6. Open items before submission

1. Faculty sponsor.
2. Decide gift card vs report-only incentive.
3. Confirm the B2C exclusion with the IRB (or design an aggregation rule that makes consumer logs non-identifiable).
4. Pilot the questionnaire with 2–3 founders outside the study (wording, time).
5. Register `analysis-plan.md` publicly (e.g. OSF) before the first questionnaire is collected.
