# SBIR → USAspending pipeline (feasibility build, 2026-11-24)

Idea #1 in `evaluation-brainstorm.md`: an SBIR/STTR award abstract is a public, dated statement of what a
young firm builds and for whom, written before any sales. Later federal contracts to the same firm are real,
public outcomes. This doc records what was built, the data terms, the task definition, first statistics on a
200-firm sample, and the risks. Data files live under `research/data/` (gitignored); nothing is redistributed.

## What was built (all scripts in `research/scripts/`)

| Step | Script | Output (`research/data/interim/`) |
|---|---|---|
| Download SBIR.gov bulk award CSV; build cohort | `data_sbir_cohort.py` | `sbir_cohort.csv.gz` (5,225 firms), `sbir_contracts.csv.gz` (118,831 SBIR contract ids) |
| Query USAspending contracts after T0 per firm (cached, resumable) | `data_sbir_outcomes.py --n 200` | `sbir_outcomes.csv.gz`, `sbir_query_log.csv` |
| Per-contract detail: product/service code (PSC), solicitation id, awarding office | `data_sbir_details.py` | `sbir_outcomes_detail.csv.gz` |
| Label table + statistics | `data_sbir_stats.py` | `sbir_labels.csv.gz`, `sbir_stats.txt` |

Raw API responses are cached in `data/raw/usaspending/` (per firm) and `data/raw/usaspending_awards/` (per award).
End-to-end run for 200 firms: ~5 min of polite API calls, 0 errors.

**Sources (verified 2026-11-24)**
- SBIR.gov bulk awards: `https://data.www.sbir.gov/awarddatapublic/award_data.csv` — HTTP 200, 367,551,355 bytes,
  last-modified 2026-01-01; 207,731 awards. Columns include Company, Agency, Branch, Phase, Program, Contract,
  Proposal Award Date, Award Year, Duns, Number Employees, State, Topic Code, Abstract. **No UEI column.** The file
  also holds personal contact fields (contact/PI/RI names, emails, phones); the scripts never load them.
- USAspending search: `POST https://api.usaspending.gov/api/v2/search/spending_by_award/` with
  `recipient_search_text: [DUNS]` works (no key). `"Awarding Office"` is accepted as a field but returned null, so
  offices come from `GET https://api.usaspending.gov/api/v2/awards/<generated_internal_id>/`
  (`awarding_agency.office_agency_name`, plus `latest_transaction_contract_data.product_or_service_code`,
  `solicitation_identifier`).

## Terms of use

- **SBIR.gov:** the site's `/data-resources` page states no licence; `/about/policies` is the SBIR program policy
  directive, not website terms. SBIR.gov is run by SBA, and SBA's website policy
  (https://www.sba.gov/about-sba/open-government/about-sbagov-website, read 2026-11-24) says: "Government
  information on SBA.gov is in the public domain … it is requested that in any subsequent use, SBA's information be
  given appropriate acknowledgement", and that material contributed by private companies may be copyrighted.
  **Abstracts are written by the firms**, so: use them for analysis, publish derived results only, do not
  redistribute abstract text, acknowledge SBA/SBIR.gov. Not legal advice; confirm with Stanford before release.
- **USAspending:** public domain / unrestricted per `datasets-real-outcomes.md` (repo CC0; GAO). Its Terms page
  was not opened in that check.

## Cohort

Firms (keyed by DUNS, 9 digits) whose **first** SBIR/STTR award of any phase is a **Phase I dated 2012–2018**.
"First" excludes SBIR mills by construction. Because ~40% of all awards lack DUNS (mostly pre-2008 and
2023+), a firm whose earlier award lacks DUNS would look new; a name check (normalized company name with an
earlier award) drops 320 such firms. Result: **5,225 firms**, abstracts present for 100%.
By first-award agency: HHS 1,932 · NSF 1,462 · DoD 1,131 · DOE 416 · USDA 202 · NASA 177 · others ≤ 59
(counts before the name check).

## Outcome definition

For firm f with first award date T0, query its contracts (award types A–D) with activity in [T0, T0 + 5 y],
keep those starting on/after T0 whose recipient name shares a token with the SBIR company name (drops ~5% of
returned contracts as possible wrong-firm matches), then classify:
- **SBIR/STTR:** award id matches an SBIR.gov contract number, or the solicitation id contains SBIR/STTR, or the
  legacy description mentions it. *SBIR.gov's list is incomplete* — verified: NASA `NNX16CC57P` is an STTR Phase I
  (solicitation `NNX16STTRPHI`) absent from SBIR.gov for that firm; the solicitation check catches it.
- **R&D:** PSC starting with "A" or PSC description starting "R&D" (catches SBIR-like and BAA research awards).
- **Purchase (primary outcome):** neither SBIR nor R&D — the government bought a product or service.

## Task definition (proposed)

- **Unit:** one firm = one task, at time T0.
- **Candidate "accounts":** federal awarding sub-agencies (≈100 in total; e.g. Department of the Navy, NASA, CDC)
  — or offices (finer; from the detail endpoint) as a second granularity.
- **Positive:** the sub-agency awards f a *purchase* contract within 5 years of T0.
- **Founder belief:** what the abstract says about intended users/markets, turned into segment beliefs either by
  (a) keyword rules mapping mission terms to agencies, or (b) an LLM whose training cutoff precedes T0
  (ChronoGPT-style), never a modern LLM on named firms. The funding agency is a known fact at T0 (it is the SBIR
  customer), so it is given to every method as a feature — not counted as foresight.
- **Segments:** agency × office type, or agency × NAICS/PSC sector of the purchase.
- **Few-label regime:** each firm has 0–4 positives, so in-firm labels are almost nil; this is an **n ≈ 0 task with
  trust pooled across firms** (calibrate how much abstracts of this kind, from this agency, can be trusted).

## First statistics (seeded random sample of 200 cohort firms)

| | |
|---|---|
| Firms with ≥ 1 contract on/after T0 (name-matched) | 63 / 200 (31.5%) |
| … ≥ 1 non-SBIR contract | 30 / 200 (15.0%) |
| … ≥ 1 **purchase** (non-R&D, non-SBIR) | **22 / 200 (11.0%)** |
| Contract mix (384 name-matched) | 40% R&D, 32% SBIR-flagged, 60% purchase-classified* |
| Purchase labels (firm × sub-agency) | 35; 13 distinct sub-agencies, 28 distinct offices |
| Sub-agencies per buying firm | median 1, max 4 |
| Median days T0 → first purchase | 157 |
| First purchase from the **SBIR funding agency** | **54.5%** of buying firms |
| First purchase from the funding sub-agency (heuristic branch match) | 31.8% |
| Buying rate by first-award agency | DoD 29% (n=35) · USDA 38% (n=8) · DOE 10% · NASA 14% · NSF 5% (n=55) · HHS 4% (n=77) |

\*Categories overlap in the raw counts (an award can be R&D and SBIR); "purchase" = neither.
Top first-purchase sub-agencies: Navy 6, NASA 5, Army 3, then single counts (NIH, Coast Guard, Air Force, VA, CDC,
APHIS, NIST, DLA).

**Reading:** outcomes are real but sparse. Over half of first purchases come from the agency that funded the SBIR —
the "memorizable" answer a model gets from the funding label alone. The interesting slice for the paper is the
other ~45%: first sales to an agency the firm had no award from. Scaling to the full cohort (~1.5 h of API calls)
should give roughly 575 buying firms and ~900 purchase labels, of which ~400 are "new agency" sales.

## Risks

- **Government ≠ commercial market** (reviewers will say so); procurement rules, set-asides and incumbency matter.
- **Sparse labels:** ~1 positive per buying firm; most firms have none. Evaluate on buying firms or with ranking
  metrics that tolerate single positives (hit@k, MRR), and report the non-buying majority separately.
- **DUNS-only join:** SBIR.gov has no UEI; DUNS was retired in 2022, so post-2022 outcomes may be missed for firms
  whose USAspending records carry only UEI. Cohort T0 ≤ 2018 with 5-year windows ends ≤ 2023, limiting this.
- **Incomplete SBIR flags** (see above) — residual SBIR/R&D awards misclassified as purchases are possible;
  spot-check a sample by hand.
- **Leakage through text:** abstracts often name the funding agency/mission ("for the Navy"); that is legitimate
  T0 information but must be treated as a known label, not a learned prior. Modern LLMs may recognise firms — redact
  names and use pre-T0 models.
- **Recipient search is text search:** DUNS could match other fields; the name-token check removes ~5%.
- **Selection:** firms that win SBIR are already filtered; "no purchase" may mean the firm never sought federal sales.

## Next steps

1. Run outcomes + details for the full 5,225-firm cohort (resumable; ~1.5 h).
2. Hand-check ~50 "purchase" contracts for residual SBIR/R&D.
3. Build the belief extractor: keyword map (abstract terms → agencies/missions) first; then a pre-T0 LLM.
4. Define the candidate set (all sub-agencies with ≥ N purchases in the cohort) and the evaluation split
   (by T0 year), then run CPR-LP / PC-RFM with the funding agency as a given feature.

## Label hand-check and evaluation results (2026-11-25, Claude)

Hand-check of 50 "purchase" records (random sample, `random_state=0`, from 9,320 purchase-flagged rows of
`sbir_outcomes_detail.csv.gz`; judged from PSC description, NAICS, award type, amount):
- **0 / 50 are R&D or SBIR** in disguise. 41 / 50 are clearly products or operational services (microcircuits,
  lab supplies, test kits, motors, ship repair, building repair, IT).
- 5 / 50 are professional support services (PSC R4xx/R6xx: "operations research", "program management",
  "other professional support"). These are purchases, but could be study-like work; in the whole set R-codes are
  9.8% of purchases and B-codes ("special studies") 1.9%. Upper bound on residual research-like labels ≈ 12%.
- 4 / 50 have $0 obligated (order modifications / IDV orders). They still mark a buyer relationship.
- All 50 are name-matched to the firm. Some look odd for an SBIR firm (food-service equipment, large-calibre gun
  parts) — likely real (manufacturers sell outside their SBIR topic) but name collisions cannot be ruled out from
  these fields.

Evaluation: `results/sbir_eval.md` (pre-registered SB1–SB3, scored). Keyword "founder" beliefs add nothing beyond
funding agency + popularity; on purchases from new agencies (56% of test purchases) popularity is best.
Simulator placement: `results/sim_calibration.md` (base rate ~9× below the PC-RFM training range over all firms).
