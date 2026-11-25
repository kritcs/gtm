# Candidate real-outcome datasets (verification report)

Date checked: 2026-11-24. Tags: [P] = verified from a primary source page fetched in this session; [S] = secondary source (mirror, docs site, search result); [M] = from model memory, NOT verified this session. Only metadata, docs and live API calls with limit<=3 were used; nothing large was downloaded.

## Summary table

| # | Dataset | Fit (1-5) | Licence OK for research + derived results? | Randomized exposure? | Key caveat |
|---|---|---|---|---|---|
| 1 | Hillstrom MineThatData 2008 | 4 | Unclear: no licence stated on the primary page | Yes (1/3 each arm) | B2C retail email; 0.9% conversion; only 8 coarse fields |
| 2a | UCI Bank Marketing, full/additional | 4 | Yes, CC BY 4.0 | No (targeted telemarketing) | Selection-biased contact; `duration` leaks the outcome |
| 3 | Open Bandit Dataset (ZOZOTOWN) | 1 | Dataset licence not stated on pages opened; code Apache-2.0 | Yes (uniform-random policy logs) | Features hashed and anonymized, no human-readable segments; outcome is a click |
| 4 | Criteo Uplift v2.1 | 2 | Non-commercial, share-alike (CC BY-NC-SA 4.0) | Yes (RCT treatment) | 12 anonymized float features, no interpretable segments |
| 5 | X Education Lead Scoring (Kaggle) | 3 | Unknown: Kaggle page could not be opened | No | B2B-ish leads but not randomized, selection-biased, provenance and licence unverified |
| 6a | UCI COIL 2000 (insurance mailing) | 3 | UCI says CC BY 4.0; older source says non-commercial research only | Not documented | 5.8k rows; zip-level demographics, 6% base rate; contact process unclear |
| 6b | KDD Cup 1998 (charity mailing) | 3 | "General use with restrictions" (must notify sponsors contacts) | Not documented | Mailing to lapsed donors; strict notification terms; large files |
| 6c | X5 RetailHero uplift | 2 | No licence stated | Treatment ratio 0.5; assignment mechanism not described | Retail loyalty-card data, fields in separate large tables |
| 7 | USAspending.gov API | n/a as outcome set; 4 as B2B firmographic source | Yes (CC0 repo; GAO says unrestricted data) | n/a (not outreach) | No contact/outreach events; "first award" is a proxy for a conversion event only |

---

## 1. Hillstrom MineThatData E-Mail Analytics and Data Mining Challenge (2008)

- Official page: https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html [P]
- Citation: no formal paper. Cite the blog post / Kevin Hillstrom, MineThatData, 2008. The scikit-uplift docs list it as "Kevin Hillstrom Dataset: MineThatData" [S] (https://www.uplift-modeling.com/en/latest/api/datasets/fetch_hillstrom.html).
- Licence: the primary page states NO licence or terms of use [P]. Treat as unlicensed-but-public; widely redistributed via scikit-uplift. For a research report this is low risk, but redistribution of the raw file should be avoided (link instead). Contact the author if redistribution matters. No no-ML clauses seen.
- Size/unit/time: 64,000 customers (one row per customer), who last purchased within the previous 12 months [P]. Row count 64,000 confirmed by [S]. Outcomes measured over two weeks after the email. Calendar dates of the campaign are not given on the page [P].
- Randomization: yes, about 1/3 Mens E-Mail, 1/3 Womens E-Mail, 1/3 No E-Mail [P]. Propensities not logged but known by design (about 1/3 each). Non-converters/true negatives included (conversion rate about 0.9%, visit about 15% [S]).
- Segment fields: Recency (months since last purchase), History_Segment (spend band), History (dollars past year), Mens, Womens (1/0 bought that category), Zip_Code (Urban/Suburban/Rural), Newbie, Channel (phone/web/multichannel) [P]. Treatment: Segment. Outcomes: Visit, Conversion, Spend [P].
- Download: direct URL http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv [P]; or `sklift.datasets.fetch_hillstrom` (pip scikit-uplift) [S]. No login.
- Fit 4/5: randomized, true negatives, human-readable segments (recency, spend band, region type, channel) map naturally to founder beliefs. Caveat: B2C retail, tiny positive rate (about 580 converters), few features; licence silent.

## 2. UCI Bank Marketing (Moro, Rita, Cortez)

- Official page: https://archive.ics.uci.edu/dataset/222/bank+marketing (ID 222) [P]. DOI 10.24432/C5K306 [P].
- Citation (UCI-given): Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI ML Repository. https://doi.org/10.24432/C5K306 [P]. Underlying paper: Moro, Cortez, Rita (2014), "A data-driven approach to predict the success of bank telemarketing", Decision Support Systems; DOI https://doi.org/10.1016/j.dss.2014.03.001 (resolves to Elsevier pii S016792361400061X; article text itself NOT opened, title/venue from memory [M]).
- Licence: CC BY 4.0 [P]. Fine for research and derived results with attribution.
- Versions on the page [P]:
  - bank-additional-full.csv: 41,188 rows, 20 inputs, ordered by date, May 2008 to November 2010.
  - bank-additional.csv: random 10% sample, 4,119 rows, 20 inputs.
  - bank-full.csv: 45,211 rows, 17 inputs (older version); bank.csv: 10% sample of it.
- Unit: one phone contact record per client (Portuguese bank telemarketing; the bank description is from the paper [M]). Calls are targeted, not randomized; no propensities. Contains both yes and no (true negatives among contacted). Base rate about 11-12% yes [M, standard figure, not on fetched page].
- Fields (from [P] variable list for the 17-input version): age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome; target y (term deposit yes/no). The "additional" version adds socio-economic context columns (e.g. emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed) and day_of_week [M for column names; page confirms 20 inputs and a day_of_week entry].
- Download: UCI zip https://archive.ics.uci.edu/static/public/222/bank+marketing.zip (about 1 MB) [P]; `pip install ucimlrepo` then `fetch_ucirepo(id=222)` [P]. No login. Note ucimlrepo may return only one version; the zip contains all four.
- Fit 4/5 for simulating segment beliefs (job, age band, education, marital, housing loan are ideal segments). Caveat: not randomized (selection bias); `duration` is only known after the call and must be dropped; "additional" version time-ordered, so use temporal splits; B2C.

## 3. Open Bandit Dataset (Saito et al., ZOZOTOWN)

- Official page: https://research.zozo.com/data.html [P]; code/small sample: https://github.com/st-tech/zr-obp [P].
- Paper: Saito, Aihara, Matsutani, Narita, "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation", NeurIPS 2021 Datasets and Benchmarks. https://arxiv.org/abs/2008.07146 [P].
- Licence: dataset page states no licence [P]; GitHub README states Apache-2.0 for the code (pipeline) and says "download the full size version for research uses" [P]. Dataset licence unverified; I did not find an explicit CC licence on either page.
- Size: about 26 million rows, one per user impression [P]; small sample in repo has 10,000 rows per campaign x policy [P]. Time span: README says a 7-day experiment; exact dates not on pages [P].
- Randomization: yes. Logs from a Uniform Random policy and a Bernoulli Thompson Sampling policy, per campaign (All/Men/Women); propensity logged (`propensity_score` on research.zozo.com / `action_prob` in the repo README, same concept) [P]. Click=0 rows are included.
- Fields: timestamp, item_id, position (1-3), click (outcome), propensity, user features 0-4 and user-item affinity; item_context.csv with item features 0-3. Features are anonymized by hashing [P]. No human-readable segments.
- Download: button on research.zozo.com/data.html (open_bandit_dataset.zip) [P]; small sample via `pip install obp` / repo `obd/` folder [P]. No login mentioned.
- Fit 1/5: gold-standard for off-policy evaluation but wrong shape: users are anonymous, no interpretable segments, outcome is a click on a fashion banner, and the "action" is an item, not a decision whom to contact. Useful only as an OPE-methods sanity check.

## 4. Criteo Uplift Modeling Dataset (Diemert et al.)

- Official page: https://ailab.criteo.com/criteo-uplift-prediction-dataset/ [P].
- Paper: Diemert, Betlei, Renaudin, Amini, Gregoir, Rahier, "A Large Scale Benchmark for Individual Treatment Effect Prediction and Uplift Modeling" (AdKDD & TargetAd, KDD 2018 per the page; arXiv version title as stated). https://arxiv.org/abs/2111.10106 [P].
- Licence: CC BY-NC-SA 4.0; non-commercial; derivatives must be shared under the same/compatible licence; attribution required; provided as-is [P]. Research OK; the share-alike clause would apply to redistributed derived data (not to our aggregate results, but check before releasing derived datasets).
- Size: v2.1 (recommended "unbiased" release): 13,979,592 rows (297 MB compressed); original release 25,309,483 rows (459 MB) had a leak and is not recommended [P]. Unit: user (ad exposure experiment). Time span: not stated on page [P] (paper says two weeks per memory [M]). Non-uniformly subsampled so true incrementality is hidden [P].
- Randomization: yes, RCT (treatment 1/0) [P]. Propensity not logged as a column; treatment assignment is randomized with a fixed ratio (about 85% treated per paper [M]). Non-converters included (very low conversion rate).
- Fields: f0-f11 dense floats (anonymized, randomly projected), treatment, conversion, visit, exposure [P]. No human-readable segments.
- Download: http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz [P] (about 297 MB, NOT downloaded); `sklift.datasets.fetch_criteo` [S]. No login.
- Fit 2/5: great randomized benchmark, but anonymized features cannot support founder-style segment beliefs. Caveat: NC-SA licence and display-ad context; huge.

## 5. X Education "Lead Scoring" (Kaggle)

- Primary page: Kaggle "Leads Dataset" https://www.kaggle.com/datasets/ashydv/leads-dataset: fetch returned only a title, so UNVERIFIED (licence, row count, uploader unconfirmed from primary). The Hugging Face mirror card names the original as Kaggle "Lead Scoring Dataset" by amritachatterjee09, so there are at least two Kaggle uploads of this data (and a third, pramod0308/lead-scoring-dataset, appeared in a search result) [S]. Provenance: the data comes from an upGrad/IIIT-B case study, not an academic paper [M]. No paper to cite.
- Licence: unknown. HF mirror (https://huggingface.co/datasets/shawhin/lead-scoring-x) states no licence and defers to Kaggle [S]. Do not redistribute until the Kaggle licence is checked manually.
- Size: 9,240 leads per a study found by search [S] (one row per lead); about 30% converted [S]. HF mirror is a processed 5,688-row subset (seven columns only) [S]. Time span: not stated anywhere I could open. Not randomized; no propensities; non-converters included.
- Fields reported in [S]: Lead Origin, Lead Source, Last Activity, Total Visits, Total Time Spent on Website, plus Tags, current occupation, Converted (target). Others (specialization, country, city, etc.) recalled from memory [M].
- Download: Kaggle (login required to download); HF mirror `datasets.load_dataset("shawhin/lead-scoring-x")` needs no login [S].
- Fit 3/5: the closest to "company contacts lead, lead converts" with readable categories (origin, source, occupation, specialization). Caveats: an education-course seller (B2C leads despite "lead scoring"), leads already engaged (e.g. Tags and Last Activity are post-contact and leak), 30% rate with unknown selection, licence/provenance uncertain, possibly semi-synthetic [M, unverified].

## 6. Other datasets found

### 6a. UCI CoIL 2000 / TIC Insurance Company Benchmark
- Page: https://archive.ics.uci.edu/dataset/125/insurance+company+benchmark+coil+2000 [P]. DOI 10.24432/C5630S [P]. Citation: van der Putten, P. (2000), Insurance Company Benchmark (COIL 2000), UCI; originally P. van der Putten and M. van Someren, CoIL Challenge 2000, Sentient Machine Research [P/S].
- Licence: UCI page says CC BY 4.0 [P]; a search snippet of the older TIC page says non-commercial research/education only [S, original page not opened]. Research use fits both.
- Size: 5,822 training customers + 4,000 evaluation (labels released separately) [P]; 86 attributes: 1-43 sociodemographic (zip-code-level, so shared by customers in one zip), 44-86 product ownership; target CARAVAN (caravan policy) [P]. Time span not given. Not randomized/no propensities as far as documented; true negatives included.
- Download: UCI zip / `fetch_ucirepo(id=125)`; no login.
- Fit 3/5: B2C insurance cross-sell, readable segments (income, education, religion codes, car/fire policies owned). Caveat: not outreach-with-exposure; target is ownership, not response to a contact; small.

### 6b. KDD Cup 1998 (veterans charity direct mail)
- Page: https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html [P]. Competition paper not located.
- Licence: "available for general use with restrictions": users must notify the named sponsor contacts and send a summary/copy of results and papers; for teaching, do not name the sponsor [P]. This is a notification obligation; check with the team before publishing.
- Size: learning and validation sets about 117 MB each uncompressed; row counts and field list not on the fetched page (they are in cup98dic.txt, not opened) [P]. From memory about 95k learning rows, 480+ fields, targets TARGET_B (responded) and TARGET_D (donation $) [M, unverified here]. Mailing to past donors; treatment randomization not documented.
- Download: zip files from the page, no login; large, not downloaded.
- Fit 3/5: real mailing with response and rich history (recency, frequency, giving amounts). Caveat: B2C charity, restrictive notification terms, selection of mailed population, old.

### 6c. X5 RetailHero uplift (ODS.ai)
- Source: https://ods.ai/competitions/x5-retailhero-uplift-modeling/data (not opened directly) and scikit-uplift doc https://www.uplift-modeling.com/en/latest/api/datasets/fetch_x5.html [S].
- Size: uplift_train 200,039 clients, clients.csv about 400,162, purchases about 45.8M rows; treatment ratio 0.5; target is a repeat purchase (response ratio 0.62) [S]. Randomization of assignment: not described [S]. Licence: none stated [S]. Citation: competition link only [S].
- Download: `sklift.datasets.fetch_x5` downloads the large files [S]; not downloaded.
- Fit 2/5: SMS communication with before/after purchases, but 62% response rate, loyalty-card retail, unclear licence and randomization. Listed for completeness.

Also seen but not verified: scikit-uplift lists Lenta and MegaFon uplift datasets [S]; I did not open their pages. No real B2B outreach dataset (cold email/sales sequence with outcomes) was found that I could verify; the "B2B" Kaggle lead datasets turned up in search results are unverified and a "LeadForge" dataset found is self-described as synthetic/mock, so do not use. Real B2B outreach outcomes appear to be absent from the public domain; consider the USAspending route below or constructing synthetic-from-real-firmographics.

## 7. USAspending.gov API (live-tested 2026-11-24, no API key)

- Base: https://api.usaspending.gov ; endpoint index https://api.usaspending.gov/docs/endpoints [P]; contract docs at https://github.com/fedspendingtransparency/usaspending-api (api_contracts/contracts/v2/...) [P].
- Licence/terms: repo README links a CC0-1.0 licence [P] (that is the code repo licence). Data statement: the API data is "provided to the public as part of the DATA Act" [P]; GAO-19-72 (2018) says Treasury makes all data available free and unrestricted [S, via search snippet, not opened]. I could not open the site's own Terms of Use page (usaspending.gov/about returned no usable content; docs.usaspending.gov did not resolve). Treat as public/free federal data; recipient names are businesses, but some recipients may be individuals (sole proprietors), so avoid publishing individual-level rows. No ML/benchmark clauses found.
- Endpoints verified:
  - POST /api/v2/search/spending_by_award/ [P + live call OK]. Filters: award_type_codes (required; A,B,C,D for contracts), time_period (start_date, end_date, date_type), recipient_type_names (e.g. small_business, category_business, sole_proprietorship), naics_codes (list or {require, exclude}), recipient_search_text (name, UEI or DUNS) [P]. Fields returned in live call: Award ID, Recipient Name, Recipient UEI, Start Date, NAICS {code, description}, Award Amount, Recipient Location (state, county, zip, city) [live]. Sort and order are supported; limit/page paginate; default limit 10 [P]. "Business Categories" field returned null on the one live row checked, so do not rely on it per award.
  - POST /api/v2/recipient/ (list recipients with amount, UEI, DUNS; live call returned total 18,322,490 recipients) [live]; GET /api/v2/recipient/<recipient_id>/ returns name, UEI, DUNS, parent, business_types array (e.g. ["government","national_government",...]), location, total_transaction_amount and total_transactions, with year = fiscal year / all / latest [P + live]. This is the source for business-type flags (including small-business categories for businesses; not checked on a small business record).
  - POST /api/v2/search/new_awards_over_time/, spending_over_time, spending_by_category/{recipient,naics}, GET /api/v2/references/naics/, POST /api/v2/autocomplete/{recipient,naics}/ [P, listed on endpoints page, not all called].
- Search date floor: the API returns a message that searches are limited to start dates on/after 2007-10-01; earlier data (back to 2000-10-01) requires the Custom Award Download or bulk_download API endpoints [live].
- FIRST award date for a vendor (recipe verified live): POST spending_by_award with filters.recipient_search_text=["<UEI>"], award_type_codes=["A","B","C","D"], time_period=[{start_date:"2007-10-01", end_date:<today>}], fields incl. "Start Date", sort="Start Date", order="asc", limit=1. Test on UEI ZJK8PC78NL96 returned Start Date 2008-09-05. Caveats: (a) left-censored at 2007-10-01 unless using bulk downloads, so "first award" is really "first since 2007-10"; (b) check `date_type` semantics (action_date vs date_signed) and use all award type groups (grants/IDVs need other award_type_codes; groups can't be mixed in one call [M]); (c) recipient_search_text is a text search, so verify returned UEI equals the requested one; (d) UEI replaced DUNS in 2022, so older awards may be keyed by DUNS (recipient endpoint carries both [P]).
- Fit: not an outreach dataset (no contacts, no non-converters unless one defines a population of registered entities, e.g. from SAM.gov, which I did not verify). Usable as a B2B outcome proxy: "which small businesses win their first federal contract in year T" with NAICS, state, business-type as segments. Score 4 as a segment-rich real B2B outcome source, but outcome is not caused by our exposure and there is no randomization.

## Recommendation (short)
1. Hillstrom (randomized, segment-friendly) and UCI Bank Marketing additional (B2B-like cold calling, readable job/age segments) as the primary pair; use Open Bandit/Criteo only as OPE method checks.
2. Before using Kaggle Lead Scoring, open the Kaggle page manually and record licence and provenance.
3. Confirm Hillstrom redistribution rights with the author if derived data will be published.
