"""Step 3 of the SBIR -> USAspending pipeline: later federal contracts of cohort firms.

For a seeded random sample of cohort firms, query USAspending (verified endpoint, no key needed):
  POST https://api.usaspending.gov/api/v2/search/spending_by_award/
  filters: award_type_codes A-D (contracts), recipient_search_text = [DUNS],
           time_period = [T0, min(T0 + 5 years, 2026-09-30)]
Raw responses are cached per firm under data/raw/usaspending/ (resumable). Polite: one request at
a time with a pause, retries with backoff.

Each returned contract is flagged as SBIR/STTR if its award id matches any SBIR.gov contract number,
or its description mentions SBIR / STTR / Small Business Innovation / Technology Transfer.
A recipient-name check guards against DUNS text matches on the wrong company.
Output: data/interim/sbir_outcomes.csv.gz (one row per contract), and a per-firm query log.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
CACHE = ROOT / "data/raw/usaspending"
URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
FIELDS = ["Award ID", "Recipient Name", "Recipient UEI", "Start Date", "Award Amount", "Awarding Agency",
          "Awarding Sub Agency", "Funding Agency", "Funding Sub Agency", "NAICS", "Description",
          "Contract Award Type", "generated_internal_id"]
SBIR_TEXT = re.compile(r"SBIR|STTR|SMALL BUSINESS INNOVATION|SMALL BUSINESS TECHNOLOGY TRANSFER", re.I)
STOP = {"INC", "LLC", "CORP", "CORPORATION", "CO", "LTD", "LP", "THE", "INCORPORATED", "COMPANY", "OF", "AND"}


def tokens(name: str) -> set[str]:
    return {t for t in re.sub(r"[^A-Z0-9 ]", " ", str(name).upper()).split() if t not in STOP and len(t) > 1}


def fetch(duns: str, start: str, end: str, max_pages: int = 10) -> dict:
    out, page, err = [], 1, None
    while page <= max_pages:
        body = {"filters": {"award_type_codes": ["A", "B", "C", "D"], "recipient_search_text": [duns],
                            "time_period": [{"start_date": start, "end_date": end}]},
                "fields": FIELDS, "limit": 100, "page": page, "sort": "Start Date", "order": "asc"}
        for attempt in range(4):
            try:
                r = requests.post(URL, json=body, timeout=60)
                if r.status_code == 200:
                    break
                err = f"HTTP {r.status_code}: {r.text[:200]}"
            except requests.RequestException as e:  # network hiccup
                err = repr(e)[:200]
            time.sleep(2 ** attempt)
        else:
            return {"results": out, "error": err, "pages": page - 1}
        js = r.json()
        out += js.get("results", [])
        if not js.get("page_metadata", {}).get("hasNext"):
            break
        page += 1
        time.sleep(0.3)
    return {"results": out, "error": None, "pages": page}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--years", type=int, default=5)
    a = ap.parse_args()
    coh = pd.read_csv(INTERIM / "sbir_cohort.csv.gz", dtype={"duns": str}, parse_dates=["t0"])
    sbir_ids = set(pd.read_csv(INTERIM / "sbir_contracts.csv.gz", dtype=str)["award_id"])
    sample = coh.sample(n=min(a.n, len(coh)), random_state=a.seed)
    CACHE.mkdir(parents=True, exist_ok=True)
    rows, log = [], []
    for i, f in enumerate(sample.itertuples(index=False)):
        start = f.t0.strftime("%Y-%m-%d")
        end = min(f.t0 + pd.DateOffset(years=a.years), pd.Timestamp("2026-09-30")).strftime("%Y-%m-%d")
        cp = CACHE / f"{f.duns}_{start}_{end}.json"
        if cp.exists():
            js = json.loads(cp.read_text())
        else:
            js = fetch(f.duns, start, end)
            if js["error"] is None:
                cp.write_text(json.dumps(js))
            time.sleep(0.5)
        ftok = tokens(f.company)
        n_match = 0
        for r in js["results"]:
            aid = re.sub(r"[^A-Z0-9]", "", str(r.get("Award ID", "")).upper())
            rtok = tokens(r.get("Recipient Name", ""))
            name_ok = bool(ftok & rtok) if ftok and rtok else False
            n_match += name_ok
            naics = r.get("NAICS") or {}
            rows.append({"duns": f.duns, "award_id": r.get("Award ID"), "start_date": r.get("Start Date"),
                         "amount": r.get("Award Amount"), "awarding_agency": r.get("Awarding Agency"),
                         "awarding_sub_agency": r.get("Awarding Sub Agency"),
                         "funding_sub_agency": r.get("Funding Sub Agency"),
                         "naics": naics.get("code") if isinstance(naics, dict) else None,
                         "award_type": r.get("Contract Award Type"),
                         "is_sbir": aid in sbir_ids or bool(SBIR_TEXT.search(str(r.get("Description", "")))),
                         "recipient_name_match": name_ok})
        log.append({"duns": f.duns, "t0": start, "end": end, "n_contracts": len(js["results"]),
                    "n_name_match": n_match, "error": js["error"], "pages": js.get("pages")})
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(sample)} firms queried", flush=True)
    pd.DataFrame(rows).to_csv(INTERIM / "sbir_outcomes.csv.gz", index=False)
    lg = pd.DataFrame(log)
    lg.to_csv(INTERIM / "sbir_query_log.csv", index=False)
    print(f"firms {len(lg)}, errors {lg['error'].notna().sum()}, contracts {len(rows):,}, "
          f"capped at 10 pages: {(lg['pages'] >= 10).sum()}")


if __name__ == "__main__":
    main()
