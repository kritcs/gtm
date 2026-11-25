"""Step 1-2 of the SBIR -> USAspending pipeline: download SBIR award data and build the cohort.

Source: SBIR.gov bulk award CSV (verified 2026-11-24: HTTP 200, 367,551,355 bytes,
last-modified 2026-01-01): https://data.www.sbir.gov/awarddatapublic/award_data.csv

Cohort: firms (keyed by DUNS) whose FIRST SBIR/STTR award of any phase is a Phase I award dated
2012-2018. "First award" means the firm has no earlier award in the file, which also drops
"SBIR mills" (firms with long award histories) by construction.

Personal contact fields in the CSV (contact / PI / RI names, emails, phones) are never loaded.
Outputs (gitignored, research/data/interim/):
  sbir_cohort.csv.gz      one row per cohort firm, with its first award's agency, branch, topic,
                          title and abstract (the dated "founder belief" text)
  sbir_contracts.csv.gz   every SBIR/STTR contract number per DUNS (to exclude SBIR awards from outcomes)
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://data.www.sbir.gov/awarddatapublic/award_data.csv"
RAW = ROOT / "data/raw/sbir/award_data.csv"
OUT = ROOT / "data/interim"
COLS = ["Company", "Award Title", "Agency", "Branch", "Phase", "Program", "Contract",
        "Proposal Award Date", "Award Year", "Award Amount", "Duns", "Number Employees", "State",
        "Topic Code", "Abstract", "Solicitation Year"]
YEARS = (2012, 2018)


def norm_duns(x) -> str | None:
    if not isinstance(x, str):
        return None
    d = re.sub(r"\D", "", x)
    if not d or set(d) == {"0"} or len(d) > 9:
        return None
    return d.zfill(9)


def norm_award_id(x) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(x).upper())


def main():
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sS", "-o", str(RAW), URL], check=True)
    df = pd.read_csv(RAW, usecols=COLS, dtype=str, low_memory=False)
    n_raw = len(df)
    df["duns"] = df["Duns"].map(norm_duns)
    df["date"] = pd.to_datetime(df["Proposal Award Date"], format="%m/%d/%Y", errors="coerce")
    yr = pd.to_numeric(df["Award Year"], errors="coerce")
    df["date"] = df["date"].fillna(pd.to_datetime(yr.astype("Int64").astype(str) + "-07-01", errors="coerce"))
    all_awards = df[df["date"].notna()].copy()             # incl. awards without DUNS, for the name check
    df = df[df["duns"].notna() & df["date"].notna()]

    contracts = df[["duns", "Contract"]].dropna().copy()
    contracts["award_id"] = contracts["Contract"].map(norm_award_id)
    contracts = contracts[contracts["award_id"].str.len() > 3][["duns", "award_id"]].drop_duplicates()

    df = df.sort_values(["duns", "date"])
    first = df.groupby("duns").head(1)
    n_awards = df.groupby("duns").size().rename("n_sbir_awards_total")
    first = first.join(n_awards, on="duns")
    coh = first[(first["Phase"] == "Phase I") & first["date"].dt.year.between(*YEARS)].copy()
    # drop firms with an earlier award under the same normalized name (e.g. an older award lacking DUNS)
    name = lambda s: s.fillna("").str.upper().str.replace(r"[^A-Z0-9 ]", "", regex=True).str.replace(  # noqa: E731
        r"\b(INC|LLC|CORP|CORPORATION|CO|LTD|LP|INCORPORATED|COMPANY)\b", "", regex=True).str.split().str.join(" ")
    all_awards["nm"] = name(all_awards["Company"])
    earliest_by_name = all_awards.groupby("nm")["date"].min()
    coh["nm"] = name(coh["Company"])
    earlier = coh["nm"].map(earliest_by_name) < coh["date"]
    print(f"dropped {int(earlier.sum()):,} firms with an earlier award under the same name")
    coh = coh[~earlier]
    coh = coh.rename(columns={"Company": "company", "Agency": "sbir_agency", "Branch": "sbir_branch",
                              "Program": "program", "Topic Code": "topic", "Award Title": "title",
                              "Abstract": "abstract", "Number Employees": "employees", "State": "state",
                              "Award Amount": "amount", "date": "t0"})
    keep = ["duns", "company", "t0", "sbir_agency", "sbir_branch", "program", "topic", "title", "abstract",
            "employees", "state", "amount", "n_sbir_awards_total"]
    coh = coh[keep]
    OUT.mkdir(parents=True, exist_ok=True)
    coh.to_csv(OUT / "sbir_cohort.csv.gz", index=False)
    contracts.to_csv(OUT / "sbir_contracts.csv.gz", index=False)
    print(f"raw awards {n_raw:,}; awards with DUNS+date {len(df):,}; firms {df['duns'].nunique():,}")
    print(f"cohort (first award = Phase I in {YEARS[0]}-{YEARS[1]}): {len(coh):,} firms")
    print("cohort by SBIR agency:\n" + coh["sbir_agency"].value_counts().head(12).to_string())
    print(f"abstract present: {coh['abstract'].notna().mean():.1%}; SBIR contract ids: {len(contracts):,}")


if __name__ == "__main__":
    sys.exit(main())
