"""Step 3b: per-contract detail from USAspending, to separate real purchases from R&D / SBIR awards.

Why: SBIR.gov's contract list misses some SBIR/STTR awards (verified: NASA NNX16CC57P is an STTR
Phase I, solicitation "NNX16STTRPHI", absent from SBIR.gov for that firm). A firm's R&D contracts
(SBIR, STTR, BAA research) are not customers buying its product, so the primary outcome is a
NON-R&D contract: product_or_service_description not starting "R&D" and PSC not starting "A".
The detail endpoint also gives the awarding office (finer "account" than sub-agency).

Endpoint (verified): GET https://api.usaspending.gov/api/v2/awards/<generated_internal_id>/
Cached per award under data/raw/usaspending_awards/. Output: data/interim/sbir_outcomes_detail.csv.gz
"""

from __future__ import annotations

import glob
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/raw/usaspending_awards"
SBIR_TEXT = re.compile(r"SBIR|STTR", re.I)


def detail(gid: str) -> dict | None:
    cp = CACHE / f"{re.sub(r'[^A-Za-z0-9_-]', '_', gid)}.json"
    if cp.exists():
        return json.loads(cp.read_text())
    for attempt in range(4):
        try:
            r = requests.get(f"https://api.usaspending.gov/api/v2/awards/{gid}/", timeout=60)
            if r.status_code == 200:
                d = r.json()
                c = d.get("latest_transaction_contract_data") or {}
                slim = {"psc": c.get("product_or_service_code"), "psc_desc": c.get("product_or_service_description"),
                        "solicitation": c.get("solicitation_identifier"),
                        "set_aside": c.get("type_set_aside_description"),
                        "office": (d.get("awarding_agency") or {}).get("office_agency_name")}
                cp.write_text(json.dumps(slim))
                time.sleep(0.25)
                return slim
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return None


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    gid = {}
    for f in glob.glob(str(ROOT / "data/raw/usaspending/*.json")):
        for r in json.loads(Path(f).read_text())["results"]:
            gid[r["Award ID"]] = r["generated_internal_id"]
    out = pd.read_csv(ROOT / "data/interim/sbir_outcomes.csv.gz", dtype={"duns": str, "naics": str})
    out = out[out["recipient_name_match"]].copy()
    rows, fails = [], 0
    for i, aid in enumerate(out["award_id"]):
        d = detail(gid[aid]) if aid in gid else None
        fails += d is None
        rows.append(d or {})
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(out)} details", flush=True)
    det = pd.DataFrame(rows, index=out.index)
    out = out.join(det)
    psc = out["psc"].fillna("").astype(str)
    out["is_rd"] = psc.str.startswith("A") | out["psc_desc"].fillna("").str.upper().str.startswith("R&D")
    out["is_sbir"] = out["is_sbir"] | out["solicitation"].fillna("").str.contains(SBIR_TEXT)
    out["is_purchase"] = ~out["is_rd"] & ~out["is_sbir"]
    out.to_csv(ROOT / "data/interim/sbir_outcomes_detail.csv.gz", index=False)
    print(f"contracts {len(out):,}; detail failures {fails}; R&D {out['is_rd'].mean():.1%}; "
          f"SBIR-flagged {out['is_sbir'].mean():.1%}; purchases {out['is_purchase'].mean():.1%}")


if __name__ == "__main__":
    main()
