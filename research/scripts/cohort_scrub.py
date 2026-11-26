"""Strip personal data from a founder's outreach export before it leaves their machine (founder cohort).

Keeps only: a salted hash of the account identifier, the segment attribute columns the founder named on the
elicitation form, the contact date, the furthest stage reached and its date. Everything else is dropped. The
salt never leaves the founder's machine, so the research team cannot reverse the hashes.

  python scripts/cohort_scrub.py export.csv scrubbed.csv --id company_domain \
      --attrs industry,employees --contacted contacted_on --stage stage --stage-date stage_date --salt-file ~/.cohort_salt

Fails loudly (and writes nothing) if a kept column looks like it contains e-mails, phone numbers or URLs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import secrets
import sys
from pathlib import Path

STAGES = ["none", "reply", "meeting", "opportunity", "pilot", "contract"]
PII = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|https?://|www\.|\+?\d[\d\s().-]{7,}\d")


def salt_from(path: Path) -> str:
    if not path.exists():
        path.write_text(secrets.token_hex(16))
        path.chmod(0o600)
    return path.read_text().strip()


def scrub(rows, id_col, attrs, contacted, stage, stage_date, salt):
    out = []
    for r in rows:
        st = (r.get(stage) or "none").strip().lower()
        if st not in STAGES:
            raise ValueError(f"unknown stage {st!r}; expected one of {STAGES}")
        rec = {"account_key": hashlib.sha256((salt + r[id_col].strip().lower()).encode()).hexdigest()[:16]}
        for a in attrs:
            v = (r.get(a) or "").strip()
            if PII.search(v):
                raise ValueError(f"column {a!r} looks like it contains personal data: refusing to export")
            rec[a] = v
        rec["contacted_on"] = (r.get(contacted) or "").strip()
        rec["stage_reached"] = st
        rec["stage_date"] = (r.get(stage_date) or "").strip() if stage_date else ""
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--id", required=True)
    ap.add_argument("--attrs", required=True)
    ap.add_argument("--contacted", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--stage-date", default=None)
    ap.add_argument("--salt-file", default=str(Path.home() / ".cohort_salt"))
    a = ap.parse_args()
    with open(a.src, newline="") as fh:
        rows = list(csv.DictReader(fh))
    try:
        out = scrub(rows, a.id, a.attrs.split(","), a.contacted, a.stage, a.stage_date, salt_from(Path(a.salt_file)))
    except ValueError as e:
        sys.exit(f"cohort_scrub: {e}")
    with open(a.dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]) if out else ["account_key"])
        w.writeheader()
        w.writerows(out)
    print(f"wrote {len(out)} rows, columns: {list(out[0]) if out else []}")


if __name__ == "__main__":
    main()
