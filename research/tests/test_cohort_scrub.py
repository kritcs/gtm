import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cohort_scrub import scrub  # noqa: E402


def rows():
    return [{"domain": "Acme.com", "industry": "fintech", "size": "50-200", "contact": "jane@acme.com",
             "date": "2027-01-05", "stage": "Meeting", "sd": "2027-01-09"}]


def test_scrub_keeps_only_whitelisted_and_hashes_id():
    out = scrub(rows(), "domain", ["industry", "size"], "date", "stage", "sd", "salt")
    assert set(out[0]) == {"account_key", "industry", "size", "contacted_on", "stage_reached", "stage_date"}
    assert "acme" not in out[0]["account_key"] and out[0]["stage_reached"] == "meeting"
    assert scrub(rows(), "domain", ["industry"], "date", "stage", None, "salt")[0]["account_key"] == out[0]["account_key"]


def test_scrub_refuses_pii_in_kept_columns_and_bad_stage():
    with pytest.raises(ValueError):
        scrub(rows(), "domain", ["contact"], "date", "stage", None, "s")
    r = rows()
    r[0]["stage"] = "maybe"
    with pytest.raises(ValueError):
        scrub(r, "domain", ["industry"], "date", "stage", None, "s")
