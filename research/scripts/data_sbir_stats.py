"""Step 4-5: label table and descriptive statistics for the SBIR -> USAspending task.

Label table data/interim/sbir_labels.csv.gz: one row per (firm, awarding sub-agency) with the firm's
first qualifying contract there within the outcome window. Two outcome definitions:
  purchase  — non-R&D, non-SBIR/STTR contract (primary: someone bought a product/service)
  non_sbir  — any contract not flagged SBIR/STTR (includes non-SBIR R&D)
Prints the statistics used in docs/sbir-pipeline.md.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
I = ROOT / "data/interim"

BRANCH_HINT = {"Air Force": "Air Force", "Army": "Army", "Navy": "Navy", "DARPA": "Defense Advanced Research",
               "MDA": "Missile Defense", "DLA": "Defense Logistics", "SOCOM": "Special Operations",
               "DTRA": "Defense Threat Reduction", "NIH": "National Institutes of Health", "CDC": "Centers for Disease",
               "FDA": "Food and Drug", "NIST": "National Institute of Standards", "NOAA": "National Oceanic"}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).upper()).strip()


def same_sub(branch, sub) -> bool:
    if not isinstance(branch, str) or not isinstance(sub, str):
        return False
    hint = next((v for k, v in BRANCH_HINT.items() if k.upper() in branch.upper()), branch)
    return norm(hint) in norm(sub) or norm(sub) in norm(hint)


def main():
    coh = pd.read_csv(I / "sbir_cohort.csv.gz", dtype={"duns": str}, parse_dates=["t0"])
    log = pd.read_csv(I / "sbir_query_log.csv", dtype={"duns": str})
    out = pd.read_csv(I / "sbir_outcomes_detail.csv.gz", dtype={"duns": str, "naics": str},
                      parse_dates=["start_date"])
    sample = coh[coh["duns"].isin(log["duns"])].set_index("duns")
    out = out.join(sample[["t0", "sbir_agency", "sbir_branch"]], on="duns")
    out = out[out["start_date"] >= out["t0"]]

    lines = [f"Sampled firms: {len(sample)} (query errors: {int(log['error'].notna().sum())})",
             f"Contracts returned (name-matched, on/after T0, within window): {len(out):,}"]
    for kind, mask in [("any contract", out.index == out.index), ("non-SBIR", ~out["is_sbir"]),
                       ("purchase (non-R&D, non-SBIR)", out["is_purchase"])]:
        firms = out[mask]["duns"].nunique()
        lines.append(f"Firms with ≥1 {kind}: {firms} / {len(sample)} ({firms / len(sample):.1%})")

    labels = []
    for kind, mask in [("purchase", out["is_purchase"]), ("non_sbir", ~out["is_sbir"])]:
        sub = out[mask].sort_values("start_date").groupby(["duns", "awarding_sub_agency"]).head(1)
        sub = sub.assign(outcome=kind)
        labels.append(sub[["duns", "t0", "sbir_agency", "sbir_branch", "awarding_agency", "awarding_sub_agency",
                           "office", "start_date", "amount", "naics", "psc", "outcome"]])
    lab = pd.concat(labels)
    lab.to_csv(I / "sbir_labels.csv.gz", index=False)

    p = lab[lab["outcome"] == "purchase"]
    first = p.sort_values("start_date").groupby("duns").head(1)
    lines += [f"Purchase labels (firm × sub-agency): {len(p)}; distinct sub-agencies {p['awarding_sub_agency'].nunique()}, "
              f"distinct offices {p['office'].nunique()}",
              f"Median sub-agencies bought from per buying firm: {p.groupby('duns').size().median():.0f} "
              f"(max {p.groupby('duns').size().max()})",
              f"Median days T0 → first purchase: {(first['start_date'] - first['t0']).dt.days.median():.0f}",
              f"First purchase from the SBIR funding AGENCY: {(first['awarding_agency'] == first['sbir_agency']).mean():.1%}",
              f"First purchase from the SBIR funding SUB-AGENCY (branch match, heuristic): "
              f"{first.apply(lambda r: same_sub(r['sbir_branch'], r['awarding_sub_agency']), axis=1).mean():.1%}",
              f"Any purchase from the SBIR funding agency: "
              f"{p.assign(m=p['awarding_agency'] == p['sbir_agency']).groupby('duns')['m'].any().mean():.1%} of buying firms",
              "", "First-purchase sub-agency (top 12):", first["awarding_sub_agency"].value_counts().head(12).to_string(),
              "", "Share of sampled firms with any purchase, by SBIR agency:",
              (sample.assign(buy=sample.index.isin(p["duns"])).groupby("sbir_agency")["buy"]
               .agg(["mean", "size"]).sort_values("size", ascending=False).head(8).round(2).to_string())]
    print("\n".join(lines))
    (I / "sbir_stats.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
