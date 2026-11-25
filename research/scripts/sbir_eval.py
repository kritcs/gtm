"""Real B2B tier: SBIR firms' first federal customers (docs/sbir-pipeline.md).

Task: firm f at t0 (first SBIR Phase I award, 2012–2018) ranks federal awarding sub-agencies by how
likely each is to make a real purchase (non-SBIR, non-R&D contract) within 5 years. Evaluated on
firms with >= 1 purchase. Time split: firms with t0 in 2012–2015 fit anything learned; 2016–2018 test.

"Founder belief" = keyword rules over the firm's own pre-t0 text (title, topic, abstract). No LLM
(a modern LLM may know later outcomes). The funding agency / branch is a known fact at t0, given to
every method.

Methods
  popularity          sub-agency purchase frequency among training firms
  funding             funding branch first, then other sub-agencies of the funding agency, then popularity
  keywords            abstract keyword belief, ties broken by popularity
  pooled              logistic regression fitted on training firms over
                        [same branch, same agency, log keyword hits, log popularity]
                      = calibrated trust in the keyword "founder" pooled across firms
  pooled_nokw         the same without the keyword feature (ablation: does the founder belief add?)

Pre-registered (committed before the full cohort's outcomes were available):
  SB1. On purchases from a sub-agency OUTSIDE the funding agency (the novel slice), pooled beats
       pooled_nokw on recall@3 (founder beliefs help where there is nothing to memorize).
  SB2. On purchases from the funding agency, funding ≈ pooled ≈ pooled_nokw (keywords add ~nothing).
  SB3. Overall, pooled >= max(popularity, funding, keywords) on MRR.
Bootstrap over firms (1,000 resamples) for 95% intervals of paired differences.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"

# keyword rules: substring of the sub-agency name -> patterns in the firm's own text
KEYWORDS = {
    "Navy": r"\bnavy\b|naval|maritime|\bship|submarine|undersea|marine corps|sonar",
    "Air Force": r"air force|\baircraft|airborne|\bspace force|\bjet\b|avionic",
    "Army": r"\barmy\b|soldier|ground vehicle|\barmor|infantry|warfighter",
    "Defense Logistics": r"logistic|spare part|supply chain|obsolescen|casting|sustainment",
    "Missile Defense": r"missile|interceptor|hypersonic",
    "Advanced Research Projects": r"\bdarpa\b",
    "Threat Reduction": r"chemical|biological|radiological|\bwmd\b|explosive",
    "Special Operations": r"special operations|\bsocom\b",
    "Defense Health": r"combat casualt|military medic|battlefield",
    "National Institutes of Health": r"patient|clinical|disease|cancer|therap|\bnih\b|biomedical",
    "Food and Drug": r"\bfda\b|drug safety|regulatory science",
    "Disease Control": r"public health|epidemi|outbreak|\bcdc\b",
    "Veterans": r"veteran",
    "Aeronautics and Space": r"\bnasa\b|spacecraft|\bspace\b|planetary|lunar|orbit|satellite",
    "Energy": r"\benergy\b|\bgrid\b|nuclear|battery|batteries|solar|fusion|\bdoe\b",
    "Agricultur": r"agricultur|\bcrop|livestock|\bfarm|forest",
    "Forest Service": r"forest|wildfire|timber",
    "Oceanic and Atmospheric": r"\bocean|weather|atmospher|fisher|\bnoaa\b|climate",
    "Standards and Technology": r"measurement|metrolog|\bnist\b|calibration standard",
    "Environmental Protection": r"pollut|emission|water quality|contaminant|\bepa\b",
    "Homeland Security": r"homeland|border|\bcbp\b|\btsa\b|emergency response|first responder",
    "Coast Guard": r"coast guard",
    "Federal Aviation": r"\bfaa\b|aviation|air traffic",
    "Transportation": r"highway|transit|\brail|traffic",
    "Geological": r"geolog|earthquake|mineral",
    "Census": r"census|survey data",
}


def keyword_hits(text: str, sub_agency: str) -> int:
    hits = 0
    for key, pat in KEYWORDS.items():
        if key.lower() in sub_agency.lower():
            hits += len(re.findall(pat, text, flags=re.I))
    return hits


def build():
    coh = pd.read_csv(INTERIM / "sbir_cohort.csv.gz", dtype={"duns": str}, parse_dates=["t0"])
    lab = pd.read_csv(INTERIM / "sbir_labels.csv.gz", dtype={"duns": str}, parse_dates=["t0"])
    lab = lab[lab["outcome"] == "purchase"].dropna(subset=["awarding_sub_agency"])
    pos = lab.groupby("duns")["awarding_sub_agency"].apply(lambda s: sorted(set(s))).to_dict()
    sub_to_agency = lab.drop_duplicates("awarding_sub_agency").set_index("awarding_sub_agency")["awarding_agency"].to_dict()
    firms = coh[coh["duns"].isin(pos)].copy()
    firms["year"] = firms["t0"].dt.year
    firms["text"] = (firms["title"].fillna("") + " " + firms["topic"].fillna("") + " " + firms["abstract"].fillna(""))
    return firms, pos, sub_to_agency


def features(firms, pos, sub_to_agency, cands, pop):
    rows = []
    for f in firms.itertuples(index=False):
        branch = str(f.sbir_branch) if isinstance(f.sbir_branch, str) else ""
        for s in cands:
            rows.append({"duns": f.duns, "year": f.year, "sub": s,
                         "same_branch": float(bool(branch) and (branch.lower() in s.lower() or s.lower() in branch.lower())),
                         "same_agency": float(sub_to_agency.get(s, "") == f.sbir_agency),
                         "kw": np.log1p(keyword_hits(f.text, s)),
                         "logpop": np.log(pop.get(s, 0) + 0.5),
                         "y": float(s in pos.get(f.duns, []))})
    return pd.DataFrame(rows)


def fit_logreg(X, y, l2=1.0, iters=200):
    w = np.zeros(X.shape[1] + 1)
    Xb = np.column_stack([np.ones(len(X)), X])
    for _ in range(iters):  # Newton / IRLS
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (y - p) - l2 * np.r_[0, w[1:]]
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        w += np.linalg.solve(H, g)
    return w


def main():
    firms, pos, sub_to_agency = build()
    train = firms[firms["year"] <= 2015]
    test = firms[firms["year"] >= 2016]
    pop = {}
    for d in train["duns"]:
        for s in pos[d]:
            pop[s] = pop.get(s, 0) + 1
    cands = sorted(set(s for d in firms["duns"] for s in pos[d]))
    F_tr = features(train, pos, sub_to_agency, cands, pop)
    F_te = features(test, pos, sub_to_agency, cands, pop)
    cols_kw = ["same_branch", "same_agency", "kw", "logpop"]
    cols_nokw = ["same_branch", "same_agency", "logpop"]
    w_kw = fit_logreg(F_tr[cols_kw].values, F_tr["y"].values)
    w_nokw = fit_logreg(F_tr[cols_nokw].values, F_tr["y"].values)
    score = lambda F, cols, w: np.column_stack([np.ones(len(F)), F[cols].values]) @ w  # noqa: E731
    F_te = F_te.assign(
        popularity=F_te["logpop"],
        funding=2 * F_te["same_branch"] + F_te["same_agency"] + 1e-3 * F_te["logpop"],
        keywords=F_te["kw"] + 1e-3 * F_te["logpop"],
        pooled=score(F_te, cols_kw, w_kw),
        pooled_nokw=score(F_te, cols_nokw, w_nokw))
    methods = ["popularity", "funding", "keywords", "pooled", "pooled_nokw"]

    # per (firm, purchased sub-agency): rank among candidates
    recs = []
    for d, g in F_te.groupby("duns"):
        for m in methods:
            order = g.sort_values(m, ascending=False, kind="stable")["sub"].tolist()
            for s in g.loc[g["y"] > 0, "sub"]:
                r = order.index(s) + 1
                slice_ = "funding_agency" if g.loc[g["sub"] == s, "same_agency"].iloc[0] > 0 else "new_agency"
                recs.append({"duns": d, "method": m, "sub": s, "rank": r, "slice": slice_})
    R = pd.DataFrame(recs)
    R["hit3"] = (R["rank"] <= 3).astype(float)
    R["rr"] = 1.0 / R["rank"]

    rng = np.random.default_rng(0)
    firms_te = R["duns"].unique()
    def boot_diff(a, b, metric, sl=None):
        S = R if sl is None else R[R["slice"] == sl]
        per = S.pivot_table(index="duns", columns="method", values=metric, aggfunc="mean")
        per = per.dropna(subset=[a, b])
        d = (per[a] - per[b]).values
        bs = [rng.choice(d, len(d)).mean() for _ in range(1000)]
        return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(d)

    lines = [__doc__.strip(), "", "## Results", "",
             f"Train firms (2012–2015, >=1 purchase): {len(train)}; test firms (2016–2018): {len(test)}; "
             f"candidate sub-agencies: {len(cands)}; test purchases: {len(R) // len(methods)} "
             f"({(R[R.method == 'pooled'].slice == 'new_agency').mean():.0%} from a new agency).", "",
             f"Pooled weights [intercept, same_branch, same_agency, kw, logpop]: {np.round(w_kw, 2).tolist()}", "",
             "| method | recall@3 all | recall@3 funding agency | recall@3 new agency | MRR all |", "|---|---|---|---|---|"]
    for m in methods:
        S = R[R.method == m]
        lines.append(f"| {m} | {S.hit3.mean():.3f} | {S[S.slice == 'funding_agency'].hit3.mean():.3f} | "
                     f"{S[S.slice == 'new_agency'].hit3.mean():.3f} | {S.rr.mean():.3f} |")
    lines += ["", "Paired differences over test firms (mean, bootstrap 95% CI, firms):", ""]
    for a, b, met, sl in [("pooled", "pooled_nokw", "hit3", "new_agency"), ("pooled", "pooled_nokw", "hit3", "funding_agency"),
                          ("pooled", "funding", "hit3", "funding_agency"), ("pooled", "popularity", "rr", None),
                          ("pooled", "funding", "rr", None), ("pooled", "keywords", "rr", None)]:
        m_, lo, hi, k = boot_diff(a, b, met, sl)
        lines.append(f"- {a} − {b}, {met}, {sl or 'all'}: {m_:+.3f} [{lo:+.3f}, {hi:+.3f}] (n = {k})")
    md = "\n".join(lines)
    print(md)
    (ROOT / "results/sbir_eval.md").write_text(md)
    R.to_json(ROOT / "results/sbir_eval.json", orient="records")


if __name__ == "__main__":
    main()
