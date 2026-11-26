"""Power analysis for the prospective founder cohort (docs/cohort/analysis-plan.md), by simulation.

Mirrors the cohort's evaluation design. Each simulated company:
- market: fprior.sim, 1,500 accounts, base rate ~ U(0.03, 0.15);
- founder reliability rho ~ U(-0.25, 1), the range we expect of real founders;
- founder-steered outreach (as in the cohort, the founder picks whom to contact) for C ~ U(60, 300) contacts;
- the first k contacts' outcomes are the "early" labels, and every method ranks the company's LATER contacts;
- evaluation is AUC of realized outcomes among later contacts. This is the only thing observable in the cohort,
  since uncontacted accounts have no outcome.

Design B ("randomized slice", added after design A showed low power): in addition, the founder contacts R = 40
accounts drawn uniformly at random from the market; methods are evaluated by AUC on that random slice (still
trained on the first k founder-chosen contacts). This is the cohort analogue of an exploration budget.

Methods: prior only (founder beliefs), labels only (CPR-LP without founder), CPR-LP (beliefs + early labels;
no graph, as real CRM exports have none). Company-level paired differences are pooled; cohorts of M companies are
resampled from the pool, and power = share of cohorts whose 95% bootstrap interval for the mean difference
excludes 0 in the predicted direction.

  python scripts/cohort_power.py      # writes results/cohort_power.md
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fprior.cprlp import CPRConfig, fit_predict  # noqa: E402
from fprior.sim import FounderConfig, MarketConfig, OutreachConfig, make_founder, make_market, run_outreach  # noqa: E402

KS = [5, 10, 20]
N_COMP = 600
MS = [10, 15, 20, 30, 50]


def auc(s, y):
    pos, neg = y > 0.5, y < 0.5
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    r = rankdata(s)
    return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())


def main():
    rows = []
    for c in range(N_COMP):
        rng = np.random.default_rng([c, 31])
        m = make_market(MarketConfig(n_accounts=1500, base_rate=float(rng.uniform(0.03, 0.15))), rng)
        rho = float(rng.uniform(-0.25, 1.0))
        f = make_founder(m, FounderConfig(reliability=rho), rng)
        C = int(rng.integers(60, 301))
        out = run_outreach(m, f, OutreachConfig(n_customers=10 ** 9, max_contacts=C), rng)
        I = np.eye(m.n)
        cfg = replace(CPRConfig(), graph_sd=0.0)
        row = {"company": c, "rho": rho, "contacts": len(out.contacted), "rate": float(out.outcome.mean())}
        rest = np.setdiff1d(np.arange(m.n), out.contacted)
        rand = np.random.default_rng([c, 32]).choice(rest, 40, replace=False)
        for k in KS:
            later = out.contacted[k:]
            yl = out.outcome[k:]
            args = (m.segment, m.n_segments, I, f.segment_logit, out.contacted[:k], out.outcome[:k])
            sc = {"prior": f.segment_logit[m.segment],
                  "labels": fit_predict(*args, replace(cfg, use_founder=False)).score,
                  "cpr": fit_predict(*args, cfg).score}
            for name, s in sc.items():
                row[f"auc_{name}_k{k}"] = auc(s[later], yl)
                row[f"aucall_{name}_k{k}"] = auc(s, m.y)   # oracle: all accounts (not observable in the cohort)
                row[f"aucrand_{name}_k{k}"] = auc(s[rand], m.y[rand])
        rows.append(row)
        if c % 100 == 99:
            print(f"{c + 1} companies", flush=True)
    (ROOT / "results/cohort_power.json").write_text(json.dumps(rows))

    rng = np.random.default_rng(0)
    L = ["# Cohort power analysis (simulation)", "",
         f"{N_COMP} simulated companies (fprior.sim; rho ~ U(−0.25, 1); 60–300 founder-chosen contacts). Metric: AUC of "
         "realized outcomes among contacts made after the first k. Script: `scripts/cohort_power.py`.", "",
         "## Company-level effect sizes (mean ± sd over companies)", "",
         "| k | CPR-LP − labels only | CPR-LP − prior only | same, oracle AUC on all accounts (CPR − labels) |",
         "|---|---|---|---|"]
    diffs = {}
    for k in KS:
        a = np.array([r[f"auc_cpr_k{k}"] - r[f"auc_labels_k{k}"] for r in rows])
        b = np.array([r[f"auc_cpr_k{k}"] - r[f"auc_prior_k{k}"] for r in rows])
        o = np.array([r[f"aucall_cpr_k{k}"] - r[f"aucall_labels_k{k}"] for r in rows])
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        diffs[k] = (a, b)
        L.append(f"| {k} | {a.mean():+.3f} ± {a.std():.3f} | {b.mean():+.3f} ± {b.std():.3f} | {np.nanmean(o):+.3f} ± {np.nanstd(o):.3f} |")
    L += ["", "## Power: share of 1,000 resampled cohorts whose 95% bootstrap CI excludes 0 (predicted sign > 0)", "",
          "| companies | " + " | ".join(f"CPR−labels k={k}" for k in KS) + " | " + " | ".join(f"CPR−prior k={k}" for k in KS) + " |",
          "|---|" + "---|" * (2 * len(KS))]

    def power(pool, M):
        hit = 0
        for _ in range(1000):
            s = rng.choice(pool, M, replace=True)
            bs = rng.choice(s, (400, M), replace=True).mean(1)
            hit += np.quantile(bs, 0.025) > 0
        return hit / 1000

    for M in MS:
        L.append(f"| {M} | " + " | ".join(f"{power(diffs[k][0], M):.2f}" for k in KS) + " | "
                 + " | ".join(f"{power(diffs[k][1], M):.2f}" for k in KS) + " |")
    L += ["", "## Design B: evaluation on a random slice of 40 accounts per company", "",
          "| k | CPR-LP − labels only | CPR-LP − prior only |", "|---|---|---|"]
    dB = {}
    for k in KS:
        a = np.array([r[f"aucrand_cpr_k{k}"] - r[f"aucrand_labels_k{k}"] for r in rows])
        b = np.array([r[f"aucrand_cpr_k{k}"] - r[f"aucrand_prior_k{k}"] for r in rows])
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        dB[k] = (a, b)
        L.append(f"| {k} | {a.mean():+.3f} ± {a.std():.3f} | {b.mean():+.3f} ± {b.std():.3f} |")
    L += ["", "| companies | " + " | ".join(f"CPR−labels k={k}" for k in KS) + " | " + " | ".join(f"CPR−prior k={k}" for k in KS) + " |",
          "|---|" + "---|" * (2 * len(KS))]
    for M in MS:
        L.append(f"| {M} | " + " | ".join(f"{power(dB[k][0], M):.2f}" for k in KS) + " | "
                 + " | ".join(f"{power(dB[k][1], M):.2f}" for k in KS) + " |")
    md = "\n".join(L)
    print(md)
    (ROOT / "results/cohort_power.md").write_text(md + "\n")


if __name__ == "__main__":
    main()
