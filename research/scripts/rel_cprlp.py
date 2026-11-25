"""CPR-LP-rel: CPR-LP with public relational regressors, on sim_rel markets across novelty.

cpr_lp_rel = CPR-LP (founder prior, learned trust, omega averaging, co-adoption GMRF) plus two
standardized public regressors in the mean, coefficients ~ N(0, extra_sd^2) integrated out:
  z(similar_adopter_score), z(log(1 + adoption degree)).
labels_only_rel = the same without the founder.

Founder / outreach RNG conventions match scripts/sanity_ladder.py.

  python scripts/rel_cprlp.py [--seeds 10] [--n-accounts 3000]
Writes results/rel_cprlp.md and .json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fprior.cprlp import CPRConfig, fit_predict  # noqa: E402
from fprior.metrics import expected_auc, lift_at_k  # noqa: E402
from fprior.sim import FounderConfig, OutreachConfig, make_founder, run_outreach  # noqa: E402
from fprior.sim_rel import RelMarketConfig, make_rel_market, similar_adopter_score  # noqa: E402

NOVELTY = [0.0, 0.5, 1.0]
RHOS = [1.0, 0.5, 0.0, -0.5]
NS = [0, 1, 5, 10, 20, 50]
METHODS = ["prior_only", "similar_adopters", "labels_only_rel", "cpr_lp", "cpr_lp_rel"]
TOP_K = 100


def z(x):
    return (x - x.mean()) / max(x.std(), 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-accounts", type=int, default=3000)
    ap.add_argument("--out", default=str(ROOT / "results" / "rel_cprlp.md"))
    a = ap.parse_args()
    base = CPRConfig()
    rows, t0 = [], time.time()
    for e in NOVELTY:
        for seed in range(a.seeds):
            market = make_rel_market(RelMarketConfig(n_accounts=a.n_accounts, novelty=e),
                                     np.random.default_rng(seed))
            sa = similar_adopter_score(market)
            X = np.column_stack([z(sa), z(np.log1p(market.degree))])
            for rho in RHOS:
                for n in NS:
                    rng = np.random.default_rng([seed, int(round(rho * 100)) + 1000, n])
                    founder = make_founder(market, FounderConfig(reliability=rho), rng)
                    out = run_outreach(market, founder, OutreachConfig(n_customers=n), rng)
                    U = np.setdiff1d(np.arange(market.n), out.contacted)
                    args = (market.segment, market.n_segments, market.graph_cov, founder.segment_logit,
                            out.contacted, out.outcome)
                    rel = fit_predict(*args, base, extra_features=X)
                    lor = fit_predict(*args, replace(base, use_founder=False), extra_features=X)
                    sc = {"prior_only": founder.segment_logit[market.segment], "similar_adopters": sa,
                          "labels_only_rel": lor.score, "cpr_lp": fit_predict(*args, base).score,
                          "cpr_lp_rel": rel.score}
                    er = np.random.default_rng([seed, 7])
                    row = {"novelty": e, "seed": seed, "rho": rho, "n": n, "contacts": int(len(out.contacted)),
                           "coef_sa_cpr": float(rel.coef[0]), "coef_deg_cpr": float(rel.coef[1]),
                           "coef_sa_lab": float(lor.coef[0]), "trust_b_rel": float(rel.trust[1])}
                    for k in METHODS:
                        row[f"lift_{k}"] = lift_at_k(sc[k][U], market.p[U], TOP_K, er)
                        row[f"auc_{k}"] = expected_auc(sc[k][U], market.p[U])
                    rows.append(row)
            print(f"novelty {e} seed {seed} done ({time.time() - t0:.0f}s)", flush=True)

    ms = lambda v: f"{np.mean(v):.2f} ± {np.std(v, ddof=1) / np.sqrt(len(v)):.2f}"  # noqa: E731
    lines = [f"# CPR-LP-rel on sim_rel markets ({a.seeds} seeds, {a.n_accounts} accounts)", "",
             f"Expected lift@{TOP_K} on uncontacted accounts, mean ± SE over seeds; Δ paired over seeds.",
             "cpr_lp_rel adds z(similar_adopter_score) and z(log degree) to CPR-LP's mean with",
             "N(0, 0.5²) coefficient priors integrated out; labels_only_rel = same, no founder.", ""]
    for e in NOVELTY:
        for rho in RHOS:
            hdr = ["n", *METHODS, "rel − cpr_lp", "rel − adopters", "coef(adopters)"]
            lines += [f"### novelty = {e}, ρ = {rho}", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
            for n in NS:
                cell = [r for r in rows if r["novelty"] == e and r["rho"] == rho and r["n"] == n]
                vals = [ms([r[f"lift_{k}"] for r in cell]) for k in METHODS]
                vals += [ms([r["lift_cpr_lp_rel"] - r["lift_cpr_lp"] for r in cell]),
                         ms([r["lift_cpr_lp_rel"] - r["lift_similar_adopters"] for r in cell]),
                         ms([r["coef_sa_cpr"] for r in cell])]
                lines.append(f"| {n} | " + " | ".join(vals) + " |")
            lines.append("")
    lines += ["## Learned coefficient on similar_adopters vs novelty (cpr_lp_rel, all ρ pooled)", "",
              "| novelty | " + " | ".join(f"n={n}" for n in NS) + " |", "|" + "---|" * (len(NS) + 1)]
    for e in NOVELTY:
        lines.append(f"| {e} | " + " | ".join(
            ms([r["coef_sa_cpr"] for r in rows if r["novelty"] == e and r["n"] == n]) for n in NS) + " |")
    md = "\n".join(lines)
    print(md)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(md)
    Path(a.out).with_suffix(".json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
