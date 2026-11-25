"""Evaluate relational PC-RFM (v1) on fprior.sim_rel markets across the novelty knob.

Same founder/outreach RNG conventions as scripts/sanity_ladder.py. Baselines are recomputed
here so that every difference is paired over the same markets, founders and contacts.

  python scripts/eval_pfn_rel.py --ckpt runs/pfn_v1/model.pt --ckpt-nf runs/pfn_v1_nf/model.pt \
      [--ckpt-v0 runs/pfn_v0/model.pt] --out results/m2_pfn_v1.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fprior.cprlp import CPRConfig, fit_predict  # noqa: E402
from fprior.cprlp_hier import fit_predict_hier  # noqa: E402
from fprior.metrics import expected_auc, lift_at_k  # noqa: E402
from fprior.pfn.data import make_tokens  # noqa: E402
from fprior.pfn.data_rel import rel_features  # noqa: E402
from fprior.pfn.model import PCRFM  # noqa: E402
from fprior.pfn.model_rel import PCRFMRel, product_features  # noqa: E402
from fprior.sim import FounderConfig, OutreachConfig, make_founder, run_outreach  # noqa: E402
from fprior.sim_rel import RelMarketConfig, make_rel_market, similar_adopter_score  # noqa: E402

NOVELTY = [0.0, 0.5, 1.0]
RHOS = [1.0, 0.5, 0.0, -0.5]
NS = [0, 1, 5, 10, 20, 50]
TOP_K = 100


def load(path, dev):
    ck = torch.load(path, map_location=dev)
    m = (PCRFMRel if ck["arch"].get("encoder") else PCRFM)(**ck["arch"]).to(dev).eval()
    m.load_state_dict(ck["state"])
    return m, ck["arch"].get("n_feat")


def tokens(market, founder, out, use_founder, relational, dev, noise_rng=None):
    t = lambda a, dt=torch.long: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)[None]  # noqa: E731
    ctx = np.zeros(market.n, dtype=bool)
    ctx[out.contacted] = True
    outcome = np.zeros(market.n)
    outcome[out.contacted] = out.outcome
    belief = founder.segment_logit[market.segment]
    if noise_rng is not None:   # control: same values scrambled across accounts (no information)
        belief = noise_rng.permutation(belief)
    x = make_tokens(t(market.industry), t(market.size),
                    t(belief, torch.float32),
                    torch.tensor([1.0 if use_founder else 0.0], device=dev),
                    t(ctx, torch.bool), t(outcome, torch.float32))
    if relational:
        x = torch.cat([x, rel_features(t(market.adoption.toarray(), torch.float32),
                                       t(market.seller_similarity, torch.float32),
                                       t(ctx, torch.bool), t(outcome, torch.float32))], -1)
    return x, t(ctx, torch.bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-nf", default=None)
    ap.add_argument("--ckpt-v0", default=None, help="no-graph v0 model, as an ablation")
    ap.add_argument("--ckpt-v1", default=None, help="v1 (features only), as an ablation of the v2 encoder")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-accounts", type=int, default=3000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = a.device
    models = {"pcrfm": (*load(a.ckpt, dev), True)}
    if a.ckpt_nf:
        models["pcrfm_nofounder"] = (*load(a.ckpt_nf, dev), False)
    if a.ckpt_v1:
        models["pcrfm_v1"] = (*load(a.ckpt_v1, dev), True)
    if a.ckpt_v0:
        models["pcrfm_v0"] = (*load(a.ckpt_v0, dev), True)
    base_methods = ["prior_only", "similar_adopters", "fusion", "labels_only", "cpr_lp", "cpr_lp_rel", "cpr_lp_hier_rel"]
    methods = base_methods + list(models) + [f"{k}_noise" for k, v in models.items() if v[2]]
    rows = []
    for e in NOVELTY:
        for seed in range(a.seeds):
            market = make_rel_market(RelMarketConfig(n_accounts=a.n_accounts, novelty=e),
                                     np.random.default_rng(seed))
            sa = similar_adopter_score(market)
            z = lambda v: (v - v.mean()) / v.std()  # noqa: E731
            X = np.column_stack([z(sa), z(np.log1p(market.degree))])
            for rho in RHOS:
                for n in NS:
                    rng = np.random.default_rng([seed, int(round(rho * 100)) + 1000, n])
                    founder = make_founder(market, FounderConfig(reliability=rho), rng)
                    out = run_outreach(market, founder, OutreachConfig(n_customers=n), rng)
                    U = np.setdiff1d(np.arange(market.n), out.contacted)
                    args = (market.segment, market.n_segments, market.graph_cov, founder.segment_logit,
                            out.contacted, out.outcome)
                    prior = founder.segment_logit[market.segment]
                    sc = {"prior_only": prior, "similar_adopters": sa,
                          "fusion": (prior - prior.mean()) / max(prior.std(), 1e-9) + z(sa),
                          "cpr_lp_rel": fit_predict(*args, CPRConfig(), extra_features=X).score,
                          # exploratory (added after pre-registration): main effects + relational regressors
                          "cpr_lp_hier_rel": fit_predict(*args, CPRConfig(),
                                                         extra_features=np.column_stack([X, np.eye(8)[market.industry], np.eye(4)[market.size]])).score,
                          "labels_only": fit_predict(*args, replace(CPRConfig(), use_founder=False)).score,
                          "cpr_lp": fit_predict(*args, CPRConfig()).score}
                    noise_models = {f"{k}_noise": v for k, v in models.items() if v[2]}
                    for name, (m, n_feat, uf) in {**models, **noise_models}.items():
                        relational = n_feat is not None and n_feat > 18
                        nr = np.random.default_rng([seed, 79, n]) if name.endswith("_noise") else None
                        x, ctx = tokens(market, founder, out, uf, relational, dev, nr)
                        if isinstance(m, PCRFMRel):
                            ad = torch.as_tensor(market.adoption.toarray(), dtype=torch.float32, device=dev)[None]
                            sm = torch.as_tensor(market.seller_similarity, dtype=torch.float32, device=dev)[None]
                            sc[name] = m.predict(x, ctx, ad, product_features(ad, sm)).cpu().numpy()
                        else:
                            sc[name] = m.predict(x, ctx).cpu().numpy()
                    er = np.random.default_rng([seed, 7])
                    row = {"novelty": e, "seed": seed, "rho": rho, "n": n}
                    for k in methods:
                        row[f"lift_{k}"] = lift_at_k(sc[k][U], market.p[U], TOP_K, er)
                        row[f"auc_{k}"] = expected_auc(sc[k][U], market.p[U])
                    rows.append(row)
            print(f"novelty {e} seed {seed} done", flush=True)

    ms = lambda v: f"{np.mean(v):.2f} ± {np.std(v, ddof=1) / np.sqrt(len(v)):.2f}"  # noqa: E731
    diffs = [("cpr_lp", "PC-RFM − CPR-LP"), ("cpr_lp_rel", "PC-RFM − CPR-LP-rel"), ("cpr_lp_hier_rel", "PC-RFM − CPR-LP-hier-rel"),
             ("fusion", "PC-RFM − fusion"), ("similar_adopters", "PC-RFM − adopters")]
    diffs += [(k, f"PC-RFM − {k}") for k in models if k != "pcrfm"]
    diffs += [("pcrfm_noise", "PC-RFM − same model, noise beliefs")]
    lines = [f"# PC-RFM v1 (relational) on sim_rel markets ({a.seeds} seeds, {a.n_accounts} accounts)", "",
             f"Expected lift@{TOP_K} on uncontacted accounts, mean ± SE over seeds; Δ columns paired.", ""]
    for e in NOVELTY:
        for rho in RHOS:
            hdr = ["n", *methods, *[d[1] for d in diffs]]
            lines += [f"### novelty = {e}, ρ = {rho}", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
            for n in NS:
                cell = [r for r in rows if r["novelty"] == e and r["rho"] == rho and r["n"] == n]
                vals = [ms([r[f"lift_{k}"] for r in cell]) for k in methods]
                vals += [ms([r["lift_pcrfm"] - r[f"lift_{k}"] for r in cell]) for k, _ in diffs]
                lines.append(f"| {n} | " + " | ".join(vals) + " |")
            lines.append("")
    md = "\n".join(lines)
    print(md)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md)
        Path(a.out).with_suffix(".json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
