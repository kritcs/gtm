"""Evaluate PC-RFM checkpoints on the M0 sanity-ladder markets (fprior.sim, with graph).

Uses the same seeds, founders and outreach as scripts/sanity_ladder.py, so the numbers
are directly comparable with prior-only, labels-only and CPR-LP (recomputed here with the
current CPR-LP code). Note v0 cannot see the graph; CPR-LP can.

  python scripts/eval_pfn.py --ckpt runs/pfn_v0/model.pt --ckpt-nf runs/pfn_v0_nf/model.pt
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
from fprior.metrics import expected_auc, lift_at_k  # noqa: E402
from fprior.pfn.data import graph_features, make_tokens  # noqa: E402
from fprior.pfn.model import PCRFM  # noqa: E402
from fprior.pfn.model_tok import PCRFMTok  # noqa: E402
from fprior.pfn.data import founder_token_features  # noqa: E402
from fprior.sim import (FounderConfig, MarketConfig, OutreachConfig, make_founder,  # noqa: E402
                        make_market, run_outreach)

RHOS = [1.0, 0.75, 0.5, 0.25, 0.0, -0.5]
NS = [0, 1, 5, 10, 20, 50]
TOP_K = 100
STATED_SD = 0.7   # neutral stated uncertainty for founders who state none (token models)


def load(path, dev):
    ck = torch.load(path, map_location=dev)
    m = (PCRFMTok if ck["arch"].get("tokens") else PCRFM)(**ck["arch"]).to(dev).eval()
    m.load_state_dict(ck["state"])
    m.graph = bool(ck.get("graph", False))
    return m


def pfn_scores(model, market, founder, out, use_founder, dev):
    t = lambda a, dt=torch.long: torch.as_tensor(a, dtype=dt, device=dev)[None]  # noqa: E731
    ctx = np.zeros(market.n, dtype=bool)
    ctx[out.contacted] = True
    outcome = np.zeros(market.n)
    outcome[out.contacted] = out.outcome
    if isinstance(model, PCRFMTok):
        # belief lives in founder tokens; this founder states no confidence -> neutral stated sd
        x = make_tokens(t(market.industry), t(market.size),
                        t(founder.segment_logit[market.segment], torch.float32),
                        torch.tensor([0.0], device=dev), t(ctx, torch.bool), t(outcome, torch.float32))
        fl = torch.as_tensor(founder.segment_logit, dtype=torch.float32, device=dev)[None]
        ftok, fmask = founder_token_features(fl, torch.full_like(fl, STATED_SD), torch.zeros_like(fl, dtype=torch.bool),
                                             torch.tensor([1.0 if use_founder else 0.0], device=dev))
        return model.predict(x, t(ctx, torch.bool), ftok, fmask).cpu().numpy()
    x = make_tokens(t(market.industry), t(market.size),
                    t(founder.segment_logit[market.segment], torch.float32),
                    torch.tensor([1.0 if use_founder else 0.0], device=dev),
                    t(ctx, torch.bool), t(outcome, torch.float32))
    if getattr(model, "graph", False):   # v0g: neighbour-label features; markets without a graph get degree 0
        A = getattr(market, "A", None)
        A = torch.zeros(market.n, market.n) if A is None else torch.as_tensor(A.toarray(), dtype=torch.float32)
        x = torch.cat([x, graph_features(A.to(dev)[None], t(ctx, torch.bool), t(outcome, torch.float32))], -1)
    return model.predict(x, t(ctx, torch.bool)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-nf", default=None, help="ablation trained without founder inputs")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-accounts", type=int, default=3000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = a.device
    models = {"pcrfm": (load(a.ckpt, dev), True)}
    if a.ckpt_nf:
        models["pcrfm_nofounder"] = (load(a.ckpt_nf, dev), False)
    methods = ["prior_only", "labels_only", "cpr_lp", *models]
    rows = []
    for seed in range(a.seeds):
        market = make_market(MarketConfig(n_accounts=a.n_accounts), np.random.default_rng(seed))
        for rho in RHOS:
            for n in NS:
                rng = np.random.default_rng([seed, int(round(rho * 100)) + 1000, n])
                founder = make_founder(market, FounderConfig(reliability=rho), rng)
                out = run_outreach(market, founder, OutreachConfig(n_customers=n), rng)
                U = np.setdiff1d(np.arange(market.n), out.contacted)
                args = (market.segment, market.n_segments, market.graph_cov, founder.segment_logit,
                        out.contacted, out.outcome)
                sc = {"prior_only": founder.segment_logit[market.segment],
                      "labels_only": fit_predict(*args, replace(CPRConfig(), use_founder=False)).score,
                      "cpr_lp": fit_predict(*args, CPRConfig()).score}
                for name, (m, uf) in models.items():
                    sc[name] = pfn_scores(m, market, founder, out, uf, dev)
                er = np.random.default_rng([seed, 7])
                row = {"seed": seed, "rho": rho, "n": n}
                for k in methods:
                    row[f"lift_{k}"] = lift_at_k(sc[k][U], market.p[U], TOP_K, er)
                    row[f"auc_{k}"] = expected_auc(sc[k][U], market.p[U])
                rows.append(row)
        print(f"seed {seed} done", flush=True)

    lines = [f"# PC-RFM v0 on sanity-ladder markets ({a.seeds} seeds, {a.n_accounts} accounts)", "",
             f"Expected lift@{TOP_K} on uncontacted accounts, mean ± SE over seeds. "
             "Δ columns are paired differences.", ""]
    for rho in RHOS:
        hdr = ["n", *methods, "PC-RFM − CPR-LP", "PC-RFM − labels-only"]
        if a.ckpt_nf:
            hdr.append("PC-RFM − no-founder")
        lines += [f"### ρ = {rho}", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for n in NS:
            cell = [r for r in rows if r["rho"] == rho and r["n"] == n]
            ms = lambda v: f"{np.mean(v):.2f} ± {np.std(v, ddof=1) / np.sqrt(len(v)):.2f}"  # noqa: E731
            vals = [ms([r[f"lift_{k}"] for r in cell]) for k in methods]
            d = lambda o: ms([r["lift_pcrfm"] - r[f"lift_{o}"] for r in cell])  # noqa: E731
            vals += [d("cpr_lp"), d("labels_only")]
            if a.ckpt_nf:
                vals.append(d("pcrfm_nofounder"))
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
