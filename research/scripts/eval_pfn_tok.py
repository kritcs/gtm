"""Does PC-RFM-tok use what founders *say* about their own confidence? (pre-registered)

Markets, founders and outreach are the M0 sanity-ladder ones (fprior.sim), as in eval_pfn.py. The belief values
are identical across conditions; only the per-segment statements change:

  neutral     stated sd 0.7 everywhere, no "no idea"   (what eval_pfn.py feeds the token model)
  calibrated  stated sd = 0.5·rms(err) + 0.5·|err_k|   (calibration 1, awareness 0.5, as in the training prior)
  inverted    the calibrated sds re-assigned in reverse rank order (confident where wrong, unsure where right)
  noidea_worst  "no idea" on the 25% of segments with the largest |err_k|; their belief set to the intercept
  noidea_rand   "no idea" on a random 25% of segments, same handling
(err_k = founder belief − true segment logit.) ρ ∈ {0.75, 0.5, 0.25} (ρ = 1 has err ≡ 0); n ∈ {0, 1, 5, 10, 20}.
Metric: expected lift@100 on uncontacted accounts, paired over 10 seeds.

Predictions (written before the model finished training; rule: paired mean difference has the predicted sign and
exceeds 2 SE in at least half of the named cells):
  T1 calibrated − neutral > 0 at n ∈ {0, 1, 5}, all three ρ       (stated confidence is used)
  T2 calibrated − inverted > 0 at n ∈ {0, 1, 5}, all three ρ       (it is used in the right direction)
  T3 noidea_worst − noidea_rand > 0 at n ∈ {0, 1, 5}, all three ρ  ("no idea" is used as missing, not as belief)
  T4 the advantages shrink with labels: mean(T1 gap at n = 20) < mean(T1 gap at n ∈ {0, 1})
  T5 no regression from the token interface: tok-neutral − v0 ≥ −0.3 in every cell (only if --ckpt-v0 given;
     note v0 was trained without --random-seg, so a difference may be the training prior, not the tokens)
If T1 fails while T3 holds, the stated-sd feature is ignored; report that rather than retrain silently.
Disclosure: after these predictions were written, a 2-seed smoke test on the step-5500 sync checkpoint was run
only to check the code; predictions were not changed.

  python scripts/eval_pfn_tok.py --ckpt runs/pfn_tok/model.pt --ckpt-v0 runs/pfn_v0/model.pt --out results/m2_pfn_tok.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fprior.metrics import lift_at_k  # noqa: E402
from fprior.pfn.data import founder_token_features, make_tokens  # noqa: E402
from fprior.pfn.model_tok import PCRFMTok  # noqa: E402
from fprior.sim import (FounderConfig, MarketConfig, OutreachConfig, make_founder,  # noqa: E402
                        make_market, run_outreach)

sys.path.insert(0, str(ROOT / "scripts"))
from eval_pfn import load, pfn_scores  # noqa: E402

RHOS = [0.75, 0.5, 0.25]
NS = [0, 1, 5, 10, 20]
TOP_K = 100
CONDS = ["neutral", "calibrated", "inverted", "noidea_worst", "noidea_rand"]


def statements(market, founder, cond, rng):
    fl = founder.segment_logit.copy()
    err = fl - market.segment_logit
    rms = np.sqrt(np.mean(err ** 2))
    cal = 0.5 * rms + 0.5 * np.maximum(np.abs(err), 0.05)
    K = len(fl)
    noidea = np.zeros(K, dtype=bool)
    sd = np.full(K, 0.7)
    if cond == "calibrated":
        sd = cal
    elif cond == "inverted":
        order = np.argsort(np.abs(err))
        sd = np.empty(K)
        sd[order] = np.sort(cal)[::-1]
    elif cond.startswith("noidea"):
        m = K // 4
        idx = np.argsort(-np.abs(err))[:m] if cond == "noidea_worst" else rng.choice(K, m, replace=False)
        noidea[idx] = True
        fl[idx] = market.intercept
    return fl, sd, noidea


def tok_scores(model, market, fl, sd, noidea, out, dev):
    t = lambda a, dt=torch.long: torch.as_tensor(a, dtype=dt, device=dev)[None]  # noqa: E731
    ctx = np.zeros(market.n, dtype=bool)
    ctx[out.contacted] = True
    outcome = np.zeros(market.n)
    outcome[out.contacted] = out.outcome
    x = make_tokens(t(market.industry), t(market.size), t(fl[market.segment], torch.float32),
                    torch.tensor([0.0], device=dev), t(ctx, torch.bool), t(outcome, torch.float32))
    ftok, fmask = founder_token_features(t(fl, torch.float32), t(sd, torch.float32), t(noidea, torch.bool),
                                         torch.tensor([1.0], device=dev))
    return model.predict(x, t(ctx, torch.bool), ftok, fmask).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-v0", default=None)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-accounts", type=int, default=3000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = a.device
    tok = load(a.ckpt, dev)
    assert isinstance(tok, PCRFMTok), "checkpoint is not a founder-token model"
    v0 = load(a.ckpt_v0, dev) if a.ckpt_v0 else None
    rows = []
    for seed in range(a.seeds):
        market = make_market(MarketConfig(n_accounts=a.n_accounts), np.random.default_rng(seed))
        for rho in RHOS:
            for n in NS:
                rng = np.random.default_rng([seed, int(round(rho * 100)) + 1000, n])
                founder = make_founder(market, FounderConfig(reliability=rho), rng)
                out = run_outreach(market, founder, OutreachConfig(n_customers=n), rng)
                U = np.setdiff1d(np.arange(market.n), out.contacted)
                er = np.random.default_rng([seed, 7])
                row = {"seed": seed, "rho": rho, "n": n}
                for cond in CONDS:
                    fl, sd, noidea = statements(market, founder, cond, np.random.default_rng([seed, n, 11]))
                    s = tok_scores(tok, market, fl, sd, noidea, out, dev)
                    row[f"lift_{cond}"] = lift_at_k(s[U], market.p[U], TOP_K, er)
                if v0 is not None:
                    row["lift_v0"] = lift_at_k(pfn_scores(v0, market, founder, out, True, dev)[U], market.p[U], TOP_K, er)
                rows.append(row)
        print(f"seed {seed} done", flush=True)

    ms = lambda v: f"{np.mean(v):+.2f} ± {np.std(v, ddof=1) / np.sqrt(len(v)):.2f}"  # noqa: E731
    pairs = [("calibrated", "neutral"), ("calibrated", "inverted"), ("noidea_worst", "noidea_rand")]
    if v0 is not None:
        pairs.append(("neutral", "v0"))
    lines = [f"# PC-RFM-tok: does it use stated confidence? ({a.seeds} seeds, {a.n_accounts} accounts)", "",
             f"Expected lift@{TOP_K} on uncontacted accounts. Paired differences, mean ± SE over seeds.", ""]
    for rho in RHOS:
        hdr = ["n", *[f"{c}" for c in CONDS], *[f"{x} − {y}" for x, y in pairs]]
        lines += [f"### ρ = {rho}", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for n in NS:
            cell = [r for r in rows if r["rho"] == rho and r["n"] == n]
            vals = [f"{np.mean([r[f'lift_{c}'] for r in cell]):.2f}" for c in CONDS]
            vals += [ms([r[f"lift_{x}"] - r[f"lift_{y}"] for r in cell]) for x, y in pairs]
            lines.append(f"| {n} | " + " | ".join(vals) + " |")
        lines.append("")

    def cellstat(rho, n, x, y):
        d = np.array([r[f"lift_{x}"] - r[f"lift_{y}"] for r in rows if r["rho"] == rho and r["n"] == n])
        return d.mean(), d.std(ddof=1) / np.sqrt(len(d))

    def frac(x, y):
        c = [cellstat(rho, n, x, y) for rho in RHOS for n in (0, 1, 5)]
        k = sum(m > 2 * s for m, s in c)
        return k * 2 >= len(c), f"{k}/{len(c)} cells > 2 SE"

    score = [("T1 stated confidence used", *frac("calibrated", "neutral")),
             ("T2 used in the right direction", *frac("calibrated", "inverted")),
             ("T3 'no idea' treated as missing", *frac("noidea_worst", "noidea_rand"))]
    g_lo = np.mean([cellstat(rho, n, "calibrated", "neutral")[0] for rho in RHOS for n in (0, 1)])
    g_hi = np.mean([cellstat(rho, 20, "calibrated", "neutral")[0] for rho in RHOS])
    score.append(("T4 advantage shrinks with labels", g_hi < g_lo, f"gap n∈{{0,1}} {g_lo:+.2f} vs n=20 {g_hi:+.2f}"))
    if v0 is not None:
        worst = min(cellstat(rho, n, "neutral", "v0")[0] for rho in RHOS for n in NS)
        score.append(("T5 no regression vs v0", worst >= -0.3, f"worst cell {worst:+.2f} (margin −0.3)"))
    lines += ["## Mechanical scoring", ""] + [f"- {'HOLDS' if h else 'FAILS'} — {nm}: {d}" for nm, h, d in score]
    md = "\n".join(lines)
    print(md)
    if a.out:
        Path(a.out).write_text(md + "\n")
        Path(a.out).with_suffix(".json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
