"""H4: without exploration, a confidently wrong belief about an avoided segment is never corrected.

Pre-registered (written before the full run):
  E1. greedy and Thompson with eta = 0 contact the hidden best segment ~never, and its posterior
      error stays near the founder's -3 logits through round 40.
  E2. with stratified exploration eta in {0.05, 0.1, 0.2} the hidden segment gets contacted, its error
      shrinks toward 0, and the washout time (first round the model ranks it in its top 3 segments)
      is finite and decreases as eta grows.
  E3. exploration costs customers early and the cost is recovered (or not) by round 40: report
      cumulative customers vs oracle per round; no prediction on the sign at round 40.

Setup: fprior.sim markets (3,000 accounts), founder exactly right except the truly best segment
(>= 20 accounts), believed 3 logits worse. Batches of 25 contacts, 40 rounds (1,000 contacts).
CPR-LP posterior with omega = 0.5; Thompson draws are per-account marginal.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fprior.explore import LoopConfig, pessimistic_founder, run_policy  # noqa: E402
from fprior.sim import MarketConfig, make_market  # noqa: E402

POLICIES = [("oracle", 0.0), ("founder", 0.0), ("greedy", 0.0), ("greedy", 0.1),
            ("thompson", 0.0), ("thompson", 0.05), ("thompson", 0.1), ("thompson", 0.2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--out", default="results/explore_h4.md")
    a = ap.parse_args()
    cfg = LoopConfig(rounds=a.rounds)
    runs = []
    t0 = time.time()
    for seed in range(a.seeds):
        market = make_market(MarketConfig(), np.random.default_rng(seed))
        founder = pessimistic_founder(market)
        for pol, eta in POLICIES:
            log, hidden = run_policy(market, founder, pol, eta, cfg, np.random.default_rng([seed, 99]))
            runs.append(dict(seed=seed, policy=pol, eta=eta, hidden=hidden, log=log))
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)

    def name(p, e):
        return p if e == 0 else f"{p} η={e}"

    def at(r, t, key):
        return r["log"][t][key]

    R = a.rounds
    checkpoints = sorted({t for t in (5, 10, 20, 30) if t < R} | {R})
    err_rounds = sorted({min(t, R - 1) for t in (10, 20, R - 1)})
    lines = [__doc__.strip().split("\n\n")[1].replace("Pre-registered", "**Pre-registered**"), "",
             f"{a.seeds} seeds, {R} rounds × {cfg.batch} contacts. Mean ± SE over seeds.", ""]
    # customers found (fraction of oracle)
    lines += ["## Customers found, as a share of the oracle's", "",
              "| policy | " + " | ".join(f"round {t}" for t in checkpoints) + " |",
              "|---|" + "---|" * len(checkpoints)]
    orc = {s: [r for r in runs if r["seed"] == s and r["policy"] == "oracle"][0] for s in range(a.seeds)}
    for pol, eta in POLICIES:
        rs = [r for r in runs if r["policy"] == pol and r["eta"] == eta]
        cells = []
        for t in checkpoints:
            v = np.array([at(r, t, "customers") / max(at(orc[r["seed"]], t, "customers"), 1) for r in rs])
            cells.append(f"{v.mean():.2f} ± {v.std(ddof=1) / np.sqrt(len(v)):.2f}")
        lines.append(f"| {name(pol, eta)} | " + " | ".join(cells) + " |")
    # hidden segment: contacts, error, rank, washout
    lines += ["", "## The wrongly believed (truly best) segment", "",
              "Error = model's mean logit for that segment minus the truth (founder starts at −3).",
              "Washout = first round the model ranks it in its top 3 segments (— = never within the horizon).", "",
              f"| policy | contacts by round {R} | " + " | ".join(f"error round {t}" for t in err_rounds)
              + f" | washout (median) | runs washed out | learned b, round {R - 1} |",
              "|---|---|" + "---|" * len(err_rounds) + "---|---|---|"]
    for pol, eta in POLICIES:
        if pol in ("oracle", "founder"):
            continue
        rs = [r for r in runs if r["policy"] == pol and r["eta"] == eta]
        hc = np.array([at(r, R, "hidden_contacts") for r in rs])
        errs = [np.array([at(r, t, "hidden_err") for r in rs]) for t in err_rounds]
        wo = []
        for r in rs:
            w = next((l["round"] for l in r["log"][:R] if 0 <= l["hidden_rank"] < 3), None)
            wo.append(w)
        washed = [w for w in wo if w is not None]
        b = np.array([at(r, R - 1, "trust_b") for r in rs])
        f = lambda v: f"{v.mean():+.2f} ± {v.std(ddof=1) / np.sqrt(len(v)):.2f}"  # noqa: E731
        lines.append(f"| {name(pol, eta)} | {hc.mean():.1f} ± {hc.std(ddof=1) / np.sqrt(len(hc)):.1f} | "
                     + " | ".join(f(e) for e in errs) + " | "
                     f"{(int(np.median(washed)) if washed else '—')} | {len(washed)}/{len(rs)} | {b.mean():.2f} |")
    md = "\n".join(lines)
    print(md)
    out = ROOT / a.out
    out.write_text(md)
    out.with_suffix(".json").write_text(json.dumps(runs))


if __name__ == "__main__":
    main()
