"""H4 with a heavy-tailed segment prior: Student-t(nu = 3) segment deviations (EM scale mixture,
fprior.cprlp.robust_segment_scales) vs the Gaussian prior of scripts/explore_h4.py.

Pre-registered (written before the full run; only a 1-market timing check had been run):
  RB1. Once the hidden (wrongly believed, truly best) segment is contacted, its posterior error shrinks
       much faster with the robust prior: error at rounds 20 and 39 is closer to 0 than for the
       Gaussian version of the same policy, for greedy, thompson and thompson eta = 0.1.
  RB2. Washout (model ranks the hidden segment in its top 3) happens earlier and in more runs.
  RB3. No loss in customers found vs the Gaussian version of the same policy at round 40 (within
       2 SE); a gain is plausible because the hidden segment is the best one.
  Known limitation stated in advance: greedy never contacts the hidden segment, so a prior that only
  changes how evidence is weighed cannot help greedy (expect greedy-robust ~ greedy).

Same markets, founder and RNG seeds as scripts/explore_h4.py (10 seeds, 40 rounds x 25 contacts).
Gaussian policies are rerun here so every comparison is paired within this script.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fprior.explore import LoopConfig, pessimistic_founder, run_policy  # noqa: E402
from fprior.sim import MarketConfig, make_market  # noqa: E402

POLICIES = [("oracle", 0.0), ("greedy", 0.0), ("greedy-robust", 0.0), ("thompson", 0.0),
            ("thompson-robust", 0.0), ("thompson", 0.1), ("thompson-robust", 0.1)]
SEEDS, R = 10, 40


def main():
    cfg = LoopConfig(rounds=R)
    runs = []
    t0 = time.time()
    for seed in range(SEEDS):
        market = make_market(MarketConfig(), np.random.default_rng(seed))
        founder = pessimistic_founder(market)
        for pol, eta in POLICIES:
            log, hidden = run_policy(market, founder, pol, eta, cfg, np.random.default_rng([seed, 99]))
            runs.append(dict(seed=seed, policy=pol, eta=eta, hidden=hidden, log=log))
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)

    name = lambda p, e: p if e == 0 else f"{p} η={e}"  # noqa: E731
    ms = lambda v: f"{np.mean(v):+.2f} ± {np.std(v, ddof=1) / np.sqrt(len(v)):.2f}"  # noqa: E731
    orc = {r["seed"]: r for r in runs if r["policy"] == "oracle"}
    lines = [__doc__.strip(), "", f"## Results ({SEEDS} seeds, {R} rounds × {cfg.batch} contacts)", "",
             "| policy | customers / oracle, round 20 | round 40 | hidden contacts | error r10 | error r20 | error r39 | washout (median) | runs washed out |",
             "|---|---|---|---|---|---|---|---|---|"]
    for pol, eta in POLICIES:
        if pol == "oracle":
            continue
        rs = [r for r in runs if r["policy"] == pol and r["eta"] == eta]
        share = lambda t: np.array([r["log"][t]["customers"] / max(orc[r["seed"]]["log"][t]["customers"], 1) for r in rs])  # noqa: E731
        hc = np.array([r["log"][R]["hidden_contacts"] for r in rs])
        err = lambda t: np.array([r["log"][t]["hidden_err"] for r in rs])  # noqa: E731
        wo = [next((l["round"] for l in r["log"][:R] if 0 <= l["hidden_rank"] < 3), None) for r in rs]
        washed = [w for w in wo if w is not None]
        lines.append(f"| {name(pol, eta)} | {ms(share(20))} | {ms(share(R))} | {hc.mean():.1f} | {ms(err(10))} | "
                     f"{ms(err(20))} | {ms(err(R - 1))} | {int(np.median(washed)) if washed else '—'} | {len(washed)}/{len(rs)} |")
    lines += ["", "### Paired differences, robust − Gaussian (same policy, same seed)", "",
              "| policy | customers r40 (count) | error r20 | error r39 |", "|---|---|---|---|"]
    for pol, eta in [("greedy", 0.0), ("thompson", 0.0), ("thompson", 0.1)]:
        g = {r["seed"]: r for r in runs if r["policy"] == pol and r["eta"] == eta}
        b = {r["seed"]: r for r in runs if r["policy"] == pol + "-robust" and r["eta"] == eta}
        dc = [b[s]["log"][R]["customers"] - g[s]["log"][R]["customers"] for s in g]
        de = lambda t: [b[s]["log"][t]["hidden_err"] - g[s]["log"][t]["hidden_err"] for s in g]  # noqa: E731
        lines.append(f"| {name(pol, eta)} | {ms(dc)} | {ms(de(20))} | {ms(de(R - 1))} |")
    md = "\n".join(lines)
    print(md)
    (ROOT / "results/explore_h4_robust.md").write_text(md)
    (ROOT / "results/explore_h4_robust.json").write_text(json.dumps(runs))


if __name__ == "__main__":
    main()
