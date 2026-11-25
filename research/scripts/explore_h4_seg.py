"""H4 with segment-level Thompson sampling (the textbook version), vs our account-level sampler.

Pre-registered (committed before running; from docs/theory-h4.md P2a/P2b):
  TS1. Segment-level Thompson with η = 0 (Gaussian prior) contacts the hidden best segment in at most 2 of
       10 seeds within 40 rounds, and its error at round 39 stays below −2.0 logits (stuck, as P2a predicts),
       unlike account-level Thompson (5/10 washed out, error −1.24, results/explore_h4.md).
  TS2. Adding stratified η = 0.1 and the Student-t prior to segment-level Thompson gets the hidden segment
       contacted in more seeds than η = 0 Gaussian.
Same markets, founder and seeds as scripts/explore_h4.py.
"""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from fprior.explore import LoopConfig, pessimistic_founder, run_policy
from fprior.sim import MarketConfig, make_market
POLICIES = [("thompsonseg", 0.0), ("thompsonseg", 0.1), ("thompsonseg-robust", 0.0), ("thompsonseg-robust", 0.1)]
SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
cfg = LoopConfig(rounds=40); runs = []; t0 = time.time()
for seed in range(SEEDS):
    m = make_market(MarketConfig(), np.random.default_rng(seed)); f = pessimistic_founder(m)
    for pol, eta in POLICIES:
        log, hidden = run_policy(m, f, pol, eta, cfg, np.random.default_rng([seed, 99]))
        runs.append(dict(seed=seed, policy=pol, eta=eta, log=log))
    print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
R = cfg.rounds
lines = [__doc__.strip(), "", f"## Results ({SEEDS} seeds, {R} rounds)", "",
         "| policy | seeds that ever contact hidden segment | hidden contacts by r40 | error r39 | washed out (top 3) |",
         "|---|---|---|---|---|"]
for pol, eta in POLICIES:
    rs = [r for r in runs if r["policy"] == pol and r["eta"] == eta]
    hc = np.array([r["log"][R]["hidden_contacts"] for r in rs]); err = np.array([r["log"][R - 1]["hidden_err"] for r in rs])
    wo = sum(any(0 <= l["hidden_rank"] < 3 for l in r["log"][:R]) for r in rs)
    lines.append(f"| {pol} η={eta} | {(hc > 0).sum()}/{len(rs)} | {hc.mean():.1f} ± {hc.std(ddof=1) / np.sqrt(len(hc)):.1f} | "
                 f"{err.mean():+.2f} ± {err.std(ddof=1) / np.sqrt(len(err)):.2f} | {wo}/{len(rs)} |")
md = "\n".join(lines); print(md)
(ROOT / "results/explore_h4_seg.md").write_text(md); (ROOT / "results/explore_h4_seg.json").write_text(json.dumps(runs))
