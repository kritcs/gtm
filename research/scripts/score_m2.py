"""Mechanically score the M2 pre-registration (results/m2_preregistration.md) from eval_pfn_rel.py outputs.

Rule (from the pre-registration): a prediction holds if the paired mean difference has the predicted sign and
exceeds 2 SE in at least half of the named cells; otherwise it fails. P3 and P6 use their own stated margins.
Written before any v1/v2 results existed.

  python scripts/score_m2.py results/m2_pfn_v1.json results/m2_pfn_v2.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def cell(rows, e, rho, n, a, b):
    d = np.array([r[f"lift_{a}"] - r[f"lift_{b}"] for r in rows
                  if r["novelty"] == e and r["rho"] == rho and r["n"] == n])
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def frac(cells, test):
    ok = [test(m, s) for m, s in cells]
    return sum(ok), len(ok)


def main():
    v1 = json.loads(Path(sys.argv[1]).read_text())
    v2 = json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv) > 2 else None
    out = []

    # P1: v1 − v1-nf > 0 for ρ ∈ {1, 0.5}, n ∈ {0,1,5,10,20}; and the gap at e = 1 larger than at e = 0
    cells = [cell(v1, e, rho, n, "pcrfm", "pcrfm_nofounder") for e in (0.0, 0.5, 1.0) for rho in (1.0, 0.5) for n in (0, 1, 5, 10, 20)]
    k, t = frac(cells, lambda m, s: m > 2 * s)
    g = {e: np.mean([cell(v1, e, rho, n, "pcrfm", "pcrfm_nofounder")[0] for rho in (1.0, 0.5) for n in (0, 1, 5, 10, 20)]) for e in (0.0, 1.0)}
    out.append(("P1 founder value", k * 2 >= t and g[1.0] > g[0.0], f"{k}/{t} cells > 2 SE; mean gap e=1 {g[1.0]:+.2f} vs e=0 {g[0.0]:+.2f}"))

    # P2: v1 − v0 > 0 at e = 0 for every ρ and n ≥ 1; |v1 − v0| < 2 SE at e = 1
    if any("lift_pcrfm_v0" in r for r in v1):
        c0 = [cell(v1, 0.0, rho, n, "pcrfm", "pcrfm_v0") for rho in (1.0, 0.5, 0.0, -0.5) for n in (1, 5, 10, 20, 50)]
        c1 = [cell(v1, 1.0, rho, n, "pcrfm", "pcrfm_v0") for rho in (1.0, 0.5, 0.0, -0.5) for n in (1, 5, 10, 20, 50)]
        k0, t0 = frac(c0, lambda m, s: m > 2 * s)
        k1, t1 = frac(c1, lambda m, s: abs(m) < 2 * s)
        out.append(("P2 relational value", k0 * 2 >= t0 and k1 * 2 >= t1, f"e=0: {k0}/{t0} > 2 SE; e=1: {k1}/{t1} within 2 SE"))

    # P3: ρ ∈ {0, −0.5}, n ≥ 5: v1 − v1-nf ≥ −0.2 (all novelty levels)
    cells = [cell(v1, e, rho, n, "pcrfm", "pcrfm_nofounder") for e in (0.0, 0.5, 1.0) for rho in (0.0, -0.5) for n in (5, 10, 20, 50)]
    worst = min(m for m, _ in cells)
    out.append(("P3 bounded harm", worst >= -0.2, f"worst cell {worst:+.2f} (margin −0.2)"))

    # P4: e = 1, ρ = 1: v1 − fusion > 0
    cells = [cell(v1, 1.0, 1.0, n, "pcrfm", "fusion") for n in (0, 1, 5, 10, 20, 50)]
    k, t = frac(cells, lambda m, s: m > 2 * s)
    out.append(("P4 beats fixed fusion at e=1", k * 2 >= t, f"{k}/{t} cells > 2 SE"))

    # P5: v2 − v1 > 0 at e = 0 for n ∈ {10, 20, 50}; |v2 − v1| < 2 SE at e = 1
    if v2 is not None and any("lift_pcrfm_v1" in r for r in v2):
        c0 = [cell(v2, 0.0, rho, n, "pcrfm", "pcrfm_v1") for rho in (1.0, 0.5, 0.0, -0.5) for n in (10, 20, 50)]
        c1 = [cell(v2, 1.0, rho, n, "pcrfm", "pcrfm_v1") for rho in (1.0, 0.5, 0.0, -0.5) for n in (10, 20, 50)]
        k0, t0 = frac(c0, lambda m, s: m > 2 * s)
        k1, t1 = frac(c1, lambda m, s: abs(m) < 2 * s)
        out.append(("P5 learned encoder", k0 * 2 >= t0 and k1 * 2 >= t1, f"e=0: {k0}/{t0} > 2 SE; e=1: {k1}/{t1} within 2 SE"))

    # P6: v1 within 0.2 of, or better than, the best of {CPR-LP, CPR-LP-rel} in ≥ 2/3 of all cells
    cells = []
    for e in (0.0, 0.5, 1.0):
        for rho in (1.0, 0.5, 0.0, -0.5):
            for n in (0, 1, 5, 10, 20, 50):
                rs = [r for r in v1 if r["novelty"] == e and r["rho"] == rho and r["n"] == n]
                best = max(np.mean([r["lift_cpr_lp"] for r in rs]), np.mean([r["lift_cpr_lp_rel"] for r in rs]))
                cells.append(np.mean([r["lift_pcrfm"] for r in rs]) >= best - 0.2)
    out.append(("P6 vs principled baseline", sum(cells) * 3 >= 2 * len(cells), f"{sum(cells)}/{len(cells)} cells within 0.2 or better"))

    for name, held, detail in out:
        print(f"{'HOLDS' if held else 'FAILS'} — {name}: {detail}")


if __name__ == "__main__":
    main()
