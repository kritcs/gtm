"""Paired comparison of one method across two real-tier runs on identical markets (same seeds, rho, n).

  python scripts/compare_real.py results/real_bank_v01.json results/real_bank_v01_ftbank.json [method] [metric]
method defaults to pcrfm, metric to lift. Prints per-(rho, n) and overall seed-level mean ± SE of (run2 − run1),
and also run2's method minus run2's cpr_lp.
"""

from __future__ import annotations

import json
import sys

import numpy as np


def main():
    f1, f2 = sys.argv[1], sys.argv[2]
    meth = sys.argv[3] if len(sys.argv) > 3 else "pcrfm"
    met = sys.argv[4] if len(sys.argv) > 4 else "lift"
    r1 = {(r["seed"], r["rho"], r["n"]): r for r in json.load(open(f1))}
    r2 = {(r["seed"], r["rho"], r["n"]): r for r in json.load(open(f2))}
    keys = sorted(set(r1) & set(r2))
    assert keys, "no matching cells"
    for k in keys[:50]:   # identical markets => identical CPR-LP scores
        assert abs(r1[k][f"{met}_cpr_lp"] - r2[k][f"{met}_cpr_lp"]) < 1e-9, "runs are not on identical markets"
    k_ = f"{met}_{meth}"

    def stat(sel, a, b):
        by = {}
        for k in keys:
            if sel(k):
                by.setdefault(k[0], []).append(a[k] - b[k])
        v = np.array([np.mean(x) for x in by.values()])
        return v.mean(), v.std(ddof=1) / np.sqrt(len(v))

    d12 = {k: r2[k][k_] - r1[k][k_] for k in keys}
    zero = {k: 0.0 for k in keys}
    gap = {k: r2[k][k_] - r2[k][f"{met}_cpr_lp"] for k in keys}
    rhos = sorted({k[1] for k in keys}, reverse=True)
    ns = sorted({k[2] for k in keys})
    print(f"{met}_{meth}: run2 − run1 (and run2 − CPR-LP), seed-level mean ± SE, {len({k[0] for k in keys})} seeds")
    print("| ρ | n | run2 − run1 | run2 − CPR-LP |\n|---|---|---|---|")
    for rho in rhos:
        for n in ns:
            a = stat(lambda k: k[1] == rho and k[2] == n, d12, zero)
            b = stat(lambda k: k[1] == rho and k[2] == n, gap, zero)
            print(f"| {rho} | {n} | {a[0]:+.3f} ± {a[1]:.3f} | {b[0]:+.3f} ± {b[1]:.3f} |")
    a = stat(lambda k: True, d12, zero)
    b = stat(lambda k: True, gap, zero)
    print(f"| all | all | {a[0]:+.3f} ± {a[1]:.3f} | {b[0]:+.3f} ± {b[1]:.3f} |")
    print(json.dumps({"overall_run2_minus_run1": a, "overall_run2_minus_cpr": b}))


if __name__ == "__main__":
    main()
