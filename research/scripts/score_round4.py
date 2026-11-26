"""Mechanical scoring of round-4 G1, G2, G5, G7 (results/round4_preregistration.md) from stress_sim JSONs.
Seed-level paired means over identical markets; thresholds on the point estimate (as in rounds 2–3)."""
import json, sys, numpy as np
def d(rows, shift, x, y, rhos=None, ns=None):
    c = {}
    for r in rows:
        if r["shift"] == shift and (rhos is None or r["rho"] in rhos) and (ns is None or r["n"] in ns):
            c.setdefault(r["seed"], []).append(r[f"lift_{x}"] - r[f"lift_{y}"])
    v = np.array([np.mean(z) for z in c.values()]); return v.mean(), v.std(ddof=1) / np.sqrt(len(v))
A = json.load(open("results/stress_sim_v0g.json")); S = json.load(open("results/stress_sim_v0g_s2.json"))
ns = sorted({r["n"] for r in A}); big = [n for n in ns if n >= 5]
print("n values:", ns, "| rho values:", sorted({r["rho"] for r in A}))
f = lambda t: f"{t[0]:+.3f} ± {t[1]:.3f}"
g1 = d(A, "S2_interaction", "pcrfm", "cpr_lp", [1.0, 0.5], big)
g2 = d(A, "S0_default", "pcrfm", "cpr_lp", [1.0, 0.5], big)
g1s2 = d(S, "S2_interaction", "pcrfm", "cpr_lp", [1.0, 0.5], big)
print(f"G1 S2 v0g − CPR-LP (> −0.05): {f(g1)} -> {'HOLDS' if g1[0] > -0.05 else 'FAILS'}")
print(f"G2 S0 v0g − CPR-LP (≥ −0.02): {f(g2)} -> {'HOLDS' if g2[0] >= -0.02 else 'FAILS'}")
for n in (0, 5):
    t = d(A, "S0_default", "pcrfm", "pcrfm_nofounder", [1.0], [n])
    print(f"G5 S0 ρ=1 n={n} v0g − v0g_nf (> 2 SE): {f(t)} -> {'HOLDS' if t[0] > 2 * t[1] else 'FAILS'}")
print(f"G7 seed-2 G1 metric {f(g1s2)}; |diff| = {abs(g1s2[0] - g1[0]):.3f} (≤ 0.05) -> {'HOLDS' if abs(g1s2[0] - g1[0]) <= 0.05 else 'FAILS'}")

# ---- G6: Bank R1–R3 for v0g (same rules as real_bank.py's pre-registration) ----
B = json.load(open("results/real_bank_v0g.json"))
def cell(rows, rho, n, x, y):
    v = np.array([r[f"lift_{x}"] - r[f"lift_{y}"] for r in rows if r["rho"] == rho and r["n"] == n])
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))
NS = [0, 1, 5, 10, 20]
r1 = [(m, n, cell(B, 1.0, n, m, "labels_only")) for m in ("cpr_lp", "pcrfm") for n in NS]
bad = [(m, n, t) for m, n, t in r1 if not t[0] > 2 * t[1]]
print("G6/R1 (ρ=1 beats labels-only > 2 SE, all n, both models):", "HOLDS" if not bad else "FAILS", [(m, n, f(t)) for m, n, t in bad])
nsb = sorted({r["n"] for r in B}); rhos = sorted({r["rho"] for r in B})
r2 = [(m, rho, n, cell(B, rho, n, m, "labels_only")) for m in ("cpr_lp", "pcrfm") for rho in (0.0, -0.5) for n in nsb if n >= 5]
bad2 = [(m, rho, n, f(t)) for m, rho, n, t in r2 if t[0] < -0.2]
print("G6/R2 (ρ≤0 loses ≤ 0.2 to labels-only, n ≥ 5):", "HOLDS" if not bad2 else "FAILS", bad2)
r3 = [(rho, n, cell(B, rho, n, "pcrfm", "cpr_lp")) for rho in rhos for n in nsb]
ok = sum(t[0] >= -0.2 for _, _, t in r3)
print(f"G6/R3 (PC-RFM within 0.2 of CPR-LP in ≥ 2/3 of cells): {ok}/{len(r3)} ->", "HOLDS" if ok >= 2 / 3 * len(r3) else "FAILS",
      "| worst", min((t[0], rho, n) for rho, n, t in r3))
# exploratory: compare to v0.1 on identical markets
V = json.load(open("results/real_bank_v01.json"))
key = lambda r: (r["seed"], r["rho"], r["n"])
vm = {key(r): r for r in V}
pair = [(r, vm[key(r)]) for r in B if key(r) in vm]
same = all(abs(a["lift_cpr_lp"] - b["lift_cpr_lp"]) < 1e-9 for a, b in pair)
dd = {}
for a, b in pair: dd.setdefault(a["seed"], []).append(a["lift_pcrfm"] - b["lift_pcrfm"])
v = np.array([np.mean(z) for z in dd.values()])
print(f"exploratory: identical markets {same}; v0g − v0.1 on Bank overall {v.mean():+.3f} ± {v.std(ddof=1)/np.sqrt(len(v)):.3f}")
for name, R in (("v0g_nf", B), ("v01_nf", V)):
    x = np.array([r["lift_pcrfm_nofounder"] for r in R if r["n"] == 0])
    print(f"exploratory: {name} lift at n=0 (no founder, no labels) {x.mean():.2f} ± {x.std(ddof=1)/np.sqrt(len(x)):.2f}")
