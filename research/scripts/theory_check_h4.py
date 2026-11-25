"""Numerical checks for docs/theory-h4.md (P1–P5).

Part A checks each formula against a simulation of the simplified model it was derived for.
Part B plugs our H4 setting into the formulas and compares with results/explore_h4*.json.
Writes results/theory_check_h4.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import integrate, stats
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

out = []


def p(s=""):
    print(s)
    out.append(s)


# ---------------------------------------------------------------- formulas

def gauss_post_mean_err(delta, n, sigma2, omega2):
    """P4 (Gaussian): E[posterior mean − θ] after n obs when the prior mean is θ − Δ."""
    return -delta * sigma2 / (sigma2 + n * omega2)


def t_post_mean(ybar, n, sigma2, m, omega, nu):
    """Exact posterior mean of θ under prior t_ν(m, ω) and likelihood ybar ~ N(θ, σ²/n) (1-D quadrature)."""
    s = np.sqrt(sigma2 / n)
    lo, hi = min(m, ybar) - 12 * max(s, omega) - 10, max(m, ybar) + 12 * max(s, omega) + 10
    f = lambda th: stats.t.pdf((th - m) / omega, nu) * stats.norm.pdf(ybar, th, s)  # noqa: E731
    z = integrate.quad(f, lo, hi, points=[m, ybar], limit=400)[0]
    return integrate.quad(lambda th: th * f(th), lo, hi, points=[m, ybar], limit=400)[0] / z


def t_bias_asym(delta, n, sigma2, omega, nu):
    """P4 (Student-t), Tweedie + prior-tail approximation, valid when σ²/n ≪ ω² and Δ ≫ ω:
    bias ≈ −(σ²/n)(ν+1)Δ / (νω² + Δ²)."""
    return -(sigma2 / n) * (nu + 1) * delta / (nu * omega ** 2 + delta ** 2)


# ---------------------------------------------------------------- Part A: simplified-model checks

rng = np.random.default_rng(0)
p("# Numerical checks for docs/theory-h4.md")
p("")
p("## Part A — formulas vs simulations of the simplified model")
p("")

# P1: greedy, independent Gaussian segments, founder right except k*.
K, B, T, S, omega, sigma = 8, 5, 200, 400, 0.5, 2.0
theta = np.linspace(0, 1.4, K)            # k* = K-1 is best
m = theta.copy()
delta = 3.0
m[-1] -= delta
gamma = theta[-2] - m[-1]                  # margin of the second-best segment over the wrong prior mean
never, regret_slope = 0, []
for rep in range(200):
    n = np.zeros(K)
    s = np.zeros(K)
    left = np.full(K, S)
    reg = 0.0
    for t in range(T):
        mu = (m / omega ** 2 + s / sigma ** 2) / (1 / omega ** 2 + n / sigma ** 2)
        for _ in range(B):
            k = int(np.argmax(np.where(left > 0, mu, -np.inf)))
            yv = theta[k] + sigma * rng.standard_normal()
            n[k] += 1; s[k] += yv; left[k] -= 1
            reg += theta[-1] - theta[k]
            mu[k] = (m[k] / omega ** 2 + s[k] / sigma ** 2) / (1 / omega ** 2 + n[k] / sigma ** 2)
    never += n[-1] == 0
    regret_slope.append(reg / (T * B))
bound = (T * B) * np.exp(-2 * gamma ** 2 / omega ** 2)
p(f"**P1 (greedy).** K={K}, B={B}, T={T}, ω={omega}, σ={sigma}, Δ={delta}, margin γ={gamma:.2f}. "
  f"k* never sampled in {never}/200 runs; P1 bound on the failure probability for the runner-up segment alone: "
  f"{min(bound, 1):.2e} (union over TB values of n). Mean per-contact regret {np.mean(regret_slope):.3f} "
  f"(≥ θ_k* − θ_runner-up = {theta[-1] - theta[-2]:.3f} whenever k* is not sampled; larger here because greedy exhausts the runner-up segment and moves down) → linear regret.")
p("")

# P2: segment-level TS vs account-level marginal TS: probability k* is drawn in a round.
p("**P2 (Thompson escape).** One round, prior only. Segment-level TS (one draw per segment, top-1 segment): "
  "P(k* chosen) vs Φ(−z)-type bound. Account-level marginal TS (one draw per account, top-B accounts, as in "
  "fprior/explore.py): expected k* accounts in the batch vs the order-statistic formula.")
for omega_s, gap in [(0.5, 1.5), (0.5, 2.5), (1.0, 2.5)]:
    # two segments: best wrongly believed at -gap relative to runner-up; both prior sd omega_s
    draws = rng.standard_normal((200000, 2)) * omega_s
    sim = np.mean(draws[:, 1] - gap > draws[:, 0])
    z = gap / (np.sqrt(2) * omega_s)
    p(f"- segment TS, ω={omega_s}, gap={gap}: simulated {sim:.2e}, Φ(−z) = {stats.norm.sf(z):.2e} (z = {z:.2f}), "
      f"expected first-sampling round 1/Φ(−z) = {1 / max(stats.norm.sf(z), 1e-300):.0f}")
# account-level: N_other accounts with mean 0, N_h hidden accounts with mean -gap, all with sd s_acc
N_other, N_h, Bacc = 2900, 95, 25
for s_acc, gap in [(0.6, 2.5), (1.3, 2.5), (1.3, 1.5)]:
    hits = []
    for _ in range(4000):
        a = rng.standard_normal(N_other) * s_acc
        h = rng.standard_normal(N_h) * s_acc - gap
        thr = np.partition(np.r_[a, h], -Bacc)[-Bacc]
        hits.append(np.sum(h >= thr))
    # formula: τ solves N_other Φ̄(τ/s) + N_h Φ̄((τ+gap)/s) = B; expected hidden = N_h Φ̄((τ+gap)/s)
    from scipy.optimize import brentq
    tau = brentq(lambda t: N_other * stats.norm.sf(t / s_acc) + N_h * stats.norm.sf((t + gap) / s_acc) - Bacc, -10, 10)
    pred = N_h * stats.norm.sf((tau + gap) / s_acc)
    p(f"- account TS, per-account sd={s_acc}, gap={gap}: simulated E[#k* accounts/round] = {np.mean(hits):.3f}, "
      f"formula = {pred:.3f}; P(≥1) simulated {np.mean(np.array(hits) > 0):.3f}, Poisson approx {1 - np.exp(-pred):.3f}")
p("")

# P4: Gaussian vs Student-t posterior bias as a function of n and Δ (exact quadrature vs formulas).
sigma2, omega, nu = 5.0, 0.5, 3
p(f"**P4 (washout).** σ²={sigma2} per observation (logit scale), ω={omega}, ν={nu}. Expected posterior-mean error "
  "(averaged over ȳ ~ N(θ, σ²/n)) — Gaussian formula vs exact; Student-t exact vs asymptotic formula.")
p("")
p("| Δ | n | Gaussian formula | Gaussian exact (sim) | Student-t exact | Student-t asymptotic |")
p("|---|---|---|---|---|---|")
for D in (1.0, 3.0, 6.0):
    for n in (10, 40, 160):
        ys = 0 + np.sqrt(sigma2 / n) * rng.standard_normal(400)
        g_exact = np.mean([(-D / omega ** 2 + n * y / sigma2) / (1 / omega ** 2 + n / sigma2) for y in ys])
        t_exact = np.mean([t_post_mean(y, n, sigma2, -D, omega, nu) for y in ys[:120]])
        p(f"| {D} | {n} | {gauss_post_mean_err(D, n, sigma2, omega ** 2):+.3f} | {g_exact:+.3f} | {t_exact:+.3f} | "
          f"{t_bias_asym(D, n, sigma2, omega, nu):+.3f} |")
p("")

# ---------------------------------------------------------------- Part B: our H4 setting

p("## Part B — our H4 runs (results/explore_h4.json, explore_h4_robust.json)")
p("")
runs = json.load(open(ROOT / "results/explore_h4.json"))
robust_path = ROOT / "results/explore_h4_robust.json"
runs_r = json.load(open(robust_path)) if robust_path.exists() else []

from fprior.cprlp import CPRConfig  # noqa: E402
from fprior.explore import pessimistic_founder, posterior  # noqa: E402
from fprior.sim import MarketConfig, make_market  # noqa: E402

cfg = CPRConfig()
s_e2 = cfg.graph_sd ** 2 + cfg.nugget_sd ** 2   # account-level variance around the segment logit
rows = []
for seed in range(10):
    market = make_market(MarketConfig(), np.random.default_rng(seed))
    founder = pessimistic_founder(market)
    hidden = int(np.argmax(founder.segment_logit - market.segment_logit < -1.0))
    mean, var, _ = posterior(market, founder.segment_logit, np.array([], int), np.array([]), cfg, 0.5)
    # P2 at round 0 with the real prior means/variances
    from scipy.optimize import brentq
    sd = np.sqrt(var)
    tau = brentq(lambda t: stats.norm.sf((t - mean) / sd).sum() - 25, -20, 20)
    hmask = market.segment == hidden
    lam0 = stats.norm.sf((tau - mean[hmask]) / sd[hmask]).sum()
    ph = expit(market.segment_logit[hidden])
    sigma2_eff = 1 / (ph * (1 - ph)) + s_e2      # info per contact about the segment logit
    g = [r for r in runs if r["seed"] == seed and r["policy"] == "greedy" and r["eta"] == 0][0]
    delta_eff = -g["log"][-2]["hidden_err"]       # greedy never samples k*: its error is the prior error
    ts = [r for r in runs if r["seed"] == seed and r["policy"] == "thompson" and r["eta"] == 0][0]
    first = next((l["round"] for l in ts["log"] if l["hidden_contacts"] > 0), None)
    n_ts = ts["log"][-2]["hidden_contacts"]
    rows.append(dict(seed=seed, lam0=lam0, first=first, delta=delta_eff, sigma2=sigma2_eff, n=n_ts,
                     err_obs=ts["log"][-2]["hidden_err"],
                     err_gauss=gauss_post_mean_err(delta_eff, n_ts, sigma2_eff, 0.25),
                     err_t=t_post_mean(0.0, max(n_ts, 1), sigma2_eff, -delta_eff, 0.5, 3) if n_ts > 0 else -delta_eff))
    if runs_r:
        tr = [r for r in runs_r if r["seed"] == seed and r["policy"] == "thompson-robust" and r["eta"] == 0]
        if tr:
            rows[-1]["n_r"] = tr[0]["log"][-2]["hidden_contacts"]
            rows[-1]["err_obs_r"] = tr[0]["log"][-2]["hidden_err"]
            nr = max(rows[-1]["n_r"], 1)
            rows[-1]["err_t_r"] = t_post_mean(0.0, nr, sigma2_eff, -delta_eff, 0.5, 3)
            rows[-1]["err_g_r"] = gauss_post_mean_err(delta_eff, nr, sigma2_eff, 0.25)

p("**P2 in our setting (account-level marginal TS, prior at round 0).** λ₀ = expected hidden-segment accounts "
  "in the first batch from the order-statistic formula with the real CPR-LP prior means and variances; "
  "observed = first round with a hidden contact (1 = the first batch).")
p("")
p("| seed | Δ_eff (prior error) | λ₀ | P(hit in round 0) = 1−e^{−λ₀} | observed first-contact round |")
p("|---|---|---|---|---|")
for r in rows:
    p(f"| {r['seed']} | {r['delta']:.2f} | {r['lam0']:.3f} | {1 - np.exp(-r['lam0']):.2f} | {r['first']} |")
lam = np.array([r["lam0"] for r in rows])
p("")
p(f"Predicted number of seeds hit in round 0: {np.sum(1 - np.exp(-lam)):.1f} of 10; observed "
  f"{sum(r['first'] == 1 for r in rows)}. Predicted median first round if the per-round rate stayed λ₀ "
  f"(geometric): {np.median(np.ceil(np.log(0.5) / np.log(np.exp(-lam)))):.0f}; observed median "
  f"{np.median([r['first'] for r in rows if r['first']]):.0f}.")
z_seg = np.mean([r["delta"] for r in rows]) / (np.sqrt(2) * 0.5)
p(f"Segment-level TS (one draw per segment from the ω = 0.5 segment prior) would instead need ≈ 1/Φ(−z) = "
  f"{1 / stats.norm.sf(z_seg):.0f} rounds (z = {z_seg:.2f}).")
p("")
p("**P4 in our setting (Thompson, Gaussian prior, round 39).** Gaussian formula with each seed's Δ_eff, its "
  "number of hidden contacts n, and σ² = 1/(p(1−p)) + graph² + nugget² — vs observed CPR-LP error.")
p("")
p("| seed | n hidden contacts | observed error | Gaussian formula | Student-t (exact 1-D) |")
p("|---|---|---|---|---|")
for r in rows:
    p(f"| {r['seed']} | {r['n']} | {r['err_obs']:+.2f} | {r['err_gauss']:+.2f} | {r['err_t']:+.2f} |")
eo = np.array([r["err_obs"] for r in rows]); eg = np.array([r["err_gauss"] for r in rows])
p("")
p(f"Mean observed {eo.mean():+.2f}, Gaussian formula {eg.mean():+.2f}; correlation across seeds "
  f"{np.corrcoef(eo, eg)[0, 1]:.2f}.")
if runs_r and "err_obs_r" in rows[0]:
    p("")
    p("**P4, robust runs (thompson-robust, ν = 3, round 39).**")
    p("")
    p("| seed | n hidden contacts | observed error (EM Student-t) | Student-t exact 1-D | Gaussian formula at same n |")
    p("|---|---|---|---|---|")
    for r in rows:
        p(f"| {r['seed']} | {r['n_r']} | {r['err_obs_r']:+.2f} | {r['err_t_r']:+.2f} | {r['err_g_r']:+.2f} |")
    er = np.array([r["err_obs_r"] for r in rows]); et = np.array([r["err_t_r"] for r in rows])
    p("")
    p(f"Mean observed {er.mean():+.2f}, exact Student-t {et.mean():+.2f}, Gaussian at the same n "
      f"{np.mean([r['err_g_r'] for r in rows]):+.2f}.")
p("")
# P3: exploration-only sampling rate and the time to washout in our setting
Kseg, Bb = 32, 25
sig2 = np.mean([r["sigma2"] for r in rows]); D = np.mean([r["delta"] for r in rows])
p("**P3/P5 in our setting.** Stratified exploration samples k* at rate ηB/K per round. Contacts needed for "
  f"|bias| ≤ ε = 0.5 with Δ = {D:.2f}, σ² = {sig2:.1f}, ω = 0.5:")
nG = (sig2 / 0.25) * (D / 0.5 - 1)
nT = next(n for n in range(1, 5000) if abs(t_post_mean(0.0, n, sig2, -D, 0.5, 3)) <= 0.5)
for eta in (0.05, 0.1, 0.2):
    rate = eta * Bb / Kseg
    p(f"- η = {eta}: rate {rate:.3f} contacts/round → Gaussian needs n = {nG:.0f} ({nG / rate:.0f} rounds from "
      f"exploration alone); Student-t needs n = {nT} ({nT / rate:.0f} rounds)")

# Batched segment-level Thompson (P2a′) and post-contact washout: from the diagnostics, if they have been run
bj = ROOT / "results/diag_theory_batched.json"
if bj.exists():
    rows_b = json.load(open(bj))
    P = np.array([r["P_exact"] for r in rows_b])
    p("")
    p("**P2a′ (batched segment-level Thompson, depletion) vs results/explore_h4_seg.md** "
      "(scripts/diag_theory_batched.py; hidden segment withheld = 'not yet contacted' path).")
    p("")
    p("| seed | P(reach k* ≤ 40), exact | product formula | single selection vs best available | observed first contact | s_k* (round 0) | rounds to deplete accounts above k* |")
    p("|---|---|---|---|---|---|---|")
    for r in rows_b:
        p(f"| {r['seed']} | {r['P_exact']:.2f} | {r['P_prod']:.2f} | {r['P_single']:.2f} | {r['first_obs']} | "
          f"{r['s_h0']:.2f} | {r['acc_above0'] / 25:.0f} |")
    p("")
    p(f"Expected seeds reaching k* within 40 rounds: exact {P.sum():.2f} (sd {np.sqrt((P * (1 - P)).sum()):.2f}), "
      f"product {sum(r['P_prod'] for r in rows_b):.2f}, single selection {sum(r['P_single'] for r in rows_b):.2f}; "
      f"observed 5. Mean per-round hazard by 10-round block: "
      f"{np.round(np.mean([r['p_by_decile'] for r in rows_b], axis=0), 4).tolist()}.")
pc = ROOT / "results/diag_theory_postcontact.md"
if pc.exists():
    p("")
    p("**P4 applied to segment-level Thompson (why the error stays near −1.9)** (scripts/diag_theory_postcontact.py):")
    p("")
    out.extend(pc.read_text().rstrip().split("\n"))
(ROOT / "results/theory_check_h4.md").write_text("\n".join(out) + "\n")
