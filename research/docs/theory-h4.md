# Theory for H4: when can a wrong founder belief be corrected?

Status labels: **[proven]** = complete proof under the stated assumptions; **[sketch]** = argument with a
step that is standard but not written out, or an approximation; **[conjecture]** = not proven.
Numerical checks: `scripts/theory_check_h4.py` → `results/theory_check_h4.md`. Empirical runs:
`results/explore_h4.md`, `results/explore_h4_robust.md`.

## 0. Simplified model

K segments; segment k has true log-odds θ_k. Each contact in segment k yields an observation that, after a
Laplace (Gaussian) approximation on the logit scale, is

  y ~ N(θ_k, σ_k²), σ_k² = 1/(p_k(1−p_k)) + s_e²,

where p_k = σ(θ_k) and s_e² is account-level variance around the segment logit (graph + nugget in CPR-LP).
Assumptions:

- **A1** Segments are independent a priori and in the likelihood (no trust pooling, no graph coupling).
- **A2** Prior θ_k ~ N(m_k, ω²) (Gaussian) or θ_k ~ t_ν(m_k, ω) (Student-t); σ_k² known.
- **A3** The founder is exactly right except on k* (the truly best segment): m_k = θ_k for k ≠ k*,
  m_{k*} = θ_{k*} − Δ with Δ > 0.
- **A4** Each round contacts B accounts; every segment has enough accounts for the horizon unless stated.
- **A5** Frequentist evaluation: θ is fixed; the prior is a (possibly wrong) modelling choice. Statements
  about "error" are about the posterior mean minus the true θ.

In CPR-LP the founder's slope b is learned (≈ 0.84 in our runs), so the *effective* prior error on k* is
Δ_eff = b·Δ ≈ 2.3–2.8 rather than Δ = 3. All checks against our runs use Δ_eff measured from the greedy
runs (greedy never samples k*, so its error *is* the prior error; P1).

## P1 — Greedy never corrects an avoided segment [proven]

Let μ_k(n) be segment k's posterior mean after n contacts (μ_k(0) = m_k). Greedy contacts, each round, the B
uncontacted accounts with the highest posterior means.

**Lemma 1.** If k* is not contacted up to round t, its posterior at t equals its prior. *Proof.* Under A1 the
posterior of θ_{k*} depends only on k*'s own observations; there are none. ∎

**Proposition 1.** Let G = {k ≠ k*: θ_k ≥ m_{k*} + γ} for some γ > 0, with Σ_{k∈G} S_k ≥ TB accounts in total.
Under the Gaussian prior (A2),

  P(greedy samples k* before round T) ≤ Σ_{k∈G} TB · exp(−2γ²/ω²).

*Proof.* Greedy can only pick a k* account in a round in which fewer than B uncontacted accounts have posterior
mean above m_{k*} (Lemma 1 keeps k*'s mean at m_{k*}). Since G holds at least TB accounts, this requires some
k ∈ G with μ_k(n) < m_{k*} ≤ θ_k − γ for some n ≤ TB. With m_k = θ_k (A3),
μ_k(n) − θ_k = c_n · (ȳ_n − θ_k), c_n = nω²/(σ² + nω²), so μ_k(n) − θ_k ~ N(0, v_n) with
v_n = c_n² σ²/n = nσ²ω⁴/(σ² + nω²)². Maximizing over n (at n = σ²/ω²) gives v_n ≤ ω²/4 for every n, hence
P(μ_k(n) < θ_k − γ) ≤ exp(−γ²/(2·ω²/4)) = exp(−2γ²/ω²). A union bound over n ≤ TB and k ∈ G gives the claim. ∎

**Corollary (linear regret).** On the complement event, every contact goes to a segment with θ_k ≤
max_{k≠k*} θ_k, so expected per-contact regret is at least θ_{k*} − max_{k≠k*} θ_k > 0 for all T rounds: regret
is linear in T, and k*'s posterior error stays exactly Δ forever (non-identifiability).

*Check.* Simplified model (K = 8, Δ = 3, ω = 0.5): k* never sampled in 200/200 runs; per-contact regret
0.387 ≥ 0.200. Our H4: greedy contacts k* 0 times in 10/10 runs; error −2.50 at round 39, equal to the prior
error (Lemma 1).

## P2 — Thompson sampling: escape time depends on *which* variance is sampled

**(a) Segment-level TS** (one draw per segment from its posterior, contact the top segment) **[proven for the
frozen-prior phase; sketch overall]**. While k* is unsampled its draw is N(m_{k*}, ω²). Against a single
runner-up j with posterior N(μ_j, v_j), P(k* chosen) = Φ(−z), z = (μ_j − m_{k*})/√(ω² + v_j). At round 0
(v_j = ω²) and gap g = μ_j − m_{k*}: z = g/(√2 ω). With more competitors the probability is smaller (k* must
beat all), so Φ(−z) is an upper bound. The first-sampling time is then stochastically larger than a geometric
variable with success probability Φ(−z): mean ≥ 1/Φ(−z) ≈ z√(2π) e^{z²/2} (Mills ratio). The "overall"
part is a sketch because v_j shrinks as competitors are sampled, which raises z and lengthens the wait further.
This matches research-map.md's "linear regret until ≈ e^{z²/2}".

*Check.* Two-segment simulation: P = 1.69e-2 vs Φ(−z) = 1.69e-2 (z = 2.12); 3.83e-2 vs 3.85e-2 (z = 1.77);
2.8e-4 vs 2.0e-4 (z = 3.54, Monte Carlo noise at 200k draws). **In our setting (Δ_eff ≈ 2.5, ω = 0.5,
z ≈ 3.5) segment-level TS would need ≈ 4,900 rounds.**

**(b) Account-level marginal TS (what `fprior/explore.py` does)** **[sketch: mean-field order statistics]**.
Each account gets an independent draw μ_v + s_v ε_v, where s_v² is the account's *predictive* variance: the
segment uncertainty ω² **plus** graph and nugget variance (irreducible account-level variability), plus trust
uncertainty. The batch is the top B draws. With the threshold τ defined by Σ_v Φ̄((τ − μ_v)/s_v) = B, the
expected number of k* accounts per batch is

  λ = Σ_{v∈k*} Φ̄((τ − μ_v)/s_v),

and P(at least one) ≈ 1 − e^{−λ} (Poisson). Because s_v ≈ 1.3 ≫ ω = 0.5 and k* has ~100 accounts, λ is
orders of magnitude larger than Φ(−z) in (a).

*Check (simplified).* Formula vs simulation of E[#k* accounts/round]: 0.019 vs 0.018; 0.001 vs 0.001.
*Check (our runs, real CPR-LP prior at round 0).* λ₀ = 0.05–0.23 per seed. Predicted seeds hit in the first
batch: 1.3 of 10; observed 2. Geometric median first round at rate λ₀: 5; observed median 3 (first-contact
rounds 1, 1, 2, 2, 2, 4, 8, 8, 10, 29).

**Interpretation.** Our Thompson sampler escapes in a few rounds not because posterior uncertainty about
θ_{k*} is large, but because it samples irreducible account-level noise — "accidental exploration". This is
a property of marginal per-account draws, not of Thompson sampling per se; a segment-level or correctly
correlated sampler would be stuck for thousands of rounds (a). The research-map statement is right for (a)
and does not describe our implementation.

## P3 — Stratified η-exploration gives consistency [proven]

Each exploration pick (a Binomial(B, η) number per round) chooses a segment uniformly among the K available,
then an account uniformly within it.

**Proposition 3.** Under A1–A4 with η > 0: (i) N_{k*}(t), the number of k* contacts by round t, satisfies
E N_{k*}(t) ≥ ηBt/K; (ii) N_{k*}(t) → ∞ almost surely; (iii) the posterior mean of θ_{k*} converges to θ_{k*}
almost surely, for the Gaussian prior and for the Student-t prior.

*Proof.* (i) Each round, each of the B slots is an exploration pick with probability η and then lands in k*
with probability 1/K; exploitation picks only add. (ii) Rounds are independent given the exploration coins,
and each round samples k* with probability ≥ 1 − (1 − η/K)^B > 0; the second Borel–Cantelli lemma gives
infinitely many such rounds a.s. (iii) Gaussian: μ(n) = (ω⁻² m + nσ⁻² ȳ_n)/(ω⁻² + nσ⁻²) → θ by the strong law,
since the prior weight vanishes. Student-t: the posterior mean is ȳ_n + (σ²/n)(log f_n)'(ȳ_n) (Tweedie, P4)
with |(log f_n)'| bounded on compacts uniformly in large n, so the correction term is O(1/n) → 0 while
ȳ_n → θ. ∎ (For (iii-t) the uniform bound is the one step written as a sketch.)

**Rate.** k* receives ≈ ηB/K contacts per round from exploration alone. In our setting (K = 32, B = 25):
0.039 / 0.078 / 0.156 contacts per round for η = 0.05 / 0.1 / 0.2.

## P4 — How fast a wrong belief washes out: Gaussian vs Student-t prior

**Gaussian [proven].** With n observations, conjugacy gives μ(n) = θ − Δ·σ²/(σ² + nω²) + c_n(ȳ_n − θ), so

  E[μ(n)] − θ = −Δ · σ²/(σ² + nω²),

and |bias| ≤ ε requires n ≥ n_ε^G = (σ²/ω²)(Δ/ε − 1). **The number of contacts needed grows linearly in Δ:
the more confidently wrong the founder, the longer it takes.**

**Student-t [Tweedie step proven; tail step sketch].** Let f_n be the marginal density of ȳ_n, i.e. the prior
convolved with N(0, σ²/n). Tweedie's formula (Robbins 1956; Efron 2011) is exact for a Gaussian likelihood
and any prior:

  E[θ | ȳ] = ȳ + (σ²/n) · (log f_n)'(ȳ).

For the Gaussian prior, (log f_n)'(ȳ) = −(ȳ − m)/(ω² + σ²/n), which recovers the linear-in-Δ bias. For the
t_ν(m, ω) prior, f_n has the prior's polynomial tails, so (log f_n)'(ȳ) ≈ −(ν + 1)(ȳ − m)/(νω² + (ȳ − m)²)
when σ²/n ≪ ω². This gives

  E[μ(n)] − θ ≈ −(σ²/n) · (ν + 1)Δ/(νω² + Δ²),

which **tends to 0 as Δ → ∞ for fixed n**: a sufficiently large prior–data conflict is resolved in favour of
the data (bounded influence of the prior). This is the classical conflict/outlier-rejection behaviour of
heavy-tailed models: Dawid (1973), O'Hagan (1979), Andrade & O'Hagan (2006). The approximation step (tail of
the convolution and of its log-derivative) is standard regular-variation reasoning, not written out here.
Contacts needed: n_ε^t ≈ (σ²/ε)(ν + 1)Δ/(νω² + Δ²), which *decreases* in Δ once Δ > √ν·ω.

*Check (simplified, σ² = 5, ω = 0.5, ν = 3).* Gaussian formula vs exact: −2.000 vs −1.998 (Δ = 3, n = 10),
−0.333 vs −0.335 (Δ = 3, n = 160), and all nine cells within 0.02. Student-t exact vs asymptotic: −0.136 vs
−0.154 (Δ = 3, n = 40); −0.124 vs −0.082 (Δ = 6, n = 40). The asymptotic form is poor when its conditions
fail (Δ = 1, n = 10: −0.543 exact vs −1.143), so quote the exact 1-D value there.
Contrast at n = 40: Gaussian bias −1.00 (Δ = 3) and −2.00 (Δ = 6); Student-t −0.14 and −0.12.

*Check (our runs, round 39).*
- Thompson, Gaussian CPR-LP: Gaussian formula with each seed's Δ_eff, n and σ² = 1/(p(1−p)) + graph² +
  nugget² gives a mean of −1.33 vs observed −1.24; correlation across seeds 0.88.
- Thompson, robust (EM Student-t, ν = 3): exact 1-D Student-t gives a mean of −0.59 vs observed −0.62;
  the Gaussian formula at the same n would give −1.09. Per-seed errors scatter more (e.g. observed +0.33 vs
  predicted −0.08 at n = 108), because the EM/Laplace CPR-LP also has trust pooling and graph terms.

## P5 — The combined statement [proven pieces + sketch]

Combining P1, P3 and P4 under A1–A5, for a founder who is Δ-wrong about the truly best segment:

1. **No exploration (greedy):** with probability ≥ 1 − |G|·TB·e^{−2γ²/ω²}, k* is never sampled, its error
   stays Δ, and regret is linear (P1). No prior choice can fix this (Lemma 1). [proven]
2. **Exploration, Gaussian prior:** k* is sampled infinitely often and the estimate is consistent (P3), but
   reaching |bias| ≤ ε takes about (K/(ηB)) · (σ²/ω²)(Δ/ε − 1) rounds — linear in Δ. [proven, using the
   exploration-only rate]
3. **Exploration, Student-t prior:** consistent (P3), and the time is about
   (K/(ηB)) · (σ²/ε)(ν + 1)Δ/(νω² + Δ²) — *decreasing* in Δ for large conflicts. [sketch: P4 tail step]

In our setting (Δ_eff = 2.5, σ² = 6.2, ω = 0.5, ε = 0.5): 99 contacts (Gaussian) vs 22 contacts (Student-t,
exact 1-D). From exploration alone at η = 0.1 that is ≈ 1,266 vs ≈ 282 rounds. **This explains the
pre-registered E2 failure:** 40 rounds of stratified exploration add only ≈ 3 contacts to k*, far too few; the
contacts that did the work came from Thompson's accidental exploration (P2b), and the Student-t prior made
those contacts count roughly twice as much (P4).

## What the theorems do and don't cover

- **Not covered:** CPR-LP's trust pooling (the learned slope b couples segments: wrong-founder evidence
  elsewhere can lower trust in k* too), the graph term, and the logistic likelihood (we use the Laplace /
  Gaussian approximation; the checks show it is adequate at the level of means). The washout bounds condition
  on the number of contacts n; under adaptive sampling n is random and data-dependent, and the bias formulas
  then hold only conditionally (a martingale argument would be needed for unconditional statements).
- **Regret:** only the linear lower bound for greedy is proven. No upper bounds on regret for TS + η or for
  the Student-t prior are derived here; the research-map regret bounds (confidence-weighted KL term D₀)
  remain unverified.
- **Bayesian vs frequentist:** everything above is frequentist in θ (fixed truth, possibly wrong prior), which
  is the right lens for "the founder is wrong". None of it is a Bayesian-regret statement.
- **P2(b)** is an approximation (independent per-account draws, mean-field threshold); it explains our
  sampler's behaviour, but the "accidental exploration" it describes depends on including irreducible
  account variance in Thompson draws, which is an implementation choice rather than a principled one.

## References

- A. P. Dawid (1973). Posterior expectations for large observations. *Biometrika* 60, 664–667 (pages per the
  author's publication list, https://www.statslab.cam.ac.uk/~apd25/publist2.pdf; some lists give 664–666).
  *Bibliographic details verified; full text not read.*
- A. O'Hagan (1979). On outlier rejection phenomena in Bayes inference. *JRSS B* 41, 358–367.
  *Bibliographic details and summary verified from secondary sources; full text not read.*
- J. A. A. Andrade and A. O'Hagan (2006). Bayesian robustness modeling using regularly varying
  distributions. *Bayesian Analysis* 1(1), https://projecteuclid.org/journals/bayesian-analysis/volume-1/issue-1/Bayesian-robustness-modeling-using-regularly-varying-distributions/10.1214/06-BA106.pdf
  *Existence verified via search; not read.*
- B. Efron (2011). Tweedie's formula and selection bias. *JASA* 106(496), 1602–1614, doi:10.1198/jasa.2011.tm11181
  (formula first reported by Robbins, 1956). *Bibliographic details verified; full text not read.*

## Erratum / open item (2026-11-24, after results/explore_h4_seg.md)

P2a's escape-time bound is for selecting one segment per round against its best competitor. The empirical test with
segment-level Thompson sampling and batches of 25 accounts spread across segments reached the hidden segment in 5/10
seeds within 40 rounds, far sooner than the ≈ 4,900-round estimate. A batched version is needed: with batch B spread
over the top-r segments of the draw, the relevant gap is to the r-th order statistic, not the maximum. Until derived,
quote P2a only for the single-selection setting.
