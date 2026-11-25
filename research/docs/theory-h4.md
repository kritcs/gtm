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

**(a) Segment-level TS, single selection** (one draw per segment from its posterior, contact the *one* top
segment per round, segments never depleted) **[proven for the frozen-prior phase; sketch overall]**. *Scope:
this is not our experimental setting, where each round takes B accounts and contacted accounts are used up;
see (a′).* While k* is unsampled its draw is N(m_{k*}, ω²). Against a single
runner-up j with posterior N(μ_j, v_j), P(k* chosen) = Φ(−z), z = (μ_j − m_{k*})/√(ω² + v_j). At round 0
(v_j = ω²) and gap g = μ_j − m_{k*}: z = g/(√2 ω). With more competitors the probability is smaller (k* must
beat all), so Φ(−z) is an upper bound. The first-sampling time is then stochastically larger than a geometric
variable with success probability Φ(−z): mean ≥ 1/Φ(−z) ≈ z√(2π) e^{z²/2} (Mills ratio). The "overall"
part is a sketch because v_j shrinks as competitors are sampled, which raises z and lengthens the wait further.
This matches research-map.md's "linear regret until ≈ e^{z²/2}".

*Check.* Two-segment simulation: P = 1.69e-2 vs Φ(−z) = 1.69e-2 (z = 2.12); 3.83e-2 vs 3.85e-2 (z = 1.77);
2.8e-4 vs 2.0e-4 (z = 3.54, Monte Carlo noise at 200k draws). With our round-0 numbers (Δ_eff ≈ 2.5,
ω = 0.5, z ≈ 3.5) single selection would need ≈ 4,900 rounds. **This does not transfer to our experiment:**
segment-level TS with batches reached k* in 5/10 seeds within 40 rounds (`results/explore_h4_seg.md`).
(a′) explains why.

**(a′) Segment-level TS, batched, with depletion** (what policy `thompsonseg` does) **[exact characterisation;
the hazard bound is a sketch]**. Each round, segment k gets one draw u_k = μ_k + s_k z_k (z_k iid N(0,1)), every
remaining account v in k is scored μ_v + s_k z_k, and the top B accounts are contacted; contacted accounts are
removed. Write R_j(t) for the accounts remaining in segment j and, for simplicity, let accounts in a segment share
the segment's mean (in CPR-LP they differ slightly through the graph term; the numerical check uses the real
per-account means).

*Exact selection condition.* k* is contacted in round t iff the remaining accounts of the segments drawn above it
number fewer than B:

  k* contacted ⇔ Σ_{j≠k*} R_j(t) · 1[u_j > u_{k*}] < B.

Conditioning on z_{k*} = z, the indicators are independent Bernoulli(q_j(z)), q_j(z) = Φ((μ_j − μ_{k*} − s_{k*} z)/s_j), so

  p_t = ∫ φ(z) · P( Σ_j R_j(t) X_j(z) < B ) dz,  X_j(z) ~ Bernoulli(q_j(z)) independent.

*The r(B) view.* If every competitor still has R_j ≥ B, a single segment drawn above k* fills the batch, so k*
must be the top draw among the non-exhausted segments A_t = {j: R_j(t) ≥ B}:

  p_t ≥ p_t^prod = ∫ φ(z) Π_{j∈A_t} Φ((μ_{k*} + s_{k*} z − μ_j)/s_j) dz,

with equality when no segment is partially depleted (0 < R_j < B). So with segments much larger than B
(here ≈ 94 accounts vs B = 25), **batching by itself does not lower the bar**: r(B) = 1. What lowers it:

1. **Depletion.** Contacting a segment uses it up. The founder's favourite segments empty first, so the best
   competitor still available drifts down toward μ_{k*} and A_t shrinks. A deterministic ceiling: k* is reached
   no later than T_dep = (accounts whose mean exceeds μ_{k*})/B rounds, the point where greedy itself would get
   there. Here T_dep = 65–98 rounds, which is why greedy needs more than 40 rounds.
2. **Partially depleted segments** (0 < R_j < B) only block part of the batch; the exact p_t includes them,
   p_t^prod does not.
3. **The draw's scale.** The segment-level posterior sd of k* is s_{k*} ≈ 0.71–0.92, not ω = 0.5. It also
   includes trust-slope uncertainty, s_b²(ℓ_{k*} − ℓ̄)², which is large because the founder's belief about k* is
   extreme.

First contact then has survival Π_{τ≤t}(1 − p_τ), with p_t increasing over time through depletion. The
e^{z²/2} behaviour of (a) holds round by round with a z_t that shrinks as competitors deplete.

*Check (`scripts/diag_theory_batched.py`).* Each seed is replayed with k* withheld (the "not yet contacted"
path), and p_t is computed by Monte Carlo with the real posterior means, segment sds and remaining accounts.

| | Expected seeds reaching k* within 40 rounds | Observed |
|---|---|---|
| Exact p_t | 3.87 (sd 1.43) | 5 |
| Product formula p_t^prod (ignores partial depletion) | 2.76 | 5 |
| Single selection against the best available competitor, sd ω | 6.30 (wrong: it ignores every other competitor) | 5 |

- **Hazard over time:** p_t rises from about 0.001–0.006 per round in rounds 1–10 to 0.006–0.063 in rounds 31–40.
- **Timing:** 77% of the predicted first contacts fall in rounds 21–40. Observed first contacts: rounds 23, 28,
  28, 31, 34.

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

**Interpretation.** There are two escape routes, and they work on different time scales:

1. **Account-level marginal TS (b)** reaches k* in a few rounds (median 3). It does so because it samples
   irreducible account-level noise, not because it is unsure about θ_{k*}: "accidental exploration".
2. **Segment-level TS (a′)** draws only segment uncertainty. It reaches k* late (rounds 23–34) and in about
   half the seeds, mostly because the founder's favourite segments are used up (depletion), helped by
   trust-slope uncertainty widening k*'s draw.

Single-selection (a), the research-map "≈ e^{z²/2}" statement, describes neither our account-level sampler nor
our batched segment-level sampler. It only holds when the top competitor never runs out of accounts.

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

*Why segment-level Thompson stays near −1.9 even with the Student-t prior (`scripts/diag_theory_postcontact.py`,
`results/diag_theory_postcontact.md`).* **Mainly contact count.** The −1.96 (Gaussian) / −1.90 (Student-t) averages
mix three groups of seeds:

| Group (Gaussian / Student-t run) | Seeds | Error at round 39 | What the formula predicts |
|---|---|---|---|
| Never contacted | 5 / 5 | The prior error: −2.2 to −2.9 | The prior error (Lemma 1) |
| Contacted late with 1–8 contacts (first contact at rounds 28–34) | 3 / 3 | −1.4 to −2.7 | Gaussian −1.97 to −2.67; Student-t −1.27 to −2.55 |
| About 90–120 contacts | 2 / 2 | Gaussian −0.49 / −0.34; Student-t −0.33 / −0.09 | Student-t −0.07 / −0.10 |

At each seed's actual n, the formulas reproduce the means: Gaussian −2.02 vs observed −1.96; Student-t −1.84 vs
observed −1.90. The account-level robust run reaches −0.62 because its contacts come early and in volume (median
50, 6/10 seeds with ≥ 20), not because the prior differs.

- **ω** sets the Gaussian half-life n = σ²/ω² ≈ 25 contacts. It explains why Student-t matters once n ≳ 20, not
  the −1.9 average.
- **Trust slope b** absorbs part of the correction in the Gaussian runs but not the Student-t runs. In the two
  heavily sampled seeds the Gaussian model's b falls from ~0.8 to 0.50, so evidence about k* lowers trust in
  the founder everywhere. The Student-t model keeps b at 0.83 / 0.70 and moves the correction into k*'s own
  deviation, so the evidence is attributed where it belongs. This affects the other segments, not the mean
  error on k*.

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
contacts that did the work came from Thompson's accidental exploration (P2b) or, for segment-level Thompson,
from depletion late in the horizon (P2a′); the Student-t prior made those contacts count roughly twice as much once
there were ≳ 20 of them (P4).

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
