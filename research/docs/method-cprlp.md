# Method draft: CPR-LP (as implemented, 2026-11-24)

Draft methods section. Notation follows `fprior/cprlp.py`, `fprior/cprlp_hier.py`, `fprior/explore.py`.
Everything below is what the code does; departures from `research-map.md` are flagged.

## Setup

A seller faces accounts v = 1…N. Each account belongs to a segment k(v) (here industry × size; K segments).
The founder states a belief m_k ∈ (0, 1) per segment: "of accounts in segment k that we contact, a fraction
m_k will convert", and we write ℓ_v = logit m_{k(v)} and ℓ̄ for its mean over accounts. Outreach contacts a set
C of accounts and observes y_v ∈ {0, 1} for v ∈ C. Uncontacted accounts are unlabeled.

## Model

Latent log-odds of conversion:

  x_v = ℓ̄ + a + b (ℓ_v − ℓ̄) + Σ_j z_vj β_j + ξ_{k(v)} + r_v + ε_v,   y_v | x_v ~ Bernoulli(σ(x_v)), v ∈ C

- **Calibrated founder prior (trust).** (a, b) ~ N(μ, S): a is the founder's level bias, b the slope ("how much
  their ranking of segments is worth"). Pooled across all segments, so 5–50 labels inform it.
  Default μ = (0, 0.5), S = diag(0.5², 0.5²); the population implied by our simulated founders is close
  (a: 0.00 ± 0.48, b: 0.34 ± 0.55; `results/matched_prior.md`).
- **Segment deviation** ξ_k ~ N(0, ω²): how far segment k departs from the calibrated belief.
  *Robust variant:* ξ_k | ω_k² ~ N(0, ω_k²), ω_k² ~ Inv-Gamma(ν/2, νω₀²/2), so ξ_k is Student-t(ν, ω₀) (ν = 3),
  fitted by EM; see `explore_h4_robust.md` and `theory-h4.md` (bounded prior influence).
- **Shared main effects (hierarchical variant).** z_v = one-hot industry and size, β_j ~ N(0, s²): evidence about
  one cell informs cells sharing its industry or size (`diag_learner_gap.md`).
- **Relational regressors (relational variant).** Additional standardized public features (e.g. similarity of the
  account's adopted products to the seller's product) enter the same way with N(0, 0.5²) coefficients.
- **Graph residual** r ~ N(0, σ_g² C), C = (I + γL)⁻² normalized to unit mean variance, L the normalized Laplacian
  of an account graph (co-adoption in the relational simulator); σ_g = 0 where no graph is used.
- **Nugget** ε_v ~ N(0, σ_n²).

All coefficient blocks (a, b, β) enter linearly with Gaussian priors, so they are **integrated out exactly** by
adding H S Hᵀ to the kernel (GPML §2.7), giving a GP with mean m_v = ℓ̄ + μ_a + μ_b(ℓ_v − ℓ̄) and kernel

  K = ω² [k(v) = k(v′)] + s² ([same industry] + [same size]) + σ_g² C + H S Hᵀ + σ_n² I.

## Inference

- Bernoulli likelihood on contacted accounts; **Laplace approximation** (Newton mode finding, GPML Alg. 3.1, with
  step-halving on the robust path). Predictive mean for every account: m + K_{·C} ∇log p(y | f̂).
  *(Departure from the research map: Laplace, not Pólya-Gamma augmentation.)*
- **Evidence averaging** over hyperparameter grids instead of point selection: ω over (0.05 … 1.5) with uniform
  weights times Laplace evidence; the main-effect scale s over (0, 0.25, 0.5, 1.0), where s = 0 is the flat model,
  so the evidence decides whether to pool. Argmax selection of ω from 1–2 labels was the main failure in M0.
- **Learned trust readout:** E[(a, b) | y] = μ + S Hᵀ_C ∇log p(y | f̂).
- **Ranking:** posterior mean logit (or E[p] via the probit approximation; no difference in our tests).

## Selection

Outreach depends on the founder's beliefs (and an exploration share η), which the model conditions on, so
selection is ignorable for this posterior. That is why no propensity weighting appears. It breaks if outreach
depends on anything the model doesn't see (warm intros, unlogged early contacts); the doubly robust PU loss
in the research map is for that case and is **not implemented**.

## Sequential version (exploration)

Per round: fit the posterior, draw from it (account-level marginal Thompson sampling) or take the mean (greedy),
mix in stratified η-exploration (uniform segment, then uniform account), contact a batch. Theory in `theory-h4.md`:
greedy cannot correct a wrong belief about an avoided segment; exploration restores consistency; heavy-tailed ξ
bounds the prior's influence. Caveat: account-level marginal draws explore "accidentally" compared with
segment-level Thompson sampling.

## What each piece fixes (evidence)

| Piece | Failure it addresses | Evidence |
|---|---|---|
| Trust slope b (pooled) | systematic founder error (inverted / useless founders) | `m0_sanity.md`, `placebo_bank.md`, `bank_bad_founders.md` |
| ω evidence averaging | noisy hyperparameters from 1–2 labels | M0 fix (`m0_sanity.md`) |
| Student-t ξ | an isolated, confidently wrong segment | `explore_h4_robust.md`, `theory-h4.md` |
| Main effects with s ∋ 0 | can't pool across related segments; over-pooling with few labels | `diag_learner_gap.md`, `m0_hier.md`, `real_bank_hier_v2.md` |
| Relational regressors | ignores public product-similarity signal | `rel_cprlp.md` |
| Exploration η | wrong beliefs about avoided segments never tested | `explore_h4*.md`, `theory-h4.md` |

## Not implemented (from the research map)

Pólya-Gamma PU likelihood; doubly robust PU loss with logged propensities; contractive-equilibrium GNN body;
unbalanced-OT alarm; multiple founders / LLM sources pooled; delay-corrected funnel stages.
