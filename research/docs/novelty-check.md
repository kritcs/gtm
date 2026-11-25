# Novelty check: learned trust in a founder's segment priors

Date: 2026-11-24. Searches were done with a web-search tool and pages fetched with a summarizing fetch tool. "Source checked" says what was actually read. The EXNEX original (Wiley, PubMed and PMC) was blocked or paywalled, so I did NOT read its full text. EXNEX details below come from secondary sources and need confirming before submission.

## Verdicts

**A. Learning per-source, per-segment trust in expert beliefs, pooled, from 5-50 labels: partially anticipated.**
- The nearest relatives are EXNEX (Neuenschwander et al. 2016) and robust MAP priors (Schmidli et al. 2014). Both are mixtures of an informative component and a robust or non-exchangeable component, and the data decide which component each stratum or study follows.
- In EXNEX, each stratum's parameter is exchangeable with the others or non-exchangeable. Per the Daniells et al. 2025 description, prior mixture weights are usually pre-specified (often equal across strata) and then updated to per-stratum posterior probabilities.
- The borrowing target in EXNEX is the other strata's parameters, via a hierarchical mean and tau. It is not an external expert's stated belief m_k. The mixture is a discrete switch (follow or ignore), not a calibration slope and intercept on the expert's logit.
- The defensible difference is the object and parameterization. We learn a continuous, pooled calibration of a stated expert belief, p_k = a + b·logit(m_k). It is shared across segments and marginalized in closed form. A deviation scale omega per segment is selected by evidence.
- The setting also differs: positive-unlabeled labels, prior-steered sampling, and 5-50 labels.
- Do not claim that "adaptive trust in an informative prior" is new. Power priors, commensurate priors and robust MAP priors all do this (see Open questions: the power/commensurate-prior source is cited from memory and not opened).
- Expert-source reliability is also studied in weak supervision (Snorkel, which learns labeling-function accuracies without ground truth). It is a related "learn source accuracy" idea, but the sources are labeling functions rather than segment-level beliefs on a graph.

**B. Doing this inside a graph / relational link-prediction model: novel, as far as I could find.**
- Searches on knowledge-graph rule confidence, human priors for link prediction, and recommender cold start found rule-confidence methods (AnyBURL-style rule confidences) and human-feature KG work (KG-HAIT). They treat rules or features as inputs to be mined or weighted. I found no paper that calibrates a per-source expert belief with a pooled trust parameter inside a GP or relational posterior.
- This is an absence-of-evidence claim from a limited search. I did not open the KG papers beyond search summaries. Say "to our knowledge".

**C. Amortizing trust with a PFN trained on simulated experts of random reliability: partially anticipated (PFNs4BO), but the defensible difference is real.**
- I read PFNs4BO's full text (Sec. 5.3 and Appendix G). The user gives an interval I and a confidence rho, and the prior is p(D|rho,I) = rho·p(D|m in I) + (1-rho)·p(D).
- The PFN is trained with rho ~ U(0,1) drawn per task, and the hint is correct with probability rho. So the PFN is trained on hints of varying reliability, but rho is supplied as an input and is assumed calibrated.
- The PFN therefore does Bayes under a stated, calibrated confidence. It does not infer, from the task's own observations, how reliable this particular source is. In their PD1 experiment a single generic user prior was used, with no study of miscalibrated confidence.
- Our PC-RFM has no reliability input. Reliability is a latent variable of the simulated founder, with many segments and a relational structure, and the network must infer it in context from a few PU labels.
- Newer work found: Huang et al. 2026 (MLHC) uses amortized NPE to borrow external data under non-exchangeability. It learns from simulated dataset pairs, but I could only confirm the abstract. It has no expert-belief input and no segment structure.
- I found no 2024-2026 PFN paper that infers prior-source reliability in context. Searches are not exhaustive.

**D. Identifiability when the prior steers which units are observed: anticipated as a general phenomenon, novel as a specific result.**
- Selection by a policy is well studied. MNAR exposure in recommenders (Schnabel et al. 2016) handles it with known or estimable propensities. Thompson sampling under misspecified priors (Simchowitz et al. 2021) shows graceful degradation in reward, and other work shows worst-case linear regret. Performative prediction (Perdomo et al. 2020) covers predictions that change outcomes.
- None of these makes our specific point: a wrong belief about a never-sampled segment cannot be corrected by pooled calibration. Pooling can even extrapolate a wrong prior into such segments, and only exploration or positivity fixes it.
- It is a short formal observation (positivity/overlap violation). It is not a deep new theorem, so present it as a proposition and a simulation, not a headline contribution.

## Table

| paper | year | what it does | overlap | how we differ | source checked | URL |
|---|---|---|---|---|---|---|
| Neuenschwander, Wandel, Roychoudhury, Bailey, EXNEX | 2016 (online 2015) | Per-stratum mixture of exchangeable and non-exchangeable priors for basket trials | A (close) | Borrows across strata, not from an expert belief. Discrete mixture, not calibration slope b. Our labels are PU and prior-steered | abstract via search summary plus bhmbasket vignette (weights are user-supplied inputs). Original full text NOT read | https://zendy.io/title/10.1002/pst.1730 ; https://cran.case.edu/web/packages/bhmbasket/vignettes/reproduceExNex.html |
| Daniells, Mozgunov, Barnett, Bedding, Jaki (Biostatistics 26(1)) | 2025 | Data-driven borrowing from historic data into baskets (EXNEX with historic data, EXppNEX, MLMixture). Says EXNEX weights are usually pre-specified and updated to posteriors | A | Discrete or power-weight borrowing from historical data, no expert-belief calibration. Power parameter in EXppNEX is fixed | full text, first ~100k of ~123k characters | https://pmc.ncbi.nlm.nih.gov/articles/PMC12204204/ |
| Schmidli et al., robust MAP priors (Biometrics 70(4)) | 2014 | Informative MAP prior plus a weakly informative robust component. Weight reacts to prior-data conflict | A | Source is historical controls. Weight is a prior-specified mixture weight, not a pooled per-segment calibration. Not graph or PU | abstract via search summary | https://boris.unibe.ch/59957 |
| Müller, Feurer, Hollmann, Hutter, PFNs4BO (ICML) | 2023 | PFN surrogate for BO. User gives interval I and confidence rho, trained on p(D\|rho,I) | C (closest), D (partly) | rho is a calibrated input, not inferred. Single-task hint on the optimum, not per-segment beliefs. No relational structure | full text (Sec. 5.3, App. G) | https://arxiv.org/abs/2305.17535 |
| Huang, Lee, He, amortized NPE data borrowing (MLHC, PMLR 340) | 2026 | Pretrained NPE borrows external data under covariate shift and outcome drift | C | No expert belief input, no segments, no graph | abstract only | https://proceedings.mlr.press/v340/huang26a.html |
| Capstick, Krishnan, Barnaghi, AutoElicit (ICML 2025) | 2024-25 | LLM-elicited priors for linear models, refined in language | A, C (weak) | Elicits the prior, does not learn how much to trust it. No pooled trust parameter | abstract only | https://arxiv.org/abs/2411.17284 |
| Wang, Evans, Nott, robust Bayesian SBI | 2025 | Amortized posteriors over a class of priors. Prior-data conflict checks | A, C (weak) | Sensitivity analysis over priors, no learned trust | abstract only | https://arxiv.org/abs/2504.09475 |
| Ratner et al., Snorkel | 2017 | Learns accuracies of labeling-function sources without ground truth | A (analogy) | Sources are labeling functions, no calibration to segment-level probabilities, no graph | abstract only | https://arxiv.org/abs/1711.10160 |
| Simchowitz et al., Bayesian decision-making under misspecified priors | 2021 | Thompson sampling regret when the prior is misspecified | D | Gives reward-gap bounds. No identifiability statement about unsampled segments | abstract only | https://arxiv.org/abs/2107.01509 |
| Schnabel et al., Recommendations as Treatments | 2016 | Propensity-weighted learning and evaluation under MNAR exposure | D | Needs propensities. Our exposure depends on the belief being tested | search summary | https://arxiv.org/pdf/1602.05352 |
| Perdomo, Zrnic, Mendler-Dünner, Hardt, Performative Prediction | 2020 | Predictions that change outcomes. Retraining converges to a stable point | D (weak) | Different problem: distribution shift from acting on predictions | abstract only | https://arxiv.org/abs/2002.06673 |

## Suggested related-work wording

**A.** Borrowing strength with robustness to prior-data conflict is standard in clinical-trial statistics, for example EXNEX (Neuenschwander et al. 2016) and robust MAP priors (Schmidli et al. 2014). These models mix an informative component with a non-exchangeable or vague one, with weights that are typically pre-specified. We instead treat the founder's segment-level beliefs as a source to be calibrated. A pooled slope and intercept on the founder's logit, shared across segments and integrated out in closed form, sets the trust. A per-segment deviation scale captures segment-specific disagreement. We make this work with 5-50 positive-unlabeled labels.

**B.** To our knowledge, expert beliefs enter graph and relational link prediction as features or logical rules whose confidences are mined or fixed. We place the calibrated founder prior as the mean of a GP posterior over a market graph, so trust in the founder is estimated jointly with the graph smoothing.

**C.** PFNs4BO (Müller et al. 2023) trains a PFN on priors with random confidence rho and conditions on a user-supplied rho. The network takes the stated confidence at face value. In PC-RFM, founders have latent random reliability and no confidence input is given. The network must infer how far to trust the founder from the observed labels, in context, which amortizes the same quantity that CPR-LP estimates by marginal likelihood.

**D.** Selection under a policy is well studied in recommender debiasing (Schnabel et al. 2016) and in Thompson sampling with misspecified priors (Simchowitz et al. 2021). Those works assume positivity or bound reward loss. We note a complementary and simple identifiability consequence. When a belief determines which segments get outreach, a wrong belief about a never-contacted segment cannot be detected from the labels, and pooled calibration can propagate the error. Exploration is therefore needed for correction.

## Open questions

1. Read the EXNEX original (Wiley/PMC). Confirm whether prior mixture weights are fixed per stratum and the exact form of the NEX component (Daniells 2025 is secondary).
2. Power priors and commensurate priors (Ibrahim and Chen; Hobbs et al.) are cited from memory and were not opened. Open them before citing.
3. PFN literature 2024-26 on user priors was searched only shallowly. Check the TabPFN-with-priors and prior-conditioning work, and the PFNs4BO follow-ups (for example arXiv 2404.04495 and 2610.37737, seen in results but not opened).
4. The KG/recommender search covered only rule-confidence and human-feature work. A deeper search is needed on "knowledge-aware recommendation with trust", "Bayesian cold-start with expert priors" and "weak supervision on graphs".
5. Bandits with expert advice (Exp4-style) and Bayesian elicitation-with-calibration (e.g., calibrating experts, Cooke's method) were not searched in depth. The second is a likely older anticipator for the "calibrate the expert" idea in A.
6. Check PU-learning "selected at random vs. not at random" literature (SAR-PU, Bekker and Davis) for D.

## Addendum (2026-11-24): Hobbs et al. 2011, commensurate priors — read in full

Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC3134568 (author manuscript, Biometrics 67(3):1047–1056), full text
read; Web Appendices not read.
- **Mechanism:** θ ~ N(θ₀, 1/τ); τ ("commensurability") gets a vague prior (simulations: Cauchy(0, 30) on log τ) and
  is learned from current data (Sec. 2.2, Eq. 4; Sec. 3).
- **Scope:** one scalar τ for the whole historical source; in the regression version τ multiplies the identity, so
  one τ governs all coefficients (Sec. 2.4). No per-stratum commensurability; no pooled learning of trust across
  strata; the only mixture (Sec. 3, Eq. 17) uses fixed (τ_j, γ_j) pairs.
- **No calibration slope:** shrinkage is toward equality (θ ≈ θ₀), never θ ≈ a + b·θ₀.
- **No prior-driven selection:** the historical data do not steer which units are observed (adaptive allocation is
  only mentioned as future work in Sec. 6).
- **Correction:** the "(1.6, 82)" figure is the 95% CI of **log τ** in the case study (n₀ = 171, n = 270; Table 3),
  not a trust interval with 60 historical points. `research-map.md` corrected.

**Effect on claim A:** strengthens the defensible differences — continuous calibration slope (not equality
shrinkage), trust pooled across many segments, inside a graph model, with prior-steered labels.
