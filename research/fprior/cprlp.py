"""CPR-LP v2, M0 version: a closed-form graph posterior centred on a calibrated founder prior.

Model over account logits x (research-map.md, "Math: Gaussian processes on graphs"):

    x(v) = lbar + a + b * (l(v) - lbar) + xi_seg(v) + r(v) + e(v)
    (a, b) ~ N(mu_trust, S)          founder calibration: level a, slope ("trust") b
    xi_k   ~ N(0, omega^2)           segment deviation from the calibrated belief
    r      ~ N(0, sigma_g^2 C)       graph-smooth residual, C = GMRF covariance
    e      ~ N(0, sigma_n^2 I)       nugget
    y_v | x_v ~ Bernoulli(sigmoid(x_v)) on contacted accounts only

l(v) is the founder's stated logit for v's segment and lbar its mean. (a, b) enter
linearly, so they are integrated out exactly by adding H S H^T to the kernel
(GPML section 2.7); the Bernoulli likelihood is handled by a Laplace approximation.

Nesting (failure mode 2 in CLAUDE.md):
- labels-only: no founder term (H = [1]); same kernel otherwise.
- fixed trust: S = 0 with (a, b) = (0, 1), i.e. the founder prior is never recalibrated.
- learned trust: S > 0, so the data can shrink b toward 0 or flip it.

Selection: outreach depends only on the founder's beliefs, which the model conditions
on, so the selection is ignorable for this Bayesian posterior. That breaks once
outreach depends on anything unobserved; the PU / propensity machinery is for that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve
from scipy.special import expit


@dataclass(frozen=True)
class CPRConfig:
    use_founder: bool = True
    learn_trust: bool = True
    omega: float = 1.0                       # segment deviation sd (used when no labels / no grid)
    # empirical-Bayes grid for omega (per-segment trust), chosen by Laplace evidence; None = fixed
    omega_grid: tuple[float, ...] | None = (0.05, 0.15, 0.3, 0.5, 0.75, 1.0, 1.5)
    graph_sd: float = 0.8
    nugget_sd: float = 0.7
    trust_mean: tuple[float, float] = (0.0, 0.5)   # population prior on (a, b)
    trust_sd: tuple[float, float] = (0.5, 0.5)
    intercept_mean: float = -2.2             # labels-only level prior (logit ~0.1)
    intercept_sd: float = 1.5


@dataclass
class CPRResult:
    logit: np.ndarray            # posterior mean logit for every account
    trust: np.ndarray | None     # posterior mean of (a, b), if the founder is used
    omega: float = float("nan")  # segment-deviation sd used (selected if a grid was given)


def laplace_bernoulli(K: np.ndarray, m: np.ndarray, y: np.ndarray,
                      iters: int = 100, tol: float = 1e-9) -> np.ndarray:
    """Mode of p(f | y) for f ~ N(m, K), y ~ Bernoulli(sigmoid f). Returns grad log p(y|f_hat)."""
    return laplace_bernoulli_evidence(K, m, y, iters, tol)[0]


def laplace_bernoulli_evidence(K: np.ndarray, m: np.ndarray, y: np.ndarray,
                               iters: int = 100, tol: float = 1e-9) -> tuple[np.ndarray, float]:
    """GPML Algorithm 3.1 with a prior mean (Newton on a strictly concave objective).

    Returns (grad log p(y | f_hat), Laplace approximation to log p(y)) — GPML eq. 3.32.
    """
    n = len(y)
    f = m.copy()
    a = np.zeros(n)
    for _ in range(iters):
        pi = expit(f)
        W = pi * (1 - pi)
        sW = np.sqrt(W)
        B = np.eye(n) + sW[:, None] * K * sW[None, :]
        Lc = np.linalg.cholesky(B)
        b = W * (f - m) + (y - pi)
        a = b - sW * cho_solve((Lc, True), sW * (K @ b))
        f_new = K @ a + m
        done = np.max(np.abs(f_new - f)) < tol
        f = f_new
        if done:
            break
    pi = expit(f)
    sW = np.sqrt(pi * (1 - pi))
    Lc = np.linalg.cholesky(np.eye(n) + sW[:, None] * K * sW[None, :])
    loglik = float(np.sum(y * np.log(pi + 1e-300) + (1 - y) * np.log(1 - pi + 1e-300)))
    log_z = -0.5 * float(a @ (f - m)) + loglik - float(np.sum(np.log(np.diag(Lc))))
    return y - pi, log_z


def fit_predict(segment: np.ndarray, n_segments: int, graph_cov: np.ndarray,
                founder_segment_logit: np.ndarray | None, contacted: np.ndarray,
                outcome: np.ndarray, cfg: CPRConfig) -> CPRResult:
    N = len(segment)
    if cfg.use_founder:
        if founder_segment_logit is None:
            raise ValueError("use_founder=True needs founder beliefs")
        ell = founder_segment_logit[segment]
        lbar = ell.mean()
        H = np.column_stack([np.ones(N), ell - lbar])
        if cfg.learn_trust:
            mu = np.asarray(cfg.trust_mean, dtype=float)
            S = np.diag(np.square(cfg.trust_sd))
        else:
            mu, S = np.array([0.0, 1.0]), np.zeros((2, 2))
        m = lbar + H @ mu
    else:
        H = np.ones((N, 1))
        mu = np.array([cfg.intercept_mean])
        S = np.array([[cfg.intercept_sd ** 2]])
        m = np.full(N, cfg.intercept_mean)

    if len(contacted) == 0:
        return CPRResult(m, mu if cfg.use_founder else None, cfg.omega)

    c = contacted
    same_seg = (segment[:, None] == segment[c][None, :]).astype(float)
    rest = cfg.graph_sd ** 2 * graph_cov[:, c] + H @ S @ H[c].T
    rest[c, np.arange(len(c))] += cfg.nugget_sd ** 2
    best = None
    for omega in (cfg.omega_grid or (cfg.omega,)):
        K_all_c = omega ** 2 * same_seg + rest
        grad, log_z = laplace_bernoulli_evidence(K_all_c[c], m[c], outcome)
        if best is None or log_z > best[0]:
            best = (log_z, omega, grad, K_all_c)
    _, omega, grad, K_all_c = best
    logit = m + K_all_c @ grad
    trust = (mu + S @ H[c].T @ grad) if cfg.use_founder else None
    return CPRResult(logit, trust, omega)
