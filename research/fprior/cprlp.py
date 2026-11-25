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

The segment-deviation scale omega (per-segment trust) is integrated over a grid with a
uniform hyperprior and Laplace-evidence weights. Plugging in the argmax was tried first
and lost badly at n = 1 (it picks large omega from one or two labels); see
scripts/diag_oracle.py.

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
    # True: average predictions over omega_grid with Laplace-evidence weights (uniform
    # hyperprior on the grid). False: plug in the argmax (type-II ML).
    omega_average: bool = True
    # "logit": rank by posterior mean logit. "p": rank by posterior E[sigmoid(x)], the
    # Bayes action for an expected-hit-rate metric such as lift@k.
    rank_by: str = "logit"
    # prior sd of the coefficient on each optional extra regressor (fit_predict extra_features)
    extra_sd: float = 0.5
    # Heavy-tailed segment deviations: xi_k ~ N(0, omega_k^2), omega_k^2 ~ Inv-Gamma(nu/2, nu*omega^2/2),
    # i.e. a Student-t(nu) prior with scale omega (the grid value). None = Gaussian (original model).
    robust_nu: float | None = None
    robust_iters: int = 15
    # cap on each segment's deviation sd under robust_nu (a truncated hyperprior). Without it, heavy
    # tails (nu = 1 tried) let EM inflate the scale of a quasi-separated segment without bound.
    robust_max_sd: float = 3.0


@dataclass
class CPRResult:
    logit: np.ndarray            # posterior mean logit for every account (mixture mean)
    trust: np.ndarray | None     # posterior mean of (a, b), if the founder is used
    omega: float = float("nan")  # omega used (argmax), or its posterior mean (averaging)
    prob: np.ndarray | None = None   # posterior E[p] per account (probit approximation)
    rank_by: str = "logit"
    coef: np.ndarray | None = None   # posterior mean of extra-regressor coefficients, if any

    @property
    def score(self) -> np.ndarray:
        """What to rank accounts by, per the config's rank_by."""
        return self.prob if self.rank_by == "p" else self.logit


def laplace_bernoulli(K: np.ndarray, m: np.ndarray, y: np.ndarray,
                      iters: int = 100, tol: float = 1e-9) -> np.ndarray:
    """Mode of p(f | y) for f ~ N(m, K), y ~ Bernoulli(sigmoid f). Returns grad log p(y|f_hat)."""
    return laplace_bernoulli_evidence(K, m, y, iters, tol)[0]


def laplace_bernoulli_evidence(K: np.ndarray, m: np.ndarray, y: np.ndarray,
                               iters: int = 100, tol: float = 1e-9) -> tuple[np.ndarray, float]:
    """GPML Algorithm 3.1 with a prior mean (Newton on a strictly concave objective).

    Returns (grad log p(y | f_hat), Laplace approximation to log p(y)) — GPML eq. 3.32.
    """
    grad, log_z, _, _ = _laplace(K, m, y, iters, tol)
    return grad, log_z


def _psi(a, f, m, y):
    """Laplace objective: log p(y | f) - 0.5 a^T (f - m), with f = K a + m."""
    return float(np.sum(y * f - np.logaddexp(0.0, f))) - 0.5 * float(a @ (f - m))


def _laplace(K, m, y, iters=100, tol=1e-9, damped=False):
    """Returns (grad, log_z, sqrt(W), chol(I + sW K sW)) at the mode.

    damped=True adds step-halving on the Laplace objective (needed when K has large blocks, e.g.
    robust segment scales); the default undamped path is unchanged (GPML Algorithm 3.1)."""
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
        a_new = b - sW * cho_solve((Lc, True), sW * (K @ b))
        f_new = K @ a_new + m
        if damped:
            psi_old = _psi(a, f, m, y)
            t = 1.0
            while _psi(a_new, f_new, m, y) < psi_old - 1e-12 and t > 1e-6:
                t *= 0.5
                a_new = a + t * (a_new - a)
                f_new = K @ a_new + m
        a = a_new
        done = np.max(np.abs(f_new - f)) < tol
        f = f_new
        if done:
            break
    pi = expit(f)
    sW = np.sqrt(pi * (1 - pi))
    Lc = np.linalg.cholesky(np.eye(n) + sW[:, None] * K * sW[None, :])
    loglik = float(np.sum(y * np.log(pi + 1e-300) + (1 - y) * np.log(1 - pi + 1e-300)))
    log_z = -0.5 * float(a @ (f - m)) + loglik - float(np.sum(np.log(np.diag(Lc))))
    return y - pi, log_z, sW, Lc


def robust_segment_scales(seg_c: np.ndarray, n_segments: int, rest_cc: np.ndarray, m_c: np.ndarray,
                          y: np.ndarray, omega0: float, nu: float, iters: int = 15, tol: float = 1e-6,
                          max_sd: float = 3.0):
    """EM for Student-t segment deviations (scale mixture of normals).

    Kernel on contacted accounts: K = sum_k omega_k^2 z_k z_k^T + rest, z_k = indicator of segment k.
    Under the Laplace approximation, xi_k is jointly Gaussian with f_c, so
        E[xi_k | y]   = omega_k^2 z_k^T g                     (g = grad log p(y | f_hat))
        Var[xi_k | y] = omega_k^2 - omega_k^4 z_k^T (K + W^-1)^-1 z_k
                      = omega_k^2 - omega_k^4 || L^-1 (sqrt(W) z_k) ||^2,   L L^T = I + sW K sW
    M-step (inverse-gamma hyperprior): omega_k^2 <- (nu omega0^2 + E[xi_k^2]) / (nu + 1).
    Segments without contacts stay at omega0^2. omega_k is capped at max_sd (truncated hyperprior).
    Returns (omega_k^2 for all segments, final fit).
    """
    om2 = np.full(n_segments, omega0 ** 2)
    present = np.unique(seg_c)
    Z = (seg_c[:, None] == present[None, :]).astype(float)            # (n, |present|)
    same = (seg_c[:, None] == seg_c[None, :]).astype(float)
    fit = None
    for _ in range(iters):
        K = same * om2[seg_c][None, :] + rest_cc
        grad, log_z, sW, Lc = _laplace(K, m_c, y, iters=200, damped=True)
        fit = (grad, log_z, sW, Lc, K)
        mk = om2[present] * (Z.T @ grad)
        A = np.linalg.solve(Lc, sW[:, None] * Z)                       # L^-1 sW Z
        vk = om2[present] - om2[present] ** 2 * np.sum(A * A, axis=0)
        new = np.minimum((nu * omega0 ** 2 + mk ** 2 + np.maximum(vk, 0.0)) / (nu + 1.0), max_sd ** 2)
        done = np.max(np.abs(new - om2[present])) < tol
        om2[present] = new
        if done:
            break
    K = same * om2[seg_c][None, :] + rest_cc
    grad, log_z, sW, Lc = _laplace(K, m_c, y, iters=200, damped=True)
    return om2, (grad, log_z, sW, Lc, K)


def _expected_p(mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """E[sigmoid(x)], x ~ N(mean, var), probit approximation (MacKay 1992)."""
    return expit(mean / np.sqrt(1.0 + np.pi * np.maximum(var, 0.0) / 8.0))


def fit_predict(segment: np.ndarray, n_segments: int, graph_cov: np.ndarray,
                founder_segment_logit: np.ndarray | None, contacted: np.ndarray,
                outcome: np.ndarray, cfg: CPRConfig,
                extra_features: np.ndarray | None = None) -> CPRResult:
    """extra_features: optional (N, q) standardized public regressors (e.g. relational scores).
    Each enters the mean linearly with coefficient ~ N(0, extra_sd^2), integrated out exactly
    like the founder trust (appended columns of H, block of S). None = original model."""
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

    p0, q = len(mu), 0
    if extra_features is not None:
        X = np.asarray(extra_features, dtype=float).reshape(N, -1)
        q = X.shape[1]
        H = np.column_stack([H, X])
        mu = np.concatenate([mu, np.zeros(q)])          # zero prior mean: m is unchanged
        S2 = np.zeros((p0 + q, p0 + q))
        S2[:p0, :p0] = S
        S2[p0:, p0:] = cfg.extra_sd ** 2 * np.eye(q)
        S = S2

    grid = tuple(cfg.omega_grid) if cfg.omega_grid else (cfg.omega,)
    need_var = cfg.rank_by == "p"
    # prior variance of each account's logit, excluding the omega term
    prior_var_rest = (cfg.graph_sd ** 2 * np.diag(graph_cov) + cfg.nugget_sd ** 2
                      + np.einsum("ij,jk,ik->i", H, S, H)) if need_var else None

    if len(contacted) == 0:
        # no evidence: averaging uses the (uniform) hyperprior; argmax falls back to cfg.omega
        omegas = np.array(grid if cfg.omega_average else (cfg.omega,))
        w = np.full(len(omegas), 1.0 / len(omegas))
        prob = (sum(wj * _expected_p(m, prior_var_rest + o ** 2) for wj, o in zip(w, omegas))
                if need_var else None)
        return CPRResult(m, mu[:p0] if cfg.use_founder else None, float(w @ omegas), prob, cfg.rank_by,
                         mu[p0:] if q else None)

    c = contacted
    same_seg = (segment[:, None] == segment[c][None, :]).astype(float)
    rest = cfg.graph_sd ** 2 * graph_cov[:, c] + H @ S @ H[c].T
    rest[c, np.arange(len(c))] += cfg.nugget_sd ** 2
    fits = []
    for omega in grid:
        if cfg.robust_nu is None:
            K_all_c = omega ** 2 * same_seg + rest
            grad, log_z, sW, Lc = _laplace(K_all_c[c], m[c], outcome)
            om_acc = None
        else:
            om2, (grad, log_z, sW, Lc, _) = robust_segment_scales(
                segment[c], n_segments, rest[c], m[c], outcome, omega, cfg.robust_nu, cfg.robust_iters,
                max_sd=cfg.robust_max_sd)
            K_all_c = same_seg * om2[segment[c]][None, :] + rest
            om_acc = om2[segment]                                       # per-account omega^2
        fits.append((log_z, omega, grad, K_all_c, sW, Lc, om_acc))
    log_zs = np.array([f[0] for f in fits])
    if cfg.omega_average:
        w = np.exp(log_zs - log_zs.max())
        w /= w.sum()
    else:
        w = np.zeros(len(fits))
        w[int(np.argmax(log_zs))] = 1.0

    logit = np.zeros(N)
    prob = np.zeros(N) if need_var else None
    beta = np.zeros(len(mu))
    for wj, (_, omega, grad, K_all_c, sW, Lc, om_acc) in zip(w, fits):
        if wj == 0.0:
            continue
        mean_j = m + K_all_c @ grad
        logit += wj * mean_j
        if need_var:
            V = cho_solve((Lc, True), (sW[:, None] * K_all_c.T))  # (n, N)
            om_term = omega ** 2 if om_acc is None else om_acc
            var_j = prior_var_rest + om_term - np.einsum("ij,ij->j", sW[:, None] * K_all_c.T, V)
            prob += wj * _expected_p(mean_j, var_j)
        beta += wj * (mu + S @ H[c].T @ grad)
    omega_used = float(w @ np.array([f[1] for f in fits]))
    trust = beta[:p0] if cfg.use_founder else None
    return CPRResult(logit, trust, omega_used, prob, cfg.rank_by, beta[p0:] if q else None)
