import numpy as np
from scipy.special import expit

from fprior.cprlp import CPRConfig, fit_predict, laplace_bernoulli
from fprior.metrics import expected_auc, lift_at_k
from fprior.sim import (FounderConfig, MarketConfig, OutreachConfig, make_founder,
                        make_market, run_outreach)

SMALL = MarketConfig(n_accounts=400)


def market(seed=0):
    return make_market(SMALL, np.random.default_rng(seed))


def test_market_base_rate_and_shapes():
    m = market()
    assert abs(m.p.mean() - SMALL.base_rate) < 1e-6
    assert m.graph_cov.shape == (m.n, m.n)
    assert np.allclose(np.mean(np.diag(m.graph_cov)), 1.0)
    assert (m.A != m.A.T).nnz == 0


def test_oracle_founder_equals_true_segment_logits():
    m = market()
    f = make_founder(m, FounderConfig(reliability=1.0), np.random.default_rng(1))
    assert np.allclose(f.segment_logit, m.segment_logit)


def test_founder_reliability_controls_correlation():
    m = market()
    true = m.segment_logit
    for rho in (0.5, 0.0, -0.5):
        f = make_founder(m, FounderConfig(reliability=rho), np.random.default_rng(2))
        assert abs(np.corrcoef(f.segment_logit, true)[0, 1] - rho) < 0.25


def test_outreach_stops_at_n_customers():
    m = market()
    f = make_founder(m, FounderConfig(), np.random.default_rng(3))
    out = run_outreach(m, f, OutreachConfig(n_customers=5), np.random.default_rng(4))
    assert out.n_customers == 5 and out.outcome[-1] == 1
    assert len(set(out.contacted.tolist())) == len(out.contacted)


def test_laplace_mode_is_stationary():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((30, 3))
    K = X @ X.T + 0.5 * np.eye(30)
    m = rng.standard_normal(30)
    y = (rng.random(30) < 0.3).astype(float)
    grad = laplace_bernoulli(K, m, y)
    f = m + K @ grad  # at the mode, f - m = K * grad log p(y|f)
    assert np.allclose(y - expit(f), grad, atol=1e-6)


def test_cpr_nests_prior_only_at_zero_labels():
    m = market()
    f = make_founder(m, FounderConfig(reliability=0.6), np.random.default_rng(6))
    res = fit_predict(m.segment, m.n_segments, m.graph_cov, f.segment_logit,
                      np.array([], dtype=int), np.array([]), CPRConfig())
    ell = f.segment_logit[m.segment]
    assert np.corrcoef(res.logit, ell)[0, 1] > 0.999


def test_learned_trust_drops_for_anti_informative_founder():
    m = make_market(MarketConfig(n_accounts=800), np.random.default_rng(7))
    bs = []
    for rho in (1.0, -0.5):
        f = make_founder(m, FounderConfig(reliability=rho), np.random.default_rng(8))
        out = run_outreach(m, f, OutreachConfig(n_customers=30), np.random.default_rng(9))
        res = fit_predict(m.segment, m.n_segments, m.graph_cov, f.segment_logit,
                          out.contacted, out.outcome, CPRConfig())
        bs.append(res.trust[1])
    assert bs[0] > bs[1]


def test_metrics():
    rng = np.random.default_rng(10)
    p = rng.random(500) * 0.2
    assert expected_auc(p, p) > 0.6
    assert abs(expected_auc(np.zeros(500), p) - 0.5) < 1e-9
    assert lift_at_k(p, p, 50, rng) > 1.5


def test_omega_average_and_expected_p():
    m = make_market(MarketConfig(n_accounts=600), np.random.default_rng(11))
    f = make_founder(m, FounderConfig(reliability=0.8), np.random.default_rng(12))
    out = run_outreach(m, f, OutreachConfig(n_customers=8), np.random.default_rng(13))
    args = (m.segment, m.n_segments, m.graph_cov, f.segment_logit, out.contacted, out.outcome)
    grid = CPRConfig().omega_grid
    avg = fit_predict(*args, CPRConfig(omega_average=True, rank_by="p"))
    assert min(grid) <= avg.omega <= max(grid)
    assert np.all((avg.prob > 0) & (avg.prob < 1))
    # averaged mean lies within the per-omega means, account by account
    singles = np.array([fit_predict(*args, CPRConfig(omega_grid=None, omega=o)).logit for o in grid])
    assert np.all(avg.logit >= singles.min(0) - 1e-9) and np.all(avg.logit <= singles.max(0) + 1e-9)


def test_extra_features_zero_prior_equals_default():
    # a regressor whose coefficient prior has zero variance must leave the model unchanged
    m = market()
    f = make_founder(m, FounderConfig(reliability=0.7), np.random.default_rng(14))
    out = run_outreach(m, f, OutreachConfig(n_customers=8), np.random.default_rng(15))
    args = (m.segment, m.n_segments, m.graph_cov, f.segment_logit, out.contacted, out.outcome)
    base = fit_predict(*args, CPRConfig())
    x = np.random.default_rng(16).standard_normal((m.n, 2))
    same = fit_predict(*args, CPRConfig(extra_sd=0.0), extra_features=x)
    assert np.allclose(base.logit, same.logit) and np.allclose(base.trust, same.trust)
    assert base.coef is None and np.allclose(same.coef, 0.0)


def test_extra_feature_coefficient_learned_positive():
    from fprior.sim_rel import RelMarketConfig, make_rel_market
    rm = make_rel_market(RelMarketConfig(n_accounts=800, novelty=0.0), np.random.default_rng(17))
    f = make_founder(rm, FounderConfig(reliability=0.0), np.random.default_rng(18))
    out = run_outreach(rm, f, OutreachConfig(n_customers=40), np.random.default_rng(19))
    x = (rm.relational - rm.relational.mean()) / rm.relational.std()   # truly predictive (hidden)
    res = fit_predict(rm.segment, rm.n_segments, rm.graph_cov, f.segment_logit,
                      out.contacted, out.outcome, CPRConfig(), extra_features=x[:, None])
    assert res.coef[0] > 0.2


def test_robust_large_nu_matches_gaussian():
    """Student-t with nu -> infinity keeps every segment at omega0: same as the Gaussian model."""
    m = make_market(MarketConfig(n_accounts=600), np.random.default_rng(21))
    f = make_founder(m, FounderConfig(reliability=0.7), np.random.default_rng(22))
    out = run_outreach(m, f, OutreachConfig(n_customers=10), np.random.default_rng(23))
    args = (m.segment, m.n_segments, m.graph_cov, f.segment_logit, out.contacted, out.outcome)
    g = fit_predict(*args, CPRConfig())
    r = fit_predict(*args, CPRConfig(robust_nu=1e9))
    assert np.max(np.abs(g.logit - r.logit)) < 1e-4


def test_robust_prior_corrects_a_confidently_wrong_segment():
    """Founder believes the best segment is 3 logits worse; 30 contacts there + 150 elsewhere.
    The Student-t prior must leave a much smaller error on that segment than the Gaussian prior."""
    from fprior.explore import pessimistic_founder, posterior
    errs = {None: [], 3.0: []}
    for seed in range(3):
        m = make_market(MarketConfig(n_accounts=1200), np.random.default_rng(seed))
        f = pessimistic_founder(m)
        hid = int(np.argmax(f.segment_logit - m.segment_logit < -1.0))
        rng = np.random.default_rng(seed + 100)
        in_h = np.flatnonzero(m.segment == hid)
        c = np.concatenate([rng.choice(in_h, size=min(30, len(in_h)), replace=False),
                            rng.choice(np.flatnonzero(m.segment != hid), 150, replace=False)])
        for nu in errs:
            mean, var, _ = posterior(m, f.segment_logit, c, m.y[c], CPRConfig(), 0.5, nu)
            assert np.all(np.isfinite(mean)) and np.all(var > 0)
            errs[nu].append(abs(mean[m.segment == hid].mean() - m.segment_logit[hid]))
    assert np.mean(errs[3.0]) < np.mean(errs[None]) - 0.3
