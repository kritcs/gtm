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
