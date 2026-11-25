import numpy as np

from fprior.metrics import expected_auc
from fprior.sim import FounderConfig, OutreachConfig, make_founder, run_outreach
from fprior.sim_rel import RelMarketConfig, make_rel_market, similar_adopter_score

CFG = RelMarketConfig(n_accounts=800)


def test_shapes_and_base_rate():
    m = make_rel_market(CFG, np.random.default_rng(0))
    assert m.adoption.shape == (CFG.n_accounts, CFG.n_products)
    assert abs(m.p.mean() - CFG.base_rate) < 1e-6
    assert abs(m.degree.mean() - CFG.adoptions_per_account) < 1.5
    assert (m.A != m.A.T).nnz == 0


def test_novelty_kills_the_similar_product_path():
    def auc(e):
        return np.mean([expected_auc(similar_adopter_score(m), m.p) for m in
                        (make_rel_market(RelMarketConfig(n_accounts=800, novelty=e), np.random.default_rng(s))
                         for s in range(3))])
    clone, new = auc(0.0), auc(1.0)
    assert clone > 0.62 and abs(new - 0.5) < 0.05


def test_founder_and_outreach_work_unchanged():
    m = make_rel_market(CFG, np.random.default_rng(1))
    f = make_founder(m, FounderConfig(reliability=1.0), np.random.default_rng(2))
    assert np.allclose(f.segment_logit, m.segment_logit)
    out = run_outreach(m, f, OutreachConfig(n_customers=5), np.random.default_rng(3))
    assert out.n_customers == 5
