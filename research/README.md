# Founder priors for low-label customer discovery

Research project (work in progress) by Peem Puangmaha, Stanford CS. It grew out of *Intuition in a bottle*, a
hackathon go-to-market engine (repository root).

**Problem.** A startup looking for its first customers has 5–50 labelled outcomes; commercial lead scoring needs
50–1,000+. Its most predictive information is what the founder believes about who will buy, which is often wrong.
Rule-based tools encode those beliefs but never learn; ML learns but cannot use them.

**Approach.** Treat founder beliefs as a *revisable prior* and learn how far to trust them.

- **CPR-LP:** closed-form Bayesian model. A Laplace-approximated Gaussian-process posterior over account logits,
  centred on a calibrated founder prior, with founder trust integrated out exactly. It nests the labels-only model.
- **PC-RFM:** a prior-fitted transformer (PFN-style). Trained on millions of simulated markets, each with a
  simulated founder of random reliability, it learns in context how much to trust the founder.

All experiments are pre-registered before they run (`results/*preregistration*.md`), and failures are reported as
failures.

## Results so far

| Finding | Evidence |
|---|---|
| **Sim-to-real.** On two real datasets (UCI Bank Marketing, Hillstrom e-mail test), PC-RFM trained only on simulations stays within 0.2 lift@100 of CPR-LP; on Hillstrom, 24/24 cells with worst gap 0.10 | `results/real_bank.md`, `results/real_hillstrom_v01.md` |
| **A good founder helps.** With real outcomes and a simulated reliable founder, both models gain +0.7 to +2.5 lift@100 over labels only on Bank. A wrong founder costs ≤ 0.2 once n ≥ 5 (it fails narrowly on Hillstrom) | same |
| **Placebo control.** PC-RFM gains about +0.3 lift even from shuffled beliefs, so its "bad-founder advantage" is mostly few-label learning, not trust. CPR-LP passes | `results/placebo_bank.md` |
| **Benchmark audit (RelBench).** A no-learning repeat + popularity heuristic reaches 93% (rel-hm) and 88% (rel-avito) of the best published MAP. On rel-hm, 99.8% of ID-GNN's hits are repeat purchases; novel-pair MAP is about 0 | `results/audit_relbench.md`, `results/idgnn_slices.md`, `docs/benchmark-audit.md` |
| **Fine-tuning on real outcomes.** +0.61 lift in-domain; no transfer across businesses (the size–rate relation flips sign between datasets) | `results/finetune_preregistration.md` |
| **Negative: the account graph.** A graph-aware PC-RFM ignores its graph features; with ≤ 20 customers even the exact Bayesian model gains nothing from the simulated graph unless it is strong and smooth | `results/round4_preregistration.md`, `results/diag_graph_regime.md` |

Next: a prospective cohort of 40–50 real founders (`docs/cohort/`).

## Layout

- `fprior/`: simulator (`sim.py`, shifts), CPR-LP (`cprlp.py`), PC-RFM prior, model and training data
  (`pfn/`), real-data loaders.
- `scripts/`: training (`train_pfn.py`), evaluations, audits, diagnostics.
- `results/`: every pre-registration and scored result.
- `docs/`: research map, paper section drafts, cohort study materials, status log.

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
.venv/bin/python scripts/sanity_ladder.py --seeds 10   # ~1 min on a laptop CPU
```

Raw datasets are not redistributed. Download UCI Bank Marketing and the MineThatData Hillstrom file into
`data/raw/` (paths in `fprior/real_bank.py`); RelBench data comes through its package.
