# Research: founder priors for low-label customer prediction

Plan: [PLAN.md](PLAN.md) · literature and math: [docs/research-map.md](docs/research-map.md)

## M0 (current)

- `fprior/sim.py` — simulator v0: market (segments + DC-SBM account graph + graph-smooth latent), founder with reliability ρ (ρ = 1 is the oracle), belief-driven outreach until n customers are found.
- `fprior/cprlp.py` — CPR-LP v2 baseline: Laplace GP posterior centred on the founder prior, trust (a, b) integrated out exactly, segment-deviation scale ω chosen by Laplace evidence. Nests labels-only and fixed-trust variants.
- `scripts/sanity_ladder.py` — sanity ladder steps 1–2; writes `results/m0_sanity.{md,json}`.

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
.venv/bin/python scripts/sanity_ladder.py --seeds 10   # ~1 min on a laptop CPU
```
