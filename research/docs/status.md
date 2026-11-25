# Research status and handoff (2026-11-24, evening)

Read this first when picking up in a new session. Background: `../CLAUDE.md`, `../PLAN.md`, `research-map.md`.

## Done (all committed)

- **M0** — simulator v0 (`fprior/sim.py`), CPR-LP v2 baseline (`fprior/cprlp.py`, ω averaged by evidence),
  sanity ladder: oracle-founder gate passes (`results/m0_sanity.md`).
- **M1** — PC-RFM v0 (no graph, `fprior/pfn/`), trained 20k steps on Colab L4. Final eval on M0 markets:
  `results/m1_pfn_v0_final.md`. H1 and H2 hold in simulation; vs CPR-LP: slightly behind with good founders
  (−0.55 ± 0.23 at n = 1), ahead with bad founders (up to +0.49).
- **RDL direction** decided (PLAN.md §3). Relational simulator `fprior/sim_rel.py` with a novelty knob.
- **Relational baselines** `results/rel_baselines.md` (pre-registered expectations held) and
  **CPR-LP-rel** `results/rel_cprlp.md` (learned coefficient on similar-product adopters).
- **PC-RFM v1** (relational features, `fprior/pfn/data_rel.py`) and **v2** (account–product encoder,
  `fprior/pfn/model_rel.py`) implemented.
- **M2 pre-registration** `results/m2_preregistration.md` — committed before any v1/v2 results. Score against it.
- **Novelty check** `docs/novelty-check.md`. EXNEX original not yet read (paywalled); PDFs go in
  `docs/papers/` (gitignored).

## Running now

Colab session **`pfn-f`** (L4, ~2.33 compute units/hr), commit `dda1d0e`, four jobs, 20k steps each,
~0.7 s/step, started ~19:50, expected done ~4 h later:

| Job | Output dir on VM |
|---|---|
| v1 (relational features) | `/content/runs/pfn_v1` |
| v1 without founder | `/content/runs/pfn_v1_nf` |
| v2 (learned encoder) | `/content/runs/pfn_v2` |
| v2 without founder | `/content/runs/pfn_v2_nf` |

Each dir has `model.pt` (saved every 500 steps), `log.jsonl`, `config.json`.

## How to pick up

```bash
cd /Users/peemmac/gtm/research
export SSL_CERT_FILE=$(.venv/bin/python -c "import certifi; print(certifi.where())")   # python.org cert workaround
C=.venv/bin/colab
$C sessions                                    # is pfn-f still alive?
$C ssh -s pfn-f -- 'for f in pfn_v1 pfn_v1_nf pfn_v2 pfn_v2_nf; do echo $f; tail -n1 /content/runs/$f/log.jsonl; done; pgrep -f train_pfn.py | wc -l'
# when done (step 20000 in all four logs, or no train_pfn processes):
for r in pfn_v1 pfn_v1_nf pfn_v2 pfn_v2_nf; do mkdir -p runs/$r; for f in model.pt log.jsonl config.json; do $C download -s pfn-f /content/runs/$r/$f runs/$r/$f; done; done
$C stop -s pfn-f                               # stop billing
.venv/bin/python scripts/eval_pfn_rel.py --ckpt runs/pfn_v1/model.pt --ckpt-nf runs/pfn_v1_nf/model.pt \
    --ckpt-v0 runs/pfn_v0/model.pt --out results/m2_pfn_v1.md
.venv/bin/python scripts/eval_pfn_rel.py --ckpt runs/pfn_v2/model.pt --ckpt-nf runs/pfn_v2_nf/model.pt \
    --ckpt-v1 runs/pfn_v1/model.pt --out results/m2_pfn_v2.md
```

`runs/` is gitignored; `runs/pfn_v0*` (M1 checkpoints) exist locally only. Then score P1–P6 in
`results/m2_preregistration.md` (report every prediction, hold or fail).

## Open decisions / next

1. **Exploration loop** (recommended next): Thompson sampling with minimum exploration η; test H4 —
   trust in a segment the founder avoids never corrects at η = 0, does at η > 0; measure washout time.
2. Founder-stated confidence + elicitation format as model input (both CPR-LP and PC-RFM).
3. Read EXNEX original once the PDF is in `docs/papers/`; update `docs/novelty-check.md`.
4. Not yet implemented from the research map: doubly robust PU loss, contractive equilibrium body,
   unbalanced-OT alarm, multiple founders, any theorem checks.

## Gotchas

- `colab exec` client times out after 30 s without output unless `--timeout 0`; training keeps running on the VM anyway.
- Only one L4 session is allowed at a time on this account; A100 was unavailable when tried.
- `sim_rel`: log-degree carries no buying signal (popularity baselines are trivially weak) — decide if realistic.
