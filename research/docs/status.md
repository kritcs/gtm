# Research status and handoff (updated 2026-11-24, ~21:00)

Read this first when picking up in a new session. Background: `../CLAUDE.md`, `../PLAN.md`, `research-map.md`.

## Results so far (all committed under `results/`)

| Result | File | One-line takeaway |
|---|---|---|
| M0 sanity ladder | `m0_sanity.md` | Oracle founder helps CPR-LP at n = 0–20 (gate passes) |
| M1 PC-RFM v0 | `m1_pfn_v0_final.md` | H1/H2 hold in sim; vs CPR-LP: ties with good founders, ahead with bad ones |
| Relational baselines | `rel_baselines.md` | Novelty knob works; plain CPR-LP can't use product similarity |
| CPR-LP-rel | `rel_cprlp.md` | Learned coefficient on similar-adopters: 0.59 for near-clones, 0 for new categories |
| H4 exploration | `explore_h4.md` | Greedy never fixes a wrong belief about an avoided segment; Thompson partly does |
| H4 robust prior | `explore_h4_robust.md`, `robust_ladder.md` | Student-t segment prior: washout 7/10 vs 5/10, error halved, neutral elsewhere; greedy still stuck → "robust prior + exploration" |
| **Real outcomes (UCI Bank)** | `real_bank.md` | R1–R3 hold: both models gain with good founders, lose little with bad ones |
| **Placebo priors** | `placebo_bank.md`, `diag_placebo_selection.md` | CPR-LP passes. PC-RFM gains ~0.3 lift even with shuffled beliefs → its "bad-founder advantage" is mostly being a better few-label learner, not trust. Same-model noise-belief control now standard |
| Sim-to-sim stress | `stress_sim.md` | PC-RFM robust to most shifts; weak when segments are pure interactions (fix: `--random-seg`) |
| **RelBench audit** | `audit_relbench.md`, `docs/benchmark-audit.md` | No-learning repeat+popularity heuristic = 93% / 88% of best published MAP; novel pairs (82–93% of positives) scored ≈ 0 by everything |
| Novelty check | `docs/novelty-check.md` | A partly anticipated (EXNEX etc.), B novel, C defensible vs PFNs4BO, D a proposition |
| Pre-registrations | `m2_preregistration.md`, script docstrings | M2 (v1/v2), H4, real_bank, placebo, stress, audit, SBIR |

## Running now (as of ~21:00)

- **Colab `pfn-f` (L4)**: v1, v1-nf, v2, v2-nf (commit `dda1d0e`), 20k steps each, done ~23:15–23:35.
  A local background job polls the VM, downloads all four to `runs/`, and runs `colab stop -s pfn-f`
  only if every checkpoint is at step 20000. If the session ended before that, check `colab sessions`.
- **SBIR outcome pull** (local, ~3 h): `scripts/data_sbir_outcomes.py --n 100000` → details → stats,
  writes `data/interim/`. Then run `scripts/sbir_eval.py` (pre-registered SB1–SB3).

## Next steps (in order)

1. When v1/v2 land: `scripts/eval_pfn_rel.py` (v1 vs v1-nf vs v0; v2 vs v2-nf vs v1), then score P1–P6 in
   `results/m2_preregistration.md`. Includes the same-model noise-belief control.
2. Next GPU round (launcher prepared in the session scratchpad; rebuild from `colab_launch_v1.py` pattern):
   PC-RFM v0.1 with `--random-seg` (+ no-founder), commit `d5ceb18` or later. Then re-run stress test S2 and real_bank.
3. When SBIR pull finishes: `scripts/sbir_eval.py`; report funding-agency vs new-agency slices.
4. Benchmark audit next: train a GNN/two-tower on rel-hm and score it per slice (novel slice is the claim).
5. Founder-belief tokens + fine-tuning on real-outcome tasks (PLAN §5).
6. Founder cohort: draft elicitation form + one-page study description (needs IRB; start early).
7. EXNEX original: read when the PDF is in `docs/papers/`.

## How to pick up Colab

```bash
cd /Users/peemmac/gtm/research
export SSL_CERT_FILE=$(.venv/bin/python -c "import certifi; print(certifi.where())")
C=.venv/bin/colab
$C sessions; $C usage
$C ssh -s pfn-f -- 'for f in pfn_v1 pfn_v1_nf pfn_v2 pfn_v2_nf; do echo $f; tail -n1 /content/runs/$f/log.jsonl; done'
for r in pfn_v1 pfn_v1_nf pfn_v2 pfn_v2_nf; do mkdir -p runs/$r; for f in model.pt log.jsonl config.json; do $C download -s pfn-f /content/runs/$r/$f runs/$r/$f; done; done
$C stop -s pfn-f
.venv/bin/python scripts/eval_pfn_rel.py --ckpt runs/pfn_v1/model.pt --ckpt-nf runs/pfn_v1_nf/model.pt --ckpt-v0 runs/pfn_v0/model.pt --out results/m2_pfn_v1.md
.venv/bin/python scripts/eval_pfn_rel.py --ckpt runs/pfn_v2/model.pt --ckpt-nf runs/pfn_v2_nf/model.pt --ckpt-v1 runs/pfn_v1/model.pt --out results/m2_pfn_v2.md
```

## Gotchas

- `colab exec` drops its client connection on long runs (even with `--timeout 0`); training continues on the VM. Poll with `colab ssh`.
- One L4 session at a time on this account; A100 was unavailable.
- Real-data tiers must randomly relabel category indices per market (PC-RFM can otherwise score by index coincidence).
- `research/data/` is gitignored (SBIR CSV 367 MB, RelBench, Bank). Never commit data.
- If the Mac was set to no-sleep: `sudo pmset -a disablesleep 0` to undo.

## Work log (append only; every agent)

- 2026-11-24 22:50 — Claude: handed SBIR (track A) and benchmark-audit extension (track B) to Codex; see docs/handoff-codex.md. Claude keeps PC-RFM / Colab (v1/v2 eval + M2 scoring, round 2: v0.1 random-seg, no-founder, founder tokens, rho-min -1).
