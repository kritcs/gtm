# Research status and handoff (updated 2026-11-26, ~03:45)

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
| **Round 4 graph PC-RFM (v0g)** | `round4_preregistration.md`, `eval_v0g.md` | G3/G4 fail: v0g ignores its graph features; one-hop and PPR label features correlate only ~0.05 with the truth, and **CPR-LP gains nothing from its graph in the simulator either** → no simulated result may be credited to the graph |
| Pre-registrations | `m2_preregistration.md`, script docstrings | M2 (v1/v2), H4, real_bank, placebo, stress, audit, SBIR |

## Running now (as of 2026-11-26 ~03:45)

- **Colab `pfn-4e` (T4)**: round-4 `pfn_v0g_nf` and `pfn_v0g_s2`, resumed from step-8500 weights after `pfn-4d`
  was reclaimed (deviation in `results/round4_preregistration.md`); done around 07:00. Watcher
  (scratchpad `watch4.sh`, log `watch4e.log`) keeps the session alive, syncs to `runs/<name>.sync`, copies to
  `runs/<name>` at step 20000 and stops the session. Then run stress_sim / real_bank for G1, G2, G5–G7
  (main v0g = `runs/local_v0g/model.pt`) and write "Scoring, part 2".
- The Mac must stay awake, with the lid open and on power; sleep killed `pfn-4d`.

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
- 2026-11-25 00:55 — Claude: SBIR pipeline (outcomes → details → stats) finished for the full cohort; data in data/interim/ — Track A ready for Codex. Colab session pfn-f was reclaimed before v1/v2 checkpoints were downloaded (lost ~11.6 units). Relaunched v1, v1-nf, v2, founder-tokens on new L4 session pfn-r (commit f505a52; note: not dda1d0e as pre-registered — sampler refactor, same distribution) with checkpoint sync every 20 min to runs/<name>.sync/.
- 2026-11-25 02:50 — Claude: pre-registered founder-token eval (scripts/eval_pfn_tok.py, T1–T5; commit 2dea623) and round 2 (results/round2_preregistration.md, Q1–Q4; commit 8919ff2). Rerun on pfn-r at step ~7.5k/20k, ETA ~04:45; round 2 (v0.1, v0.1-nf, v0.2-rho, v2-nf) launches right after.
- 2026-11-25 05:35 — Claude: rerun finished; all four checkpoints local (runs/pfn_v1, pfn_v1_nf, pfn_v2, pfn_tok). M2 scored (results/m2_preregistration.md): P2, P4, P6 hold; P1 (founder value not larger in new categories), P3 (worst −0.33), P5 (v2 helps only at e = 1, n = 0) fail. Founder tokens (results/m2_pfn_tok.md): T2/T4/T5 hold, T1/T3 fail; bank R1–R3 hold; same-model segment-noise control shows its bank edge is learner quality, not founder use, and v0's NC3 'inverted founder cost' disappears under a segment-shuffle null (results/bank_noise_control_tok.md). Round 2 (v0.1, v0.1-nf, v0.2-rho, v2-nf; commit 8919ff2) training on pfn-r, ETA ~09:00, synced every 20 min.
- 2026-11-25 10:10 — Claude: round 2 scored (results/round2_preregistration.md: Q1c/Q2/Q3b/Q4 hold; Q1a/b fail — random-seg closes ~35% of the S2 gap; Q3a fails in the positive direction — rho-min −1 model learns to invert inverted founders; Q3c fails). Round 3 on A100 (results/round3_preregistration.md): seg-cross embedding does not change the S2 gap (S1 fails); seed replicates show the S2 gap and all M2 verdicts are stable across training seeds (S3, S4 hold); v0.3 bank R1–R3 hold. Colab balance 156.65; no session running.
- 2026-11-25 14:10 — Claude: took Tracks A and B back from Codex (never started; handoff marked superseded). SBIR eval scored (results/sbir_eval.md: SB1 fails — keyword beliefs add nothing; SB2 partly fails — funding rule beats pooled on funding-agency purchases; SB3 holds on point estimate only; popularity best on new-agency purchases); SBIR row in sim_calibration.md (base rate ~9× below training range); 0/50 hand-checked purchases are R&D (docs/sbir-pipeline.md). Audit extension (results/audit_relbench_ext.md, commit 0d435b4 pre-reg): repeat+pop = 63%/107% (rel-trial), 40%/76% (rel-stack) of best published; E1–E3 fail, E4 holds. Published ID-GNN on rel-hm reproduced exactly (2.81; Colab L4 ~9 units): I1–I3 hold, 99.8% of hits are repeats, novel MAP 0.00 (results/idgnn_slices.md). Colab balance 147.49, no session. Gotcha: `pgrep -f X` over `colab ssh` matches its own command line — use `pgrep -f '[i]dgnn'`-style patterns.
- 2026-11-25 16:40 — Claude: ID-GNN extension (pre-reg 276f94e; results/idgnn_ext_scoring.md). rel-stack reproduced (13.38, 10.89; J1 holds, J3 holds, J4 1/2); exploratory: 91% of ID-GNN hits on user-post-comment are the user's own threads, own-thread heuristic = 78% of ID-GNN (scripts/audit_stack_ownthread.py). rel-trial: NaN loss from epoch 1 twice (second time fresh cache) — J1 fails as 'could not reproduce', J2/J4 not scored; not pursued. Colab sessions all stopped. Gotcha: running 4 RelBench ID-GNN jobs at once on L4 high-mem (52 GB) OOM-kills the rel-trial ones.
- 2026-11-25 18:15 — Claude: doing all four next steps. (1) Paper section draft `docs/paper/sec-audit.md`. (2) Founder cohort v0.2 (`docs/cohort/`: analysis-plan, irb-packet, updated form/description; `scripts/cohort_scrub.py` + tests; `scripts/cohort_power.py` → results/cohort_power.md): evaluating on founder-chosen later contacts has ~15% power even at 50 companies; a 40-account randomized slice gives 56% / 76% at 30 / 50 companies → target 40–50 companies. (3) Round 4 graph-aware PC-RFM v0g (`--graph`; pre-reg results/round4_preregistration.md, commit b99ddd1) training on Colab L4 session pfn-4 (A100 unavailable), synced every 20 min by scratchpad sync_any.sh, which stops the session at step 20000. (4) Second real tier Hillstrom (`load_hillstrom`, `real_bank.py --dataset`): H1 fails (3–4/5 cells), H2 fails narrowly (−0.21 at ρ = −0.5, n = 5), H3 holds 24/24 — sim-to-real replicates. Real-outcome fine-tuning (`scripts/finetune_pfn.py`, pre-reg results/finetune_preregistration.md F1–F6) running locally on CPU (4 models).
- 2026-11-25 21:50 — Claude: fine-tuning scored (results/finetune_preregistration.md): F1/F2/F3/F5/F6 hold, F4 fails as expected. In-domain Bank gain (+0.61 lift) is population-specific: fine-tuned models learn the segment size–rate relation, which flips sign between Bank (−0.36) and Hillstrom (+0.37); cross-dataset fine-tuning gives nothing. Round 4 (v0g) lost three Colab sessions (Mac sleep re-enabled → no keepalive → runtime reclaimed after ~15–20 min; DNS outage; CLI auth expiry); now training locally on MPS (runs/pfn_v0g, _nf, _s2; caffeinate keeps the Mac awake while they run; logs in the session scratchpad). Colab balance ~141.
- 2026-11-26 03:45 — Claude: round 4 part 1 scored. Main v0g (local MPS) done; G3 (+0.024 ± 0.015) and G4
  (+0.004 ± 0.014) fail. Diagnostics (`scripts/diag/`): v0g ignores its graph on its own prior; neighbour-label
  signal corr ≈ 0.045 (PPR up to α = 0.95: ≈ 0.05); CPR-LP with graph vs no graph/permuted graph: ≈ 0 on S0 and S2.
  The graph is unlearnable at n ≤ 20 in this simulator, and the round-4 premise was wrong. Colab nf/s2 resumed on
  `pfn-4e` (`train_pfn.py --resume`, now with exact-resume checkpoints).

