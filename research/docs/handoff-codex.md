# Handoff brief for Codex (2026-11-24, ~22:50)

You are taking over two CPU-only tracks of a research project while Claude continues the GPU / PC-RFM track in
parallel. Read `../AGENTS.md`, `../CLAUDE.md` and `status.md` first. Results so far are summarized in
`paper-outline.md`.

## What the project is (one paragraph)

Startups with 5–50 customers must find more. We treat the founder's beliefs about which customer segments will buy
as a revisable prior, learn how much to trust them from few outcomes, and explore enough to catch them when wrong.
Two models: **CPR-LP** (closed-form Bayesian; `fprior/cprlp.py`, hierarchical variant `fprior/cprlp_hier.py`) and
**PC-RFM** (transformer trained on simulated markets; `fprior/pfn/`). Evaluation must be credible without public
startup outreach data, hence real-outcome tiers, placebo controls and pre-registration.

## Ownership (to avoid conflicts)

| Codex owns | Claude owns (do not edit) |
|---|---|
| `scripts/sbir_*.py`, `scripts/data_sbir_*.py`, `docs/sbir-pipeline.md`, `results/sbir_*` | `fprior/pfn/**`, `scripts/*pfn*.py`, `scripts/train_pfn.py`, `scripts/score_m2.py`, `results/m2_*`, `results/m1_*` |
| `scripts/audit_*.py`, `docs/benchmark-audit.md`, `results/audit_*` | Colab sessions and `runs/` |
| new files you create for these tracks | `docs/paper-outline.md` (append suggestions to `status.md` instead) |

Shared library files (`fprior/cprlp.py`, `fprior/cprlp_hier.py`, `fprior/metrics.py`, `fprior/sim*.py`): only
backward-compatible additions with tests (see AGENTS.md).

## Track A — SBIR first customers (real B2B) — highest priority

State: the pipeline (`docs/sbir-pipeline.md`) built a cohort of 5,225 firms (first SBIR Phase I in 2012–2018) and
was pulling USAspending outcomes for all of them: `scripts/data_sbir_outcomes.py --n 100000` →
`scripts/data_sbir_details.py` → `scripts/data_sbir_stats.py`, writing to `data/interim/`. At handoff the outcome
pull had finished and the detail step was at ~2,000/13,016 records. **It ran as a background job in Claude's
session; if that session ended it may have stopped.** The scripts cache API responses, so re-running resumes:

```bash
cd /Users/peemmac/gtm/research
export SSL_CERT_FILE=$(.venv/bin/python -c "import certifi; print(certifi.where())")
ls -la data/interim/            # check sbir_outcomes_detail.csv.gz / sbir_labels.csv.gz timestamps
.venv/bin/python scripts/data_sbir_details.py && .venv/bin/python scripts/data_sbir_stats.py
```
(Check first whether a run is still going: `pgrep -fl data_sbir`.) Be polite to the API (the scripts sleep between calls).

Then:
1. **Run the pre-registered evaluation** `scripts/sbir_eval.py` (predictions SB1–SB3 in its docstring; committed
   before the data existed). Append a "Scoring (written after the run)" section to `results/sbir_eval.md`. Report
   the funding-agency vs new-agency slices — the new-agency slice is the one that matters.
2. **Add CPR-LP-style pooled trust** as a further method (pre-register first, new script `scripts/sbir_eval_cpr.py`):
   treat each firm's keyword beliefs over sub-agencies as the founder prior and learn the calibration (a, b) pooled
   across *training* firms (time split as in `sbir_eval.py`). This is the "trust pooled across founders" idea.
3. **Add the row for SBIR to `results/sim_calibration.md`**: base rate of purchases per (firm, sub-agency), spread of
   sub-agency effects, how concentrated first purchases are. Bank sat inside the simulator's ranges; SBIR probably
   won't (sparse labels, ~13+ candidates), and saying where it sits matters.
4. Hand-check ~50 "purchase" labels flagged as uncertain in `docs/sbir-pipeline.md` (SBIR / R&D contracts
   miscounted as purchases) and report the error rate.
5. Write findings in `docs/sbir-pipeline.md` (new section) and log in `status.md`.

Constraints: no LLM-derived beliefs (leakage); publish only derived numbers (abstracts are firms' own text);
reviewers will say "government ≠ commercial", so report it honestly as a B2B proxy.

## Track B — benchmark audit extension (CPU)

State: `results/audit_relbench.md` (heuristics, rel-hm and rel-avito) and `results/audit_models_subsample.md`
(item-CF / BPR / LightGCN on a 20k-user rel-hm sample). Finding so far: a no-learning repeat + popularity heuristic
reaches 93% / 88% of the best published MAP; 82–93% of test positives are novel pairs, and nothing tested beats
popularity there. Pre-registered A1 ("most positives memorizable") **failed** — keep reporting that.

Next:
1. **rel-trial** (~683 MB, high repeat rates per the RelBench paper) and, if memory allows, **rel-stack**: rerun
   `scripts/audit_relbench.py` slices + heuristics (pre-register per task first). A contrast task where repeats
   dominate tests whether the critique is specific or general.
2. **Tune the learned models more fairly** on the rel-hm subsample (more epochs, embedding size, item attributes —
   download article attributes if small): the current models reach 1.15 overall vs published ID-GNN 2.81, so the
   claim is only "realistic recommenders". Report the per-slice result for the best-tuned one. Fix seeding so runs
   are reproducible.
3. Do **not** run the published ID-GNN (needs torch_geometric + GPU; Claude will schedule it on Colab).
4. Update `docs/benchmark-audit.md` (Part 3) and log in `status.md`.

## How to report back

Append to `status.md` → "Work log": date/time, what ran, commit hashes, one line per finding (hold/fail for every
pre-registered prediction). Claude reads that file when continuing.
