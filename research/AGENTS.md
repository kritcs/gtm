# Agent instructions (Codex and other agents)

This folder is a research project. Read, in order: `CLAUDE.md` (project rules and thesis — they apply to every
agent, not just Claude), `docs/status.md` (current state), and your task brief (e.g. `docs/handoff-codex.md`).

## Non-negotiable research rules

- **Pre-register before running.** Every experiment script starts with a docstring stating predictions and the
  hold/fail rule; commit it **before** running the full experiment. Score afterwards by appending a
  "Scoring (written after the run)" section. Never edit predictions after seeing results.
- **Report everything,** including failures and abandoned variants. No tuning to make a prediction hold.
- **Never invent numbers, citations, URLs or dataset fields.** Mark anything unverified.
- **No leakage:** time-based splits; point-in-time features; no LLM-derived beliefs from models that may know the
  outcomes (use keyword rules or chronologically clean models).
- **Data never goes in git.** `research/data/` and `research/runs/` are gitignored; check with `git check-ignore`.

## Working alongside other agents

- Stay inside the files your brief assigns you. If you need to change a shared file (`fprior/cprlp.py`,
  `fprior/sim.py`, `fprior/metrics.py`, `fprior/pfn/**`), make it backward compatible, keep defaults bit-for-bit
  unchanged, and add a test.
- Do **not** start Colab sessions unless your brief says so (the account allows one L4 session at a time).
- Before committing: `git pull --rebase`, run `.venv/bin/pytest -q` in `research/`, commit only your files,
  then push. End commit messages with a `Co-Authored-By:` line for your agent.
- Log what you did at the bottom of `docs/status.md` under "Work log" (append only).

## Environment

- Python: `research/.venv/bin/python` (numpy, scipy, torch, pandas, relbench installed).
- macOS python.org build: for HTTPS from Python set
  `export SSL_CERT_FILE=$(research/.venv/bin/python -c "import certifi; print(certifi.where())")`.
