Founder benefit for PC-RFM measured against its OWN no-information mode (UCI Bank Marketing).

After the placebo tests (results/diag_placebo_selection.md), the fair founder-benefit comparison for
PC-RFM is the same network shown (a) the founder's beliefs vs (b) the same values scrambled across
accounts (no information). Same markets, outreach and relabelling as scripts/real_bank.py.

Pre-registered (committed before running):
  NC1. ρ = 1: true − noise > 0 at every n in {0, 1, 5, 10, 20} (> 2 SE).
  NC2. ρ = 0.5: true − noise > 0 at n ≤ 5.
  NC3. ρ ∈ {0, −0.5}, n ≥ 5: |true − noise| < 0.2 (a useless founder adds nothing beyond the model's
       own label learning; an inverted one may cost a little at small n).

## Results (30 seeds): runs/pfn_tok/model.pt, noise = segment; true beliefs − noise beliefs (lift@100)

| ρ | n = 0 | n = 1 | n = 5 | n = 10 | n = 20 | n = 50 |
|---|---|---|---|---|---|---|
| 1.0 | +2.28 ± 0.14 | +1.96 ± 0.18 | +1.17 ± 0.15 | +0.93 ± 0.15 | +0.82 ± 0.11 | +0.48 ± 0.08 |
| 0.5 | +1.01 ± 0.19 | +0.92 ± 0.18 | +0.33 ± 0.18 | +0.48 ± 0.13 | +0.32 ± 0.11 | +0.14 ± 0.07 |
| 0.0 | -0.14 ± 0.14 | -0.18 ± 0.13 | +0.40 ± 0.15 | +0.04 ± 0.17 | -0.02 ± 0.11 | +0.06 ± 0.08 |
| -0.5 | -0.42 ± 0.12 | -0.08 ± 0.14 | +0.01 ± 0.10 | +0.29 ± 0.18 | -0.14 ± 0.14 | +0.25 ± 0.12 |
## Scoring (written after the run)

Deviation from the original control: `pfn_tok` takes beliefs only as per-segment tokens, so the original
"scramble values across accounts" noise cannot be fed to it (the first attempt crashed). Noise here = the 32
segment beliefs shuffled across segments (`NOISE=segment`, added to the script after the crash). For a
like-for-like comparison v0 was rerun under the same control: `bank_noise_control_v0_segnoise.md`.

| check | v0 (account noise, original) | v0 (segment noise) | tok (segment noise) |
|---|---|---|---|
| NC1 ρ = 1, n ≤ 20, > 2 SE | holds | holds | holds |
| NC2 ρ = 0.5, n ≤ 5, > 2 SE | n = 5 fails | n = 5 fails (+0.16 ± 0.15) | n = 5 fails (+0.33 ± 0.18) |
| NC3 \|true − noise\| < 0.2, ρ ≤ 0, n ≥ 5 | fails (−0.29, −0.23: inverted costs) | fails (+0.32, +0.22, +0.23) | fails (+0.40, +0.29) |

Findings:
1. **Founder value is the same for tok and v0** under the same control (ρ = 1: +2.28 vs +2.34 at n = 0,
   +0.82 vs +0.96 at n = 20). So the token model's extra lift with bad founders in `real_bank_tok.md`
   (e.g. ρ = −0.5, n = 10: 1.70 vs 1.32) is **not** founder use; it is a better few-label learner (most likely
   the `--random-seg` training prior). Report it as such.
2. **The NC3 "inverted founder costs" failure for v0 depended on the control.** Against a segment-shuffled
   founder the inverted founder costs nothing (v0, ρ = −0.5, n = 10/20: +0.02 / −0.01 vs −0.29 / −0.23 against
   account-scrambled noise). Account-scrambled beliefs are easy to recognise as noise and ignore; a structured
   but wrong founder is not. Which null is right depends on the claim: "the model ignores useless input" →
   account noise; "a wrong founder is no worse than a random founder" → segment noise. Both are now reported.
3. NC3 fails under segment noise in the **positive** direction (true wrong beliefs beat shuffled ones by
   0.2–0.4 at a few cells): consistent with the outreach-coupled selection effect seen in
   `diag_placebo_selection.md`, not a founder-quality effect. Not investigated further here.
