"""Model & proof: real training code on the synthetic market, precomputed for the demo.

    make sim      (or: uv run python -m sim.build)

Writes sim/out/results.json. Every section is tagged with one badge:

  synthetic  - real training/eval code ran here, on the seeded synthetic market
               (api/market/simulate.py). Shows the method works on data with a
               known planted structure; says nothing about real-world accuracy.
  simulated  - a mock-up of a roadmap step that was NOT run (needs a GPU pod).
               Numbers come from a seeded generator below, chosen to be plausible.
  measured   - results on real public data. There are none yet; the slot stays empty.

Sections:
  market_model   LightGCN vs ALS on company-tool adoption: loss + validation Recall@10
  prior_residual score = logit(founder prior) + correction(x), trained across a time split
  thompson       Beta priors + Thompson sampling over simulated outcome rounds
  backtest       2024 -> 2025-26 Recall@50 with baselines, and the ablation
  planted        did the pipeline recover the patterns planted in the generator?
  voice_dpo      LoRA + DPO voice fine-tune (simulated mock-up)
  maturity       0 customers -> priors, tens -> Beta updates, hundreds -> fine-tune
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import numpy as np

from api import demo
from api.insight.hit import find_hit
from api.insight.ranking import pain_match
from api.insight.stats import bootstrap_recall_ci, pct_rank
from api.market.graph import build_graph, snapshot
from api.market.simulate import (PRODUCT_DESCRIPTION, SIZE_ORDER, TOOLS, World, generate, month_index,
                                 month_label)
from api.ml import backtest as bt
from api.ml.als import ALS
from api.ml.lightgcn import LightGCN
from api.ml.market import MarketModel

OUT = Path(__file__).resolve().parent / "out" / "results.json"
SEED = 7
SYNTH = "Synthetic data: real training code, synthetic market. Shows the method works, not real-world accuracy."
SIMUL = "Simulated: a mock-up of the planned step. No model was trained for this; numbers are from a seeded generator."

# The synthetic founder's intake: belief that a contacted account in the segment converts,
# and how sure they are (Peem's scheme: "fintech, 70%, quite sure" -> Beta(14, 6)).
FOUNDER_BELIEFS = {
    "saas-billing": (0.60, "quite sure"), "fintech": (0.55, "sure"), "martech": (0.50, "sure"),
    "insurtech": (0.40, "hunch"), "hr-tech": (0.30, "hunch"), "ecommerce": (0.20, "hunch"),
    "logistics": (0.10, "hunch"), "healthtech": (0.10, "hunch"), "devtools": (0.10, "hunch"),
    "edtech": (0.10, "hunch"), "ml-platform": (0.05, "hunch"),
}
STRENGTH = {"quite sure": 20, "sure": 12, "hunch": 4}


def beta_prior(belief: float, conf: str) -> tuple[float, float]:
    k = STRENGTH[conf]
    return belief * k, (1 - belief) * k


def r3(x):
    return float(round(float(x), 4))


# ------------------------------------------------------------------ 1. market model
def market_model(world: World, full) -> dict:
    asof = bt.ASOF
    snap = snapshot(full, asof)
    pid = world.product_tool_id
    C, T = len(world.companies), len(world.tools)
    pairs = np.array([(c, t) for (c, t) in snap.adopted if t != pid], np.int64)
    before = {}
    for c, t in pairs:
        before.setdefault(int(c), set()).add(int(t))
    new = {}
    for (c, t), (m, _) in full.adopted.items():
        if m > asof and t != pid and t not in before.get(c, set()):
            new.setdefault(c, set()).add(t)
    users = [c for c in new if c in before]

    def recall_at(score_fn, k=10):
        rs = []
        for c in users:
            s = score_fn(c).copy()
            s[list(before[c])] = -np.inf
            s[pid] = -np.inf
            top = set(np.argsort(-s)[:k].tolist())
            rs.append(len(top & new[c]) / len(new[c]))
        return float(np.mean(rs))

    pop = np.bincount(pairs[:, 1], minlength=T).astype(float)
    pop_recall = recall_at(lambda c: pop)

    gcurve = []
    def g_cb(ep, U, V):
        gcurve.append({"epoch": ep, "val_recall": r3(recall_at(lambda c: V @ U[c]))})
    t0 = time.time()
    gcn = LightGCN(C, T, dim=48, seed=0).fit(pairs, epochs=200, callback=g_cb, every=10)
    g_secs = time.time() - t0
    loss = gcn.loss_curve
    for row in gcurve:
        e = row["epoch"]
        row["loss"] = r3(np.mean(loss[max(0, e - 5):e]))

    # ALS regularization chosen on this validation set (reported, not hidden)
    sweep = {}
    for reg in (0.1, 1.0, 10.0, 50.0):
        m_ = ALS(C, T, dim=32, reg=reg, seed=0).fit(pairs, iters=8)
        sweep[reg] = recall_at(lambda c: m_.V @ m_.U[c])
    best_reg = max(sweep, key=sweep.get)
    acurve = []
    t0 = time.time()
    als = ALS(C, T, dim=32, reg=best_reg, seed=0).fit(pairs, iters=12,
                                                      callback=lambda it, U, V: acurve.append(
                                                          {"iter": it, "val_recall": r3(recall_at(lambda c: V @ U[c]))}))
    a_secs = time.time() - t0
    for row, l in zip(acurve, als.loss_curve):
        row["loss"] = r3(l)

    return {"badge": "synthetic", "caption": SYNTH,
            "task": f"Rank tools each company adopts after {month_label(asof)} (new adoptions only), Recall@10",
            "train_edges": int(len(pairs)), "companies": C, "tools": T, "eval_companies": len(users),
            "popularity_recall": r3(pop_recall),
            "lightgcn": {"curve": gcurve, "seconds": round(g_secs, 1), "layers": 3, "dim": 48, "epochs": 200,
                         "loss": "BPR"},
            "als": {"curve": acurve, "seconds": round(a_secs, 1), "dim": 32, "iters": 12, "reg": best_reg,
                    "reg_sweep": {str(k): r3(v) for k, v in sweep.items()},
                    "loss": "weighted squared (implicit ALS)"}}


# ------------------------------------------------------------ 2. prior + correction
SEGMENTS = list(FOUNDER_BELIEFS)
FEATURES = (["market score (LightGCN cold start)", "billing + CRM in stack", "RevOps hire, last 6 months",
             "finance hire, last 12 months", "pain match (text)"] + [f"size {x}" for x in SIZE_ORDER]
            + [f"segment {x}" for x in SEGMENTS])
HORIZON = 12


def features_at(world: World, full, T: int, model: MarketModel):
    """Features from data dated <= T only; label = adopts in (T, T + HORIZON]."""
    snap = snapshot(full, T)
    pid = world.product_tool_id
    early = sorted(snap.adopters(pid))
    y_hat, _ = model.market_scores(PRODUCT_DESCRIPTION, [c for c in early if c in model.graph.adopters(pid)])
    pain, _ = pain_match(model, model.text_vec(bt.FOUNDER_PAIN + " " + PRODUCT_DESCRIPTION))
    billing, crm = set(TOOLS["billing"][0]), set(TOOLS["crm"][0])
    rows, ids = [], []
    for c in world.companies:
        if c.id in set(early) or not snap.active(c.id, 0, T + 1):
            continue
        names = {world.tools[t].name for t in snap.company_tools(c.id)}
        w6 = snap.feature_window(c.id, T - 5, T + 1)
        w12 = snap.feature_window(c.id, T - 11, T + 1)
        rows.append([0.0, float(bool(names & billing) and bool(names & crm)), float("role:revops" in w6),
                     float(any(k in w12 for k in ("role:controller", "role:finops", "role:billing_specialist"))), 0.0]
                    + [float(c.size == x) for x in SIZE_ORDER] + [float(c.industry == x) for x in SEGMENTS])
        ids.append(c.id)
    ids = np.array(ids)
    X = np.array(rows)
    X[:, 0] = pct_rank(y_hat[ids])
    X[:, 4] = pct_rank(pain[ids])
    later = {c for c, m in full.adopters(pid).items() if T < m <= T + HORIZON}
    y = np.array([c in later for c in ids], float)
    return ids, X, y


def fit_logistic(X, y, offset, l2=1.0, iters=25):
    """Ridge logistic regression with a fixed offset, solved by Newton / IRLS."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    R = l2 * np.eye(Xb.shape[1])
    R[-1, -1] = 0  # don't shrink the intercept
    curve = []
    for i in range(iters):
        z = offset + Xb @ w
        p = 1 / (1 + np.exp(-z))
        curve.append({"iter": i, "loss": r3(np.mean(np.logaddexp(0, z) - y * z))})
        H = (Xb.T * (p * (1 - p))) @ Xb + R
        g = Xb.T @ (p - y) + R @ w
        w -= np.linalg.solve(H, g)
    return w[:-1], w[-1], curve


def prior_residual(world: World, full) -> dict:
    # rolling training months; every label window ends by the cut-off, so nothing leaks into the test
    train_months = [month_index(m) for m in ("2023-03", "2023-06", "2023-09", "2023-12")]
    t_test = bt.ASOF
    m_train = MarketModel(snapshot(full, month_index("2022-12")), epochs=120)  # one model, older than every train month
    m_test = MarketModel(snapshot(full, t_test), epochs=120)
    parts = [features_at(world, full, T, m_train) for T in train_months]
    ids_tr = np.concatenate([p[0] for p in parts])
    X_tr = np.vstack([p[1] for p in parts])
    y_tr = np.concatenate([p[2] for p in parts])
    ids_te, X_te, y_te = features_at(world, full, t_test, m_test)
    assert max(train_months) + HORIZON <= t_test, "train labels would overlap the test period"

    def prior_logit(ids):
        out = []
        for c in ids:
            a, b = beta_prior(*FOUNDER_BELIEFS[world.companies[c].industry])
            m = a / (a + b)
            out.append(np.log(m / (1 - m)))
        return np.array(out)

    off_tr, off_te = prior_logit(ids_tr), prior_logit(ids_te)
    w_c, b_c, _ = fit_logistic(X_tr, y_tr, np.zeros(len(y_tr)))
    w_r, b_r, curve_r = fit_logistic(X_tr, y_tr, off_tr)
    k = 50
    pos = y_te.astype(bool)
    variants = {
        "founder prior only": off_te,
        "correction only (no prior)": X_te @ w_c + b_c,
        "logit(prior) + correction": off_te + X_te @ w_r + b_r,
    }
    rows = {}
    for name, sc in variants.items():
        top = np.zeros(len(sc), bool)
        top[np.argsort(-sc, kind="stable")[:k]] = True
        hits = top[pos]
        lo, hi = bootstrap_recall_ci(hits, int(pos.sum()), k)
        rows[name] = {"recall": r3(hits.mean()), "ci95": [r3(lo), r3(hi)]}
    return {"badge": "synthetic", "caption": SYNTH,
            "formula": "score(c) = logit(prior_mean(segment(c))) + w·x(c) + b",
            "split": {"train_features_asof": ", ".join(month_label(m) for m in train_months),
                      "train_labels": f"adopted in the next {HORIZON} months (all before {month_label(t_test)})",
                      "test_features_asof": month_label(t_test),
                      "test_labels": f"adopted in the {HORIZON} months after the cut-off"},
            "train": {"n": int(len(y_tr)), "positives": int(y_tr.sum())},
            "test": {"n": int(len(y_te)), "positives": int(y_te.sum()), "k": k},
            "loss_curve": curve_r, "methods": rows,
            "weights": {f: r3(v) for f, v in zip(FEATURES, w_r)},
            "correction_model": "logistic on graph features (incl. the LightGCN score); a full GNN residual is the roadmap"}


# ----------------------------------------------------------------- 3. Thompson
def thompson(world: World, full) -> dict:
    asof = bt.ASOF
    pid = world.product_tool_id
    ad = full.adopters(pid)
    segs = list(FOUNDER_BELIEFS)
    truth = {}
    for s in segs:
        cand = [c.id for c in world.companies if c.industry == s and ad.get(c.id, 10**9) > asof]
        conv = [c for c in cand if c in ad]
        truth[s] = (len(conv) + 0.5) / (len(cand) + 1)
    p = np.array([truth[s] for s in segs])
    best = p.max()
    rounds, batch, runs = 40, 5, 300
    priors = np.array([beta_prior(*FOUNDER_BELIEFS[s]) for s in segs])
    # calibrated: keep the founder's ranking and confidence, rescale the level to the market base rate
    beliefs = np.array([FOUNDER_BELIEFS[s][0] for s in segs])
    base = float(np.mean(p))
    cal_mean = np.clip(beliefs / beliefs.mean() * base, 0.005, 0.95)
    k_ = np.array([STRENGTH[FOUNDER_BELIEFS[s][1]] for s in segs])
    cal = np.stack([cal_mean * k_, (1 - cal_mean) * k_], 1)

    def run(policy, rng):
        src = priors if policy in ("ts_founder", "greedy_founder") else cal if policy == "ts_calibrated" else None
        a = src[:, 0].copy() if src is not None else np.ones(len(segs))
        b = src[:, 1].copy() if src is not None else np.ones(len(segs))
        cum, regret, width = [], [], []
        total = reg = 0.0
        for _ in range(rounds):
            for _ in range(batch):
                if policy == "random":
                    s = rng.integers(len(segs))
                elif policy == "greedy_founder":
                    s = int(np.argmax(a / (a + b)))
                else:
                    s = int(np.argmax(rng.beta(a, b)))
                y = rng.random() < p[s]
                total += y
                reg += best - p[s]
                if policy != "greedy_founder":
                    a[s] += y
                    b[s] += 1 - y
            cum.append(total)
            regret.append(reg)
            # 90% interval width of the posterior for the truly best segment
            i = int(np.argmax(p))
            draws = rng.beta(a[i], b[i], 2000)
            width.append(float(np.quantile(draws, 0.95) - np.quantile(draws, 0.05)))
        return np.array(cum), np.array(regret), np.array(width), a / (a + b)

    pols = {"ts_calibrated": "Thompson, founder priors rescaled to base rate",
            "ts_founder": "Thompson, founder priors as stated", "ts_flat": "Thompson, flat priors",
            "greedy_founder": "Founder priors, never updated", "random": "Random segment"}
    out = {}
    rng = np.random.default_rng(SEED)
    for key, label in pols.items():
        C, R, Wd, M = zip(*[run(key, rng) for _ in range(runs)])
        out[key] = {"label": label, "cum_conversions": [r3(x) for x in np.mean(C, 0)],
                    "cum_regret": [r3(x) for x in np.mean(R, 0)], "best_width": [r3(x) for x in np.mean(Wd, 0)],
                    "final_posterior": {s: r3(v) for s, v in zip(segs, np.mean(M, 0))}}
    return {"badge": "synthetic", "caption": SYNTH,
            "setup": f"{rounds} rounds × {batch} contacts, {runs} runs averaged. A contact in segment s converts "
                     f"with the planted rate p_s.",
            "truth": {s: r3(v) for s, v in truth.items()},
            "founder_prior_mean": {s: r3(b[0]) for s, b in FOUNDER_BELIEFS.items()},
            "calibrated_prior_mean": {s: r3(v) for s, v in zip(segs, cal_mean)},
            "best_segment": segs[int(np.argmax(p))], "policies": out}


# ----------------------------------------------------------------- 4. planted
def planted(world: World, full) -> dict:
    cust = demo.pick_customers(full)
    card, rows = find_hit(full, cust, " ".join(demo.script(full)["answers"]))
    from api.engine import Engine, Session  # local: builds the live model once more
    eng = Engine(SEED)
    s = Session(eng)

    async def go():
        for a in eng.script["answers"]:
            await s.answer(a)
        return await s.run_reading()
    cards = {c.id: c for c in asyncio.run(go())}
    disc = cards["discovery"]
    seg_truth = {c.id for c in world.companies if c.industry in ("logistics", "healthtech") and c.latent[7] > 1.0}
    found = [x["id"] for x in disc.companies]
    in_seg = sum(c in seg_truth for c in found)
    return {"badge": "synthetic", "caption": SYNTH,
            "items": [
                {"planted": "Companies that post a RevOps role are much more likely to buy within 6 months",
                 "recovered": card.stats.get("feature") == "role:revops",
                 "evidence": f"hit finder top feature: {card.stats.get('feature')}, q = {card.stats.get('q_value', 1):.2e}, "
                             f"tested {len(rows)} features"},
                {"planted": "Ops-heavy logistics and healthtech companies start buying in 2025 and describe "
                            "the pain with different words",
                 "recovered": in_seg >= 0.6 * max(len(found), 1),
                 "evidence": f"{in_seg} of {len(found)} companies on the discovery card are in the planted segment; "
                             f"term found: \"{disc.stats.get('phrase')}\""},
                {"planted": "Hiring posts contain noisy LLM extractions (4% hallucinated tools)",
                 "recovered": len(full.rejected) > 0,
                 "evidence": f"Reviewer rejected {len(full.rejected)} unsupported triples before they reached the graph"},
            ]}


# ----------------------------------------------------------------- 5. voice (mock)
def voice_dpo() -> dict:
    rng = np.random.default_rng(SEED + 1)
    sft_steps = np.arange(0, 301, 10)
    sft = 2.15 * np.exp(-sft_steps / 90) + 1.32 + rng.normal(0, 0.03, len(sft_steps))
    dpo_steps = np.arange(0, 91, 3)
    dpo_loss = 0.693 - 0.29 * (1 - np.exp(-dpo_steps / 30)) + rng.normal(0, 0.012, len(dpo_steps))
    margin = 1.65 * (1 - np.exp(-dpo_steps / 28)) + rng.normal(0, 0.05, len(dpo_steps))
    n_judged = 120
    wins = 81
    phat = wins / n_judged
    se = np.sqrt(phat * (1 - phat) / n_judged)
    return {"badge": "simulated", "caption": SIMUL,
            "config": {"base": "8B open-weights instruct model", "adapter": "LoRA r=16, alpha=32, q/k/v/o",
                       "sft": "300 steps on 410 of the founder's sent emails, weighted toward ones that converted",
                       "dpo": "240 edit pairs (chosen = founder's edit, rejected = draft), beta 0.1, 3 epochs",
                       "hardware": "1 GPU pod, about 40 minutes (not run)"},
            "sft_curve": [{"step": int(s), "loss": r3(v)} for s, v in zip(sft_steps, sft)],
            "dpo_curve": [{"step": int(s), "loss": r3(l), "reward_margin": r3(m)}
                          for s, l, m in zip(dpo_steps, dpo_loss, margin)],
            "eval": {"win_rate_vs_base": r3(phat), "ci95": [r3(phat - 1.96 * se), r3(phat + 1.96 * se)],
                     "n_judged": n_judged, "ai_flags_per_email": {"base": 1.8, "tuned": 0.5}}}


# ----------------------------------------------------------------------- main
def main():
    t0 = time.time()
    world = generate(SEED)
    full = build_graph(world)
    print("market model…", flush=True)
    mm = market_model(world, full)
    print("prior + correction…", flush=True)
    pr = prior_residual(world, full)
    print("thompson…", flush=True)
    ts = thompson(world, full)
    print("backtest…", flush=True)
    back = bt.run(world, full, verbose=False)
    back |= {"badge": "synthetic", "caption": SYNTH}
    print("planted patterns…", flush=True)
    pl = planted(world, full)
    top_w = max(pr["weights"], key=lambda f: pr["weights"][f])
    pl["items"].insert(1, {
        "planted": "The same RevOps trigger, learned from outcomes instead of tested",
        "recovered": top_w == "RevOps hire, last 6 months",
        "evidence": f"largest correction weight: {top_w} ({pr['weights'][top_w]:+.2f}); "
                    f"segment logistics {pr['weights']['segment logistics']:+.2f}, "
                    f"saas-billing {pr['weights']['segment saas-billing']:+.2f} (correcting the founder's prior)"})
    vo = voice_dpo()
    ab = back["methods"]
    maturity = {"badge": "synthetic", "caption": SYNTH, "stages": [
        {"stage": "0 customers", "method": "founder priors + market model (cold start from the description)",
         "evidence": f"First 10 rounds: {ts['policies']['ts_calibrated']['cum_conversions'][9]:.1f} conversions with "
                     f"rescaled founder priors, {ts['policies']['ts_flat']['cum_conversions'][9]:.1f} with flat priors, "
                     f"{ts['policies']['ts_founder']['cum_conversions'][9]:.1f} with priors as stated",
         "badge": "synthetic"},
        {"stage": "tens of outcomes", "method": "Beta updates per rule and segment",
         "evidence": f"90% interval on the best segment narrows from "
                     f"{ts['policies']['ts_calibrated']['best_width'][0]:.2f} to "
                     f"{ts['policies']['ts_calibrated']['best_width'][-1]:.2f} over 40 rounds", "badge": "synthetic"},
        {"stage": "hundreds of outcomes", "method": "per-startup fine-tune: correction model + voice LoRA/DPO",
         "evidence": f"Correction model (synthetic) and voice DPO (simulated mock-up, win rate "
                     f"{vo['eval']['win_rate_vs_base']:.2f})", "badge": "simulated"},
    ]}
    out = {"seed": SEED, "built_seconds": round(time.time() - t0, 1),
           "badges": {"synthetic": SYNTH, "simulated": SIMUL,
                      "measured": "Measured: real public data, with n and confidence intervals."},
           "market_model": mm, "prior_residual": pr, "thompson": ts, "backtest": back,
           "ablation": {"badge": "synthetic", "caption": SYNTH, "k": back["k"], "positives": back["positives"],
                        "methods": {k: ab[k] for k in ("market model alone", "founder alone", "market + founder")}},
           "planted": pl, "voice_dpo": vo, "maturity": maturity,
           "measured": {"badge": "measured", "items": [],
                        "note": "No measured results yet. Planned: the same backtest on the public HN hiring corpus "
                                "vs an LLM ranking, scored only on months after the LLM's training cutoff, "
                                "with n and 95% CIs."}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT} in {out['built_seconds']}s")


if __name__ == "__main__":
    main()
