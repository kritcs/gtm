"""Fine-tune a simulator-trained PC-RFM on real-outcome tasks (PLAN.md §5: "a fine-tuning stage on real-outcome
tasks"). Pre-registration: results/finetune_preregistration.md.

Data hygiene. The evaluation scripts (real_bank.py, bank_noise_control.py) split each pool once into
A (founder experience) and B (market pool) with split_pool(seed 12345). Fine-tuning never touches B:
A is split again (seed 777) into A1 — founder beliefs for fine-tuning tasks — and A2 — the accounts and real
outcomes of fine-tuning markets. So the outcomes a fine-tuned model is evaluated on were never trained on.
Category indices are relabelled per market exactly as at evaluation (make_bank_market).

Each step mixes a simulated batch (target = true p, as in pre-training; guards against forgetting) and a real
batch (target = realized outcome, founder simulated with rho ~ U(-0.5, 1), n ~ the pre-training distribution,
founder-steered outreach as in fprior.sim.run_outreach). Loss: BCE on uncontacted accounts.

  python scripts/finetune_pfn.py --init runs/pfn_v01/model.pt --dataset bank --out runs/pfn_v01_ft_bank
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_pfn import load  # noqa: E402
from fprior.pfn.data import PriorConfig, make_tokens, sample_batch  # noqa: E402
from fprior.real_bank import load_pool, make_bank_market, split_pool  # noqa: E402
from fprior.sim import FounderConfig, OutreachConfig, make_founder, run_outreach  # noqa: E402


def real_batch(pool, A1, A2, B, N, rng, use_founder, dev):
    xs, ctxs, ys = [], [], []
    for _ in range(B):
        m = make_bank_market(pool, A1, A2, N, rng)
        rho = rng.uniform(-0.5, 1.0)
        n = int(min(50, math.floor(math.exp(rng.uniform(0, math.log(52))) - 1)))
        f = make_founder(m, FounderConfig(reliability=rho), rng)
        out = run_outreach(m, f, OutreachConfig(n_customers=n), rng)
        ctx = np.zeros(m.n, dtype=bool)
        ctx[out.contacted] = True
        y = np.zeros(m.n)
        y[out.contacted] = out.outcome
        t = lambda a, dt=torch.long: torch.as_tensor(a, dtype=dt, device=dev)[None]  # noqa: E731
        xs.append(make_tokens(t(m.industry), t(m.size), t(f.segment_logit[m.segment], torch.float32),
                              torch.tensor([1.0 if use_founder else 0.0], device=dev),
                              t(ctx, torch.bool), t(y, torch.float32)))
        ctxs.append(t(ctx, torch.bool))
        ys.append(t(m.y, torch.float32))
    return torch.cat(xs), torch.cat(ctxs), torch.cat(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--dataset", required=True, choices=["bank", "hillstrom"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-accounts", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(a.seed)
    ck = torch.load(a.init, map_location=dev)
    model = load(a.init, dev)
    assert not ck["arch"].get("tokens") and not getattr(model, "graph", False), "plain PC-RFM only"
    prior = PriorConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["prior"].items()})
    use_founder = prior.p_no_founder < 1.0
    pool = load_pool(a.dataset)
    A, _B = split_pool(pool)                       # B: evaluation pool, never used here
    perm = np.random.default_rng(777).permutation(A)
    A1, A2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
    rng = np.random.default_rng(a.seed)
    g = torch.Generator(device=dev).manual_seed(a.seed + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(args=vars(a), init_step=ck.get("step"), n_A1=len(A1), n_A2=len(A2))
    (out / "config.json").write_text(json.dumps(meta, indent=1))
    log = open(out / "log.jsonl", "w")
    t0, run_s, run_r = time.time(), None, None
    for step in range(1, a.steps + 1):
        model.train()
        sb = sample_batch(prior, a.batch, g, dev)
        ls = (F.binary_cross_entropy_with_logits(model(sb.x, sb.context), sb.target_p, reduction="none")
              * (~sb.context)).sum() / (~sb.context).sum()
        x, ctx, y = real_batch(pool, A1, A2, a.batch, a.n_accounts, rng, use_founder, dev)
        lr_ = (F.binary_cross_entropy_with_logits(model(x, ctx), y, reduction="none") * (~ctx)).sum() / (~ctx).sum()
        loss = 0.5 * (ls + lr_)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run_s = ls.item() if run_s is None else 0.98 * run_s + 0.02 * ls.item()
        run_r = lr_.item() if run_r is None else 0.98 * run_r + 0.02 * lr_.item()
        if step % 100 == 0 or step == a.steps:
            rec = dict(step=step, sim_loss=run_s, real_loss=run_r, sec=round(time.time() - t0, 1))
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            torch.save({"state": model.state_dict(), "arch": ck["arch"], "prior": ck["prior"], "step": step,
                        "finetune": meta}, out / "model.pt")


if __name__ == "__main__":
    main()
