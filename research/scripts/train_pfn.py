"""Train PC-RFM v0 on freshly sampled simulated tasks (no fixed dataset; every batch is new).

Loss: cross-entropy against the true buying probability p on uncontacted accounts
(same optimum as sampled labels, lower variance). We also log the excess loss
KL(p || q), which is 0 only for a model that knows p exactly.

  python scripts/train_pfn.py --steps 20000 --out runs/pfn_v0          # with founder
  python scripts/train_pfn.py --steps 20000 --out runs/pfn_v0_nf --no-founder   # ablation
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fprior.pfn.data import PriorConfig, sample_batch  # noqa: E402
from fprior.pfn.model import PCRFM  # noqa: E402


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def excess_loss(logits, p, mask):
    q = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    pc = p.clamp(1e-6, 1 - 1e-6)
    kl = pc * (pc / q).log() + (1 - pc) * ((1 - pc) / (1 - q)).log()
    return (kl * mask).sum() / mask.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n-accounts", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--no-founder", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--out", default="runs/pfn_v0")
    a = ap.parse_args()

    dev = pick_device(a.device)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prior = PriorConfig(n_accounts=a.n_accounts, p_no_founder=1.0 if a.no_founder else 0.0)
    arch = dict(d=a.d, layers=a.layers, heads=4, ff=2 * a.d)
    model = PCRFM(**arch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    warm = min(1000, a.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    g = torch.Generator(device=dev).manual_seed(a.seed + 1)
    g_val = torch.Generator(device=dev).manual_seed(10_000 + a.seed)
    val = [sample_batch(prior, a.batch, g_val, dev) for _ in range(8)]
    meta = dict(args=vars(a), prior=asdict(prior), arch=arch, device=dev,
                n_params=sum(p.numel() for p in model.parameters()))
    (out / "config.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
    log = open(out / "log.jsonl", "a")
    t0, run = time.time(), 0.0

    for step in range(1, a.steps + 1):
        model.train()
        b = sample_batch(prior, a.batch, g, dev)
        logits = model(b.x, b.context)
        mask = (~b.context).float()
        loss = (F.binary_cross_entropy_with_logits(logits, b.target_p, reduction="none") * mask).sum() / mask.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % a.log_every == 0 or step == a.steps:
            model.eval()
            with torch.no_grad():
                kl = [excess_loss(model(v.x, v.context), v.target_p, (~v.context).float()).item() for v in val]
            rec = dict(step=step, loss=run, val_excess=sum(kl) / len(kl), lr=sched.get_last_lr()[0],
                       sec=round(time.time() - t0, 1))
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            torch.save({"state": model.state_dict(), "arch": arch, "prior": asdict(prior), "step": step},
                       out / "model.pt")
    print(f"done in {time.time() - t0:.0f}s -> {out / 'model.pt'}")


if __name__ == "__main__":
    main()
