"""Exploratory: does v0g use its graph features on its own training prior? Shuffle the 4 graph-feature
columns across accounts within each task (equivalent to a node-permuted graph) and compare excess loss."""
import sys, torch; sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from dataclasses import fields
from fprior.pfn.data import PriorConfig, sample_batch, N_GFEAT
from fprior.pfn.model import PCRFM
from train_pfn import excess_loss
torch.set_num_threads(2)
ck = torch.load("runs/local_v0g/model.pt", map_location="cpu")
m = PCRFM(**ck["arch"]); m.load_state_dict(ck["state"]); m.eval()
pr = {k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["prior"].items() if k in {f.name for f in fields(PriorConfig)}}
prior = PriorConfig(**pr)
g = torch.Generator().manual_seed(4242)
rows = []
with torch.no_grad():
    for _ in range(16):
        b = sample_batch(prior, 16, g, "cpu")
        mask = (~b.context).float()
        x2 = b.x.clone()
        perm = torch.argsort(torch.rand(b.x.shape[:2], generator=g), 1)
        x2[..., -N_GFEAT:] = torch.gather(b.x[..., -N_GFEAT:], 1, perm[..., None].expand(-1, -1, N_GFEAT))
        x0 = b.x.clone(); x0[..., -N_GFEAT:] = 0
        for i in range(b.x.shape[0]):
            sl = slice(i, i + 1)
            e = [excess_loss(m(x[sl], b.context[sl]), b.target_p[sl], mask[sl]).item() for x in (b.x, x2, x0)]
            rows.append((b.info["w_graph"][i].item(), int(b.context[i].sum()), *e))
import numpy as np
R = np.array(rows)
print(f"tasks {len(R)}; mean excess  true {R[:,2].mean():.4f}  permuted {R[:,3].mean():.4f}  zeroed {R[:,4].mean():.4f}")
for lo, hi in [(0, .2), (.2, .5), (.5, .8)]:
    s = (R[:,0] >= lo) & (R[:,0] < hi)
    d = R[s,3] - R[s,2]
    print(f"w_graph [{lo},{hi}): n={s.sum():3d} permuted-true {d.mean():+.4f} ± {d.std()/np.sqrt(s.sum()):.4f}")
for lo, hi in [(0, 6), (6, 21), (21, 10**6)]:
    s = (R[:,1] >= lo) & (R[:,1] < hi)
    d = R[s,3] - R[s,2]
    print(f"labels [{lo},{hi}): n={s.sum():3d} permuted-true {d.mean():+.4f} ± {d.std()/np.sqrt(s.sum()):.4f}")
