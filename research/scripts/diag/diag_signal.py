"""Exploratory: how much label signal reaches the 1-hop neighbour features in the v0g prior?"""
import sys, torch, numpy as np; sys.path.insert(0, ".")
from fprior.pfn.data import PriorConfig, sample_batch, N_GFEAT
torch.set_num_threads(2)
prior = PriorConfig(graph=True, random_seg_structure=True)
g = torch.Generator().manual_seed(7)
cov, cor, cor_hi = [], [], []
for _ in range(8):
    b = sample_batch(prior, 16, g, "cpu")
    gf = b.x[..., -N_GFEAT:]
    lp = torch.logit(b.target_p.clamp(1e-4, 1 - 1e-4))
    for i in range(b.x.shape[0]):
        un = ~b.context[i]
        has = (gf[i, :, 1] + gf[i, :, 2] > 0) & un
        cov.append(has.float().sum().item() / un.sum().item())
        if has.sum() > 10:
            c = np.corrcoef(gf[i, has, 3].numpy(), lp[i, has].numpy())[0, 1]
            cor.append(c)
            if b.info["w_graph"][i] > .5: cor_hi.append(c)
print(f"share of unlabelled accounts with >=1 labelled neighbour: {np.mean(cov):.3f} (median {np.median(cov):.3f})")
print(f"corr(neighbour buy share, true logit) among them: mean {np.nanmean(cor):+.3f}; w_graph>0.5: {np.nanmean(cor_hi):+.3f} (n={len(cor_hi)})")
print("labels per task: median", int(b.context.sum(1).median()))
