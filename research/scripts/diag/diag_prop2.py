"""Exploratory (prior only): signal of prop_features vs one-hop share under the proposed v0g2 prior, by regime."""
import sys, torch, numpy as np; sys.path.insert(0, ".")
from fprior.pfn.data import PriorConfig, sample_batch, N_GFEAT, N_PFEAT
torch.set_num_threads(2)
prior = PriorConfig(graph=True, graph_prop=True, random_seg_structure=True, smoothness_log=True,
                    smoothness_range=(1.0, 100.0), w_graph_range=(0.0, 0.9))
g = torch.Generator().manual_seed(5)
R = []
for _ in range(8):
    b = sample_batch(prior, 16, g, "cpu")
    lp = torch.logit(b.target_p.clamp(1e-4, 1 - 1e-4))
    one, s5, s9 = b.x[..., -N_PFEAT - 1], b.x[..., -4], b.x[..., -2]
    for i in range(16):
        un = ~b.context[i]
        gshare = b.info["w_graph"][i].item() * (1 - b.info["w_seg"][i].item())
        R.append([gshare] + [np.corrcoef(f[i, un].numpy(), lp[i, un].numpy())[0, 1] for f in (one, s5, s9)])
R = np.array(R)
for lo, hi in [(0, .2), (.2, .5), (.5, 1)]:
    s = (R[:, 0] >= lo) & (R[:, 0] < hi)
    print(f"graph share of logit var [{lo},{hi}) n={s.sum():3d}: corr 1-hop {np.nanmean(R[s,1]):+.3f}  PPR.5 {np.nanmean(R[s,2]):+.3f}  PPR.9 {np.nanmean(R[s,3]):+.3f}")
