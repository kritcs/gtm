import sys, torch, numpy as np; sys.path.insert(0, ".")
from fprior.pfn.data import PriorConfig, sample_graph, N_IND
torch.set_num_threads(2)
g = torch.Generator().manual_seed(3)
for gam in [(1, 10), (10, 10), (100, 100), (1000, 1000)]:
    cfg = PriorConfig(graph=True, smoothness_range=gam)
    ind = torch.randint(0, N_IND, (8, 1024), generator=g)
    A, r = sample_graph(cfg, ind, g, "cpu")
    deg = A.sum(-1).clamp_min(1)
    nb = torch.einsum("bij,bj->bi", A, r) / deg
    c = [np.corrcoef(r[i].numpy(), nb[i].numpy())[0, 1] for i in range(8)]
    print(f"gamma {gam}: corr(r, mean neighbour r) = {np.mean(c):.3f}; mean degree {A.sum(-1).mean():.1f}")
