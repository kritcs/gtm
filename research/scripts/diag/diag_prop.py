"""Exploratory (prior only): signal in multi-hop propagated label features vs the one-hop share.
f_alpha = PPR_alpha(visible centred labels) / (PPR_alpha(visible mask) + eps), PPR by 30 power-iteration steps on
the symmetric-normalised adjacency. Reports corr with the true logit among unlabelled accounts, and the partial
corr after removing the true segment effect (what the graph adds beyond segments)."""
import sys, torch, numpy as np; sys.path.insert(0, ".")
from fprior.pfn.data import PriorConfig, sample_batch, N_GFEAT
import fprior.pfn.data as D
torch.set_num_threads(2)
cap = {}
orig = D.sample_graph
def spy(*a, **k):
    A, r = orig(*a, **k); cap["A"], cap["r"] = A, r; return A, r
D.sample_graph = spy
def ppr(S, v, a, K=30):
    out = (1 - a) * v
    z = out.clone()
    for _ in range(K):
        z = a * torch.einsum("bij,bj->bi", S, z); out = out + z  # sum_k (1-a) a^k S^k v
        z = z  # noqa
    return out
prior = PriorConfig(graph=True, random_seg_structure=True)
g = torch.Generator().manual_seed(11)
keys = ["1hop", 0.5, 0.8, 0.9, 0.95]
res = {k: [] for k in keys}; resr = {k: [] for k in keys}; hi = {k: [] for k in keys}
for _ in range(8):
    b = sample_batch(prior, 16, g, "cpu")
    A, r = cap["A"], cap["r"]; d = A.sum(-1).clamp_min(1); S = A / d.sqrt()[:, :, None] / d.sqrt()[:, None, :]
    ctx = b.context.float(); yc = (b.y.float() - (b.y.float() * ctx).sum(1, keepdim=True) / ctx.sum(1, keepdim=True)) * ctx
    lp = torch.logit(b.target_p.clamp(1e-4, 1 - 1e-4))
    feats = {"1hop": b.x[..., -1]}
    for a in keys[1:]:
        feats[a] = ppr(S, yc, a) / (ppr(S, ctx, a) + 1e-3)
    for i in range(b.x.shape[0]):
        un = ~b.context[i]
        for k in keys:
            f = feats[k][i, un].numpy()
            c = np.corrcoef(f, lp[i, un].numpy())[0, 1]; cr = np.corrcoef(f, r[i, un].numpy())[0, 1]
            res[k].append(c); resr[k].append(cr)
            if b.info["w_graph"][i] > .5: hi[k].append(c)
for k in keys:
    print(f"{str(k):>5}: corr(f, true logit) {np.nanmean(res[k]):+.3f} | w_graph>0.5 {np.nanmean(hi[k]):+.3f} | corr(f, field r) {np.nanmean(resr[k]):+.3f}")
