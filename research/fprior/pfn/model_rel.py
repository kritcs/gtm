"""PC-RFM v2: learned relational encoder (RDL direction, PLAN.md section 3).

Two token types — accounts and products — linked by the adoption table. Each block:
  1. account <- product : mean of the products the account uses
  2. product <- account : mean of the product's adopters (carries label states of known
                          customers / non-buyers, so evidence flows account -> product -> account)
  3. account attention to the labelled (context) accounts and itself, as in v0/v1.
Unlike v0/v1 this is transductive: an account's score can depend on other unlabelled
accounts through shared products, so inference runs on the whole market at once.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model import Block, PCRFM

N_PFEAT = 3


def product_features(adoption: torch.Tensor, sim: torch.Tensor) -> torch.Tensor:
    """adoption (B, N, P), sim (B, P) -> (B, P, 3): z(sim), raw sim, z(log adopters)."""
    def z(x):
        return (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp_min(1e-6)
    ldeg = torch.log1p(adoption.sum(1))
    return torch.stack([z(sim), sim, z(ldeg)], -1)


class Bipartite(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.na, self.np = nn.LayerNorm(d), nn.LayerNorm(d)
        self.p2a = nn.Linear(d, d)
        self.a2p = nn.Linear(d, d)
        self.ffp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, h, hp, A_row, A_col):
        h = h + torch.bmm(A_row, self.p2a(self.np(hp)))          # account <- products
        hp = hp + torch.bmm(A_col, self.a2p(self.na(h)))          # product <- adopters
        return h, hp + self.ffp(hp)


class PCRFMRel(PCRFM):
    def __init__(self, d: int = 128, layers: int = 6, heads: int = 4, ff: int = 256,
                 n_feat: int = 24, encoder: bool = True):
        super().__init__(d, layers, heads, ff, n_feat)
        self.pembed = nn.Sequential(nn.Linear(N_PFEAT, d), nn.GELU(), nn.Linear(d, d))
        self.bip = nn.ModuleList([Bipartite(d) for _ in range(layers)])

    def forward(self, x, context, adoption, pfeat):
        deg_a = adoption.sum(2, keepdim=True).clamp_min(1)
        deg_p = adoption.sum(1, keepdim=True).clamp_min(1)
        A_row = adoption / deg_a                                   # (B, N, P)
        A_col = (adoption / deg_p).transpose(1, 2)                 # (B, P, N)
        h, hp = self.embed(x), self.pembed(pfeat)
        mask = self.attention_mask(context)
        for bip, blk in zip(self.bip, self.blocks):
            h, hp = bip(h, hp, A_row, A_col)
            h = blk(h, mask)
        return self.out(h).squeeze(-1)

    @torch.no_grad()
    def predict(self, x, context, adoption, pfeat):  # noqa: D401 - full-market forward
        return self(x, context, adoption, pfeat)[0]


__all__ = ["PCRFMRel", "product_features", "Block"]
