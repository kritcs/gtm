"""PC-RFM v0 network: a TabPFN-style transformer over account tokens.

Every account attends to the contacted (labelled) accounts and to itself, never to
other unlabelled accounts. So each prediction depends only on its own features plus
the context, which makes the model independent of market size and lets inference
run in chunks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import N_FEAT


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, dropout: float = 0.0):
        super().__init__()
        self.h = heads
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.drop = dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None],
                                           dropout_p=self.drop if self.training else 0.0)
        x = x + self.proj(a.transpose(1, 2).reshape(B, N, d))
        return x + self.ff(self.n2(x))


class PCRFM(nn.Module):
    def __init__(self, d: int = 128, layers: int = 6, heads: int = 4, ff: int = 256):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(N_FEAT, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, heads, ff) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    @staticmethod
    def attention_mask(context: torch.Tensor) -> torch.Tensor:
        """(B, N) bool -> (B, N, N) bool, True = may attend: key is context, or key is self."""
        N = context.shape[1]
        eye = torch.eye(N, dtype=torch.bool, device=context.device)
        return context[:, None, :] | eye[None]

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        mask = self.attention_mask(context)
        for blk in self.blocks:
            h = blk(h, mask)
        return self.out(h).squeeze(-1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, context: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
        """Score every account of one market (B = 1) by appending query chunks to the context."""
        assert x.shape[0] == 1
        ctx_idx = context[0].nonzero().squeeze(-1)
        out = torch.empty(x.shape[1], device=x.device)
        for s in range(0, x.shape[1], chunk):
            q = torch.arange(s, min(s + chunk, x.shape[1]), device=x.device)
            q = q[~context[0, q]]
            idx = torch.cat([ctx_idx, q])
            c = torch.zeros(1, len(idx), dtype=torch.bool, device=x.device)
            c[0, :len(ctx_idx)] = True
            out[q] = self(x[:, idx], c)[0, len(ctx_idx):]
        if len(ctx_idx):
            c = torch.ones(1, len(ctx_idx), dtype=torch.bool, device=x.device)
            out[ctx_idx] = self(x[:, ctx_idx], c)[0]
        return out
