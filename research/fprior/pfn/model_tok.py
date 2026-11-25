"""PC-RFM with founder-statement tokens (PLAN.md §3 "founder-belief tokens").

Sequence = [founder tokens (one per segment statement); account tokens]. Attention keys allowed:
  - every founder token (if the founder exists),
  - contacted (labelled) accounts,
  - the token itself.
Accounts carry no belief value; they must find the relevant statement by attention (industry / size
features on both token types). Founder tokens also see the labelled accounts, so a statement's
representation can be revised by the evidence about its segment.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .data import N_FEAT, N_FTOK
from .model import Block


class PCRFMTok(nn.Module):
    def __init__(self, d: int = 128, layers: int = 6, heads: int = 4, ff: int = 256,
                 n_feat: int = N_FEAT, n_ftok: int = N_FTOK, tokens: bool = True):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_feat, d), nn.GELU(), nn.Linear(d, d))
        self.fembed = nn.Sequential(nn.Linear(n_ftok, d), nn.GELU(), nn.Linear(d, d))
        self.type_emb = nn.Parameter(torch.zeros(2, d))
        self.blocks = nn.ModuleList([Block(d, heads, ff) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x, context, f_tok, f_mask):
        B, N, _ = x.shape
        K = f_tok.shape[1]
        h = torch.cat([self.fembed(f_tok) + self.type_emb[0], self.embed(x) + self.type_emb[1]], 1)
        key_ok = torch.cat([f_mask, context], 1)                           # (B, K + N)
        eye = torch.eye(K + N, dtype=torch.bool, device=x.device)
        mask = key_ok[:, None, :] | eye[None]
        for blk in self.blocks:
            h = blk(h, mask)
        return self.out(h[:, K:]).squeeze(-1)

    @torch.no_grad()
    def predict(self, x, context, f_tok, f_mask, chunk: int = 1024):
        """Score all accounts of one market (B = 1), query accounts in chunks (they don't see each other)."""
        assert x.shape[0] == 1
        ctx_idx = context[0].nonzero().squeeze(-1)
        out = torch.empty(x.shape[1], device=x.device)
        for s in range(0, x.shape[1], chunk):
            q = torch.arange(s, min(s + chunk, x.shape[1]), device=x.device)
            q = q[~context[0, q]]
            idx = torch.cat([ctx_idx, q])
            c = torch.zeros(1, len(idx), dtype=torch.bool, device=x.device)
            c[0, :len(ctx_idx)] = True
            out[q] = self(x[:, idx], c, f_tok, f_mask)[0, len(ctx_idx):]
        if len(ctx_idx):
            c = torch.ones(1, len(ctx_idx), dtype=torch.bool, device=x.device)
            out[ctx_idx] = self(x[:, ctx_idx], c, f_tok, f_mask)[0]
        return out
