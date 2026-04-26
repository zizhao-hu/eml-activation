"""Causal self-attention with three normalization choices.

--attn = "softmax"     : standard softmax (baseline).
--attn = "eml-norm-v1" : single-projection EML normalization.
--attn = "eml-norm-v2" : twin-projection EML (strict generalization of softmax).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .operator import eml


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        block_size: int,
        attn: str = "softmax",
        dropout: float = 0.0,
        eml_pos: str = "softplus",
        eml_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if attn not in {"softmax", "eml-norm-v1", "eml-norm-v2"}:
            raise ValueError(f"unknown attn variant: {attn!r}")

        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.attn = attn
        self.eml_pos = eml_pos
        self.eml_eps = eml_eps

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        # v2: an extra K projection feeds the aggregate leg of EML
        self.k_y = nn.Linear(d_model, d_model, bias=False) if attn == "eml-norm-v2" else None
        self.proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        causal = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("causal_mask", causal.view(1, 1, block_size, block_size))

        self._init_weights()

    def _init_weights(self) -> None:
        # Smaller scale for projections feeding EML legs (keeps exp(L) ~ 1 at step 0).
        scale = 0.1 if self.attn != "softmax" else 1.0
        for lin in [self.qkv, self.proj]:
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
            with torch.no_grad():
                lin.weight.mul_(scale)
        if self.k_y is not None:
            nn.init.kaiming_uniform_(self.k_y.weight, a=math.sqrt(5))
            with torch.no_grad():
                self.k_y.weight.mul_(0.1)

    def _split_heads(self, t: Tensor) -> Tensor:
        B, T, _ = t.shape
        return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scale = 1.0 / math.sqrt(self.head_dim)
        L = (q @ k.transpose(-2, -1)) * scale  # (B, H, T, T)

        mask = self.causal_mask[:, :, :T, :T]  # 1 where allowed, 0 where masked

        if self.attn == "softmax":
            L = L.masked_fill(mask == 0, float("-inf"))
            A = F.softmax(L, dim=-1)
        elif self.attn == "eml-norm-v1":
            # Aggregate over allowed keys via softplus (always positive).
            sp = F.softplus(L) * mask
            agg = sp.sum(dim=-1, keepdim=True) + self.eml_eps  # (B,H,T,1), already > 0
            agg = agg.expand_as(L)
            # agg is already positive; "abs" pos mode is a no-op + eps.
            r = eml(L, agg, pos="abs", eps=self.eml_eps)
            w = F.softplus(r) * mask
            A = w / (w.sum(dim=-1, keepdim=True) + self.eml_eps)
        else:  # eml-norm-v2
            k_y = self._split_heads(self.k_y(x))
            L_y = (q @ k_y.transpose(-2, -1)) * scale
            # Aggregate over allowed keys via exp(L_y) clamped for safety.
            L_y_clamped = L_y.clamp(-15.0, 15.0)
            agg = (torch.exp(L_y_clamped) * mask).sum(dim=-1, keepdim=True) + self.eml_eps
            agg = agg.expand_as(L)
            r = eml(L, agg, pos="abs", eps=self.eml_eps)
            w = F.softplus(r) * mask
            A = w / (w.sum(dim=-1, keepdim=True) + self.eml_eps)

        A = self.attn_dropout(A)
        out = A @ v  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.resid_dropout(self.proj(out))
