"""TinyGPT — small char-level transformer used as the EML-attention test bed."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention import CausalSelfAttention


@dataclass
class GPTConfig:
    vocab_size: int = 256          # byte-level
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    dropout: float = 0.0
    attn: str = "softmax"          # softmax | eml-norm-v1 | eml-norm-v2


class FFN(nn.Module):
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, mult * d_model)
        self.fc2 = nn.Linear(mult * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(
            d_model=cfg.d_model,
            n_head=cfg.n_head,
            block_size=cfg.block_size,
            attn=cfg.attn,
            dropout=cfg.dropout,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FFN(cfg.d_model, mult=4, dropout=cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Tie weights.
        self.head.weight = self.tok_emb.weight
        self._init_global()

    def _init_global(self) -> None:
        # nanoGPT-style small init for embeddings and FFN linears.
        for emb in (self.tok_emb, self.pos_emb):
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        for block in self.blocks:
            for lin in (block.ffn.fc1, block.ffn.fc2):
                nn.init.normal_(lin.weight, mean=0.0, std=0.02)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)

    def forward(self, idx: Tensor, targets: Tensor | None = None):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    def num_params(self) -> int:
        # Subtract tied head params (counted via tok_emb already).
        n = sum(p.numel() for p in self.parameters())
        return n - self.head.weight.numel()
