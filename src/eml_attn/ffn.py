"""FFN variants: ReLU MLP, GELU MLP, SwiGLU, EML-GLU.

All variants share the (h: [B, T, d_model]) -> [B, T, d_model] interface.
GLU variants halve d_ff to keep total parameter count comparable to ungated baselines.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops.eml import eml


class ReLUMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, bias: bool = True):
        super().__init__()
        self.fc_in = nn.Linear(d_model, d_ff, bias=bias)
        self.fc_out = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc_out(F.relu(self.fc_in(h)))


class GELUMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, bias: bool = True):
        super().__init__()
        self.fc_in = nn.Linear(d_model, d_ff, bias=bias)
        self.fc_out = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc_out(F.gelu(self.fc_in(h)))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, bias: bool = True):
        super().__init__()
        self.fc_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.fc_up = nn.Linear(d_model, d_ff, bias=bias)
        self.fc_out = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc_out(F.silu(self.fc_gate(h)) * self.fc_up(h))


class EMLGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, bias: bool = True):
        super().__init__()
        self.fc_x = nn.Linear(d_model, d_ff, bias=bias)
        self.fc_y = nn.Linear(d_model, d_ff, bias=bias)
        self.fc_out = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc_out(eml(self.fc_x(h), self.fc_y(h)))


def build_ffn(kind: str, d_model: int, d_ff_mult: int = 4, bias: bool = True) -> nn.Module:
    """Build FFN by name. d_ff is halved for GLU variants to match param count."""
    kind = kind.lower()
    if kind in ("relu", "gelu"):
        d_ff = d_ff_mult * d_model
        cls = {"relu": ReLUMLP, "gelu": GELUMLP}[kind]
        return cls(d_model, d_ff, bias=bias)
    elif kind in ("swiglu", "emlglu"):
        # GLU has 3 linear layers (vs 2 for ungated). To match params,
        # use d_ff = (d_ff_mult * d_model) * 2/3, rounded to multiple of 8.
        target = int(d_ff_mult * d_model * 2 / 3)
        d_ff = ((target + 7) // 8) * 8
        cls = {"swiglu": SwiGLU, "emlglu": EMLGLU}[kind]
        return cls(d_model, d_ff, bias=bias)
    else:
        raise ValueError(f"unknown ffn kind: {kind!r}")
