"""Expressivity sanity checks for EML.

Mode A — fixed closed-form compositions (no learned scalars). Verifies that the
Odrzywolek constructions match the corresponding standard functions to floating
point tolerance under our PyTorch eml() implementation.

Mode B — small parameterized binary tree of EML nodes (each leg fed by a
learned linear map). Used to check that EML cells are *trainable* targets like
sigmoid and softmax via plain Adam.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .operator import eml


# ---------------------------------------------------------------------------
# Mode A — closed-form compositions
# ---------------------------------------------------------------------------

def closed_form_exp(x: Tensor) -> Tensor:
    """exp(x) = eml(x, 1)."""
    one = torch.ones_like(x)
    return eml(x, one, pos="abs", eps=1e-12)


def closed_form_ln(x: Tensor) -> Tensor:
    """ln(x) = eml(1, eml(eml(1, x), 1)). Requires x > 0."""
    one = torch.ones_like(x)
    inner = eml(one, x, pos="abs", eps=1e-12)              # e - ln(x)
    middle = eml(inner, one, pos="abs", eps=1e-12)         # exp(e - ln(x))
    return eml(one, middle, pos="abs", eps=1e-12)          # e - ln(middle) = ln(x)


def closed_form_softmax(L: Tensor, dim: int = -1) -> Tensor:
    """Softmax via EML compositions.

    softmax_i(L) = exp(L_i) / Σ_j exp(L_j)
                 = exp(L_i) · exp(−ln(Σ_j exp(L_j)))
                 = closed_form_exp(L_i − closed_form_ln(Σ_j closed_form_exp(L_j))).
    """
    e = closed_form_exp(L)
    Z = e.sum(dim=dim, keepdim=True)
    log_Z = closed_form_ln(Z)
    return closed_form_exp(L - log_Z)


# ---------------------------------------------------------------------------
# Mode B — parameterized EML cells
# ---------------------------------------------------------------------------

class EMLCell(nn.Module):
    """One eml node with two learned linear legs."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.x_proj = nn.Linear(dim_in, dim_out)
        self.y_proj = nn.Linear(dim_in, dim_out)
        for lin in (self.x_proj, self.y_proj):
            with torch.no_grad():
                lin.weight.mul_(0.1)
                lin.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return eml(self.x_proj(x), self.y_proj(x), pos="softplus", eps=1e-6)


class EMLTree(nn.Module):
    """Stacked EML cells (depth = number of layers)."""

    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, depth: int = 2) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers: list[nn.Module] = []
        d_prev = dim_in
        for _ in range(depth - 1):
            layers.append(EMLCell(d_prev, dim_hidden))
            d_prev = dim_hidden
        layers.append(EMLCell(d_prev, dim_out))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def fit_target(
    target: Callable[[Tensor], Tensor],
    model: nn.Module,
    sampler: Callable[[int], Tensor],
    *,
    steps: int = 2000,
    batch: int = 256,
    lr: float = 3e-3,
    loss_fn: str = "mse",
) -> dict:
    """Adam-fit a parameterized model to a target function.

    Returns a dict with the loss curve and the final value.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
    for _ in range(steps):
        x = sampler(batch)
        y_true = target(x).detach()
        y_pred = model(x)
        if loss_fn == "mse":
            loss = F.mse_loss(y_pred, y_true)
        elif loss_fn == "kl":
            log_pred = torch.log(F.softmax(y_pred, dim=-1) + 1e-12)
            loss = F.kl_div(log_pred, y_true, reduction="batchmean")
        else:
            raise ValueError(f"unknown loss_fn: {loss_fn}")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        history.append(float(loss.detach().cpu()))
    return {"final": history[-1], "history": history}
