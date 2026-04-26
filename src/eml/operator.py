"""Exp-Minus-Log operator: eml(x, y) = exp(x) - ln(y).

Odrzywolek (arXiv 2603.21852, 2026) showed that {eml, 1} expresses every
elementary function. Here we provide a numerically careful PyTorch
implementation suitable for use as a building block inside neural networks
(in particular, as the binary primitive in attention's normalization slot).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

_FP32_CLAMP = 20.0
_LOWPREC_CLAMP = 15.0


def _positive(y: Tensor, mode: str, eps: float) -> Tensor:
    """Map y to a strictly positive tensor suitable for the ln() leg."""
    if mode == "softplus":
        return F.softplus(y) + eps
    if mode == "abs":
        return y.abs() + eps
    if mode == "exp_reparam":
        # treat y itself as log-domain: ln(exp(y)) = y; clamp for safety
        return y
    raise ValueError(f"unknown positivity mode: {mode!r}")


def eml(
    x: Tensor,
    y: Tensor,
    *,
    pos: str = "softplus",
    clamp: float | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """Compute eml(x, y) = exp(x) - ln(y) elementwise.

    The y leg is first mapped to a strictly positive tensor via ``pos``
    (default: softplus(y) + eps). The x leg is clamped to keep exp(x)
    finite. Computation runs in fp32 inside an autocast-disabled region
    so the operator is robust under bf16/fp16 mixed precision.
    """
    if clamp is None:
        clamp = _FP32_CLAMP if x.dtype == torch.float32 else _LOWPREC_CLAMP

    out_dtype = torch.promote_types(x.dtype, y.dtype)

    # Run in at least fp32 precision: if inputs are bf16/fp16, promote; if
    # they are already fp32 or fp64, keep their precision intact (gradcheck
    # requires fp64 to round-trip exactly).
    work_dtype = out_dtype if out_dtype in (torch.float32, torch.float64) else torch.float32

    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        xw = x.to(work_dtype)
        yw = y.to(work_dtype)

        if pos == "exp_reparam":
            ln_y = yw.clamp(-clamp, clamp)
        else:
            y_pos = _positive(yw, pos, eps)
            ln_y = torch.log(y_pos)

        exp_x = torch.exp(xw.clamp(-clamp, clamp))
        out = exp_x - ln_y

    return out.to(out_dtype)
