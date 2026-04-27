"""EML primitive: eml(x, y) = exp(x) - ln(y).

Real-valued restriction: y is passed through softplus + eps to enforce y > 0.
x is clamped to [-CLAMP_X, CLAMP_X] before exp to prevent overflow.
"""
import torch
import torch.nn.functional as F

CLAMP_X = 10.0
EPS = 1e-6


def eml(x: torch.Tensor, y: torch.Tensor, eps: float = EPS, clamp_x: float = CLAMP_X) -> torch.Tensor:
    x_safe = x.clamp(-clamp_x, clamp_x)
    y_safe = F.softplus(y) + eps
    return torch.exp(x_safe) - torch.log(y_safe)
