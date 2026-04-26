"""Expressivity sanity checks for EML.

Mode A: verify closed-form EML compositions match torch.{exp, log, softmax}.
Mode B: train a small parameterized EML tree to imitate sigmoid and softmax.
Saves a plot to expressivity.png and prints a summary table.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from eml.expressivity import (  # noqa: E402
    EMLTree,
    closed_form_exp,
    closed_form_ln,
    closed_form_softmax,
    fit_target,
)


def mode_a_checks() -> dict[str, float]:
    """Run Mode-A closed-form checks. Returns max abs error vs torch reference."""
    torch.manual_seed(0)
    out: dict[str, float] = {}

    x_pos = torch.linspace(0.05, 5.0, 200, dtype=torch.float64)
    out["closed_form_exp_vs_torch"] = float(
        (closed_form_exp(x_pos) - torch.exp(x_pos)).abs().max()
    )
    out["closed_form_ln_vs_torch"] = float(
        (closed_form_ln(x_pos) - torch.log(x_pos)).abs().max()
    )

    L = torch.randn(64, 8, dtype=torch.float64)
    sm_eml = closed_form_softmax(L, dim=-1)
    sm_ref = torch.softmax(L, dim=-1)
    out["closed_form_softmax_vs_torch"] = float((sm_eml - sm_ref).abs().max())

    return out


def mode_b_fit_sigmoid() -> dict:
    torch.manual_seed(1)
    model = EMLTree(dim_in=1, dim_hidden=16, dim_out=1, depth=3)

    def sampler(n: int) -> torch.Tensor:
        return torch.empty(n, 1).uniform_(-5.0, 5.0)

    res = fit_target(
        target=torch.sigmoid,
        model=model,
        sampler=sampler,
        steps=2000,
        batch=256,
        lr=3e-3,
        loss_fn="mse",
    )
    res["model"] = model
    return res


def mode_b_fit_softmax() -> dict:
    torch.manual_seed(2)
    n = 4
    model = EMLTree(dim_in=n, dim_hidden=32, dim_out=n, depth=2)

    def sampler(b: int) -> torch.Tensor:
        return torch.randn(b, n) * 1.5

    def target(x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=-1)

    res = fit_target(
        target=target,
        model=model,
        sampler=sampler,
        steps=2500,
        batch=256,
        lr=3e-3,
        loss_fn="kl",
    )
    res["model"] = model
    return res


def main() -> None:
    print("== Mode A — closed-form EML compositions vs torch ==")
    mode_a = mode_a_checks()
    for name, err in mode_a.items():
        print(f"  {name:<35s} max|err| = {err:.3e}")

    print("\n== Mode B — parameterized EML tree fits ==")
    sig = mode_b_fit_sigmoid()
    print(f"  sigmoid (depth=3, hidden=16) MSE: {sig['final']:.3e}")

    sm = mode_b_fit_softmax()
    print(f"  softmax n=4 (depth=2, hidden=32) KL: {sm['final']:.3e}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    # sigmoid fit visualization
    xs = torch.linspace(-5, 5, 200).unsqueeze(-1)
    with torch.no_grad():
        ys = sig["model"](xs).squeeze(-1)
    ax[0].plot(xs.squeeze(-1), torch.sigmoid(xs).squeeze(-1), label="sigmoid (target)", lw=2)
    ax[0].plot(xs.squeeze(-1), ys, "--", label="EML tree fit")
    ax[0].set_title(f"sigmoid via Mode-B EML tree (MSE={sig['final']:.2e})")
    ax[0].legend()

    # sigmoid loss curve
    ax[1].semilogy(sig["history"])
    ax[1].set_title("sigmoid fit loss (MSE, log-y)")
    ax[1].set_xlabel("step")

    # softmax loss curve
    ax[2].semilogy(sm["history"])
    ax[2].set_title("softmax-n4 fit loss (KL, log-y)")
    ax[2].set_xlabel("step")

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), os.pardir, "expressivity.png")
    fig.savefig(out, dpi=120)
    print(f"\nSaved plot to {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
