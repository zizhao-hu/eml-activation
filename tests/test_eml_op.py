import torch

from eml_attn.ops.eml import eml, CLAMP_X, EPS


def test_shape_and_dtype():
    x = torch.randn(4, 8)
    y = torch.randn(4, 8)
    out = eml(x, y)
    assert out.shape == (4, 8)
    assert out.dtype == torch.float32


def test_broadcasting():
    x = torch.randn(4, 1)
    y = torch.randn(1, 8)
    out = eml(x, y)
    assert out.shape == (4, 8)


def test_no_nan_for_negative_y():
    x = torch.zeros(10)
    y = torch.full((10,), -1000.0)
    out = eml(x, y)
    assert torch.isfinite(out).all(), f"got non-finite: {out}"


def test_no_inf_for_huge_x():
    x = torch.full((10,), 1000.0)
    y = torch.ones(10)
    out = eml(x, y)
    assert torch.isfinite(out).all(), f"got non-finite: {out}"
    # exp(CLAMP_X) - log(softplus(1)+eps) ≈ exp(10) - log(1.31)
    expected = torch.exp(torch.tensor(CLAMP_X)) - torch.log(torch.nn.functional.softplus(torch.tensor(1.0)) + EPS)
    assert torch.allclose(out, expected.expand_as(out), rtol=1e-5)


def test_gradient_finite_difference():
    """Compare autograd to central-difference gradient on a few random points."""
    torch.manual_seed(0)
    x = (torch.randn(5) * 0.5).requires_grad_(True)  # keep within clamp range
    y = (torch.randn(5) * 0.5).requires_grad_(True)
    out = eml(x, y).sum()
    out.backward()

    eps = 1e-3
    for i in range(5):
        for var, grad in [(x, x.grad), (y, y.grad)]:
            v0 = var[i].item()
            var.data[i] = v0 + eps
            f_plus = eml(x.detach(), y.detach()).sum().item()
            var.data[i] = v0 - eps
            f_minus = eml(x.detach(), y.detach()).sum().item()
            var.data[i] = v0
            num_grad = (f_plus - f_minus) / (2 * eps)
            assert abs(num_grad - grad[i].item()) < 1e-2, f"grad mismatch at {i}: num={num_grad} auto={grad[i].item()}"


def test_eml_x_one_recovers_exp():
    """eml(x, 1) should ≈ exp(x) - log(softplus(1)+eps) ≈ exp(x) - 0.31"""
    x = torch.tensor([0.0, 1.0, 2.0])
    y = torch.ones(3)
    out = eml(x, y)
    expected = torch.exp(x) - torch.log(torch.nn.functional.softplus(torch.tensor(1.0)) + EPS)
    assert torch.allclose(out, expected, rtol=1e-5)
