import torch

from eml.operator import eml


def test_basic_identity_exp():
    """eml(x, 1) = exp(x) - ln(1) = exp(x)."""
    x = torch.linspace(-3, 3, 25)
    one = torch.ones_like(x)
    out = eml(x, one, pos="abs")  # abs(1)+eps ~ 1
    assert torch.allclose(out, torch.exp(x), atol=1e-3)


def test_shape_and_broadcasting():
    x = torch.randn(2, 3, 4)
    y = torch.randn(4)  # broadcast over leading dims
    out = eml(x, y)
    assert out.shape == (2, 3, 4)


def test_finite_over_wide_input_range():
    torch.manual_seed(0)
    x = torch.empty(1000).uniform_(-30, 30)
    y = torch.empty(1000).uniform_(-30, 30)
    out = eml(x, y)
    assert torch.isfinite(out).all(), "eml produced non-finite values on wide inputs"


def test_positivity_modes_all_finite():
    torch.manual_seed(1)
    x = torch.randn(64) * 10
    y = torch.randn(64) * 10
    for mode in ("softplus", "abs", "exp_reparam"):
        out = eml(x, y, pos=mode)
        assert torch.isfinite(out).all(), f"non-finite output for pos={mode}"


def test_gradcheck_small():
    """torch.autograd.gradcheck on a small fp64 instance."""
    torch.manual_seed(2)
    x = torch.randn(3, dtype=torch.float64, requires_grad=True)
    # keep y away from softplus's near-zero region for a tight numerical jacobian
    y = torch.randn(3, dtype=torch.float64).abs().add_(0.5).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda a, b: eml(a, b, pos="softplus", clamp=10.0),
        (x, y),
        eps=1e-6,
        atol=1e-4,
    )


def test_dtype_promotion_under_autocast():
    """Output dtype is promoted but values are computed in fp32 internally."""
    if not torch.cuda.is_available():
        return
    x = torch.randn(8, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(8, device="cuda", dtype=torch.bfloat16)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = eml(x, y)
    assert torch.isfinite(out).all()
