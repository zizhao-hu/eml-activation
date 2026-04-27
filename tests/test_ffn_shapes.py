import torch

from eml_attn.ffn import build_ffn


def test_all_variants_same_shape():
    h = torch.randn(2, 16, 128)
    for kind in ("relu", "gelu", "swiglu", "emlglu"):
        ffn = build_ffn(kind, d_model=128)
        out = ffn(h)
        assert out.shape == h.shape, f"{kind}: got {out.shape}"
        assert out.dtype == h.dtype


def test_param_counts_close():
    """ReLU and GELU match each other (same arch); SwiGLU and EML-GLU match each other.
    The gated pair should be within 5% of the ungated pair via halved d_ff."""
    counts = {}
    for kind in ("relu", "gelu", "swiglu", "emlglu"):
        ffn = build_ffn(kind, d_model=128)
        counts[kind] = sum(p.numel() for p in ffn.parameters())
        print(f"  {kind}: {counts[kind]:,}")
    assert counts["relu"] == counts["gelu"]
    assert counts["swiglu"] == counts["emlglu"]
    # within 10% to allow rounding to multiple of 8
    ratio = counts["swiglu"] / counts["relu"]
    assert 0.9 <= ratio <= 1.1, f"GLU/ungated param ratio = {ratio}"


def test_emlglu_finite():
    h = torch.randn(2, 16, 128) * 5  # bigger inputs to stress-test
    ffn = build_ffn("emlglu", d_model=128)
    out = ffn(h)
    assert torch.isfinite(out).all()


def test_emlglu_backward_finite():
    h = torch.randn(2, 16, 128, requires_grad=True)
    ffn = build_ffn("emlglu", d_model=128)
    out = ffn(h).sum()
    out.backward()
    assert h.grad is not None
    assert torch.isfinite(h.grad).all()
    for p in ffn.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
