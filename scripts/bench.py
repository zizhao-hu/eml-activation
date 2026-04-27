"""Forward + backward speed bench across FFN variants.

Usage:
    python scripts/bench.py
"""
import argparse
import csv
import time
from pathlib import Path
from statistics import median

import torch

from eml_attn.model import GPT, GPTConfig


def time_one(model: GPT, x: torch.Tensor, y: torch.Tensor, mode: str, n_iters: int, device: str) -> float:
    """Return median ms/iter over n_iters timed iterations."""
    times = []
    for _ in range(n_iters):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if mode == "fwd":
            with torch.no_grad():
                _, _ = model(x, y)
        else:
            _, loss = model(x, y)
            model.zero_grad(set_to_none=True)
            loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return median(times)


def bench_one_variant(ffn_kind: str, T: int, batch: int, device: str, warmup: int, n_iters: int, vocab_size: int) -> dict:
    cfg = GPTConfig(vocab_size=vocab_size, block_size=T, n_layer=4, n_head=4, d_model=128, d_ff_mult=4, ffn_kind=ffn_kind)
    model = GPT(cfg).to(device)
    x = torch.randint(0, cfg.vocab_size, (batch, T), device=device)
    y = torch.randint(0, cfg.vocab_size, (batch, T), device=device)

    # warmup
    for _ in range(warmup):
        _, loss = model(x, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()

    fwd_ms = time_one(model, x, y, "fwd", n_iters, device)
    fb_ms = time_one(model, x, y, "fb", n_iters, device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        _, loss = model(x, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_mb = float("nan")

    n_params = sum(p.numel() for p in model.parameters())
    tokens_per_sec_fb = (batch * T) / (fb_ms / 1000)
    return {
        "ffn": ffn_kind,
        "T": T,
        "batch": batch,
        "n_params": n_params,
        "fwd_ms": fwd_ms,
        "fb_ms": fb_ms,
        "tokens_per_sec": tokens_per_sec_fb,
        "peak_mb": peak_mb,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ts", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--n_iters", type=int, default=50)
    ap.add_argument("--device", default=None)
    ap.add_argument("--vocab_size", type=int, default=65)
    ap.add_argument("--out", default="runs/bench.csv")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    rows = []
    for T in args.Ts:
        for kind in ("relu", "gelu", "swiglu", "emlglu"):
            print(f"benching ffn={kind} T={T} ...", end=" ", flush=True)
            row = bench_one_variant(kind, T, args.batch, device, args.warmup, args.n_iters, args.vocab_size)
            print(f"fwd={row['fwd_ms']:.2f} ms  fwd+bwd={row['fb_ms']:.2f} ms  {row['tokens_per_sec']:.0f} tok/s  params={row['n_params']:,}")
            rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
