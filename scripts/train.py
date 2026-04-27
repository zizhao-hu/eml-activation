"""Train a tiny GPT with a chosen FFN variant on shakespeare_char.

Usage:
    python scripts/train.py --ffn relu --steps 2000 --out runs/relu
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch

from eml_attn.data import CharDataset
from eml_attn.model import GPT, GPTConfig


def get_lr(step: int, max_steps: int, max_lr: float, min_lr: float, warmup: int) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup) / (max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(model: GPT, dataset: CharDataset, batch_size: int, block_size: int, eval_iters: int, device: str) -> dict[str, float]:
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = dataset.get_batch(split, batch_size, block_size, device=device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffn", required=True, choices=["relu", "gelu", "swiglu", "emlglu"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--d_ff_mult", type=int, default=4)
    ap.add_argument("--max_lr", type=float, default=3e-3)
    ap.add_argument("--min_lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--eval_interval", type=int, default=200)
    ap.add_argument("--eval_iters", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save_ckpt", action="store_true", default=False)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dataset = CharDataset()
    cfg = GPTConfig(
        vocab_size=dataset.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        d_ff_mult=args.d_ff_mult,
        ffn_kind=args.ffn,
    )
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ffn={args.ffn}  params={n_params:,}  config={cfg}")

    (out / "config.json").write_text(json.dumps({**vars(args), "vocab_size": dataset.vocab_size, "n_params": n_params}, indent=2))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    log_path = out / "loss.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "lr", "train_loss", "val_loss", "step_ms"])

        t_start = time.perf_counter()
        for step in range(args.steps):
            lr = get_lr(step, args.steps, args.max_lr, args.min_lr, args.warmup)
            for g in optimizer.param_groups:
                g["lr"] = lr

            t0 = time.perf_counter()
            xb, yb = dataset.get_batch("train", args.batch_size, args.block_size, device=device)
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step_ms = (time.perf_counter() - t0) * 1000

            if not torch.isfinite(loss):
                print(f"!!! step {step}: loss is {loss.item()} — aborting")
                break

            if step % args.eval_interval == 0 or step == args.steps - 1:
                losses = estimate_loss(model, dataset, args.batch_size, args.block_size, args.eval_iters, device)
                elapsed = time.perf_counter() - t_start
                print(f"step {step:5d} | lr {lr:.2e} | train {losses['train']:.4f} | val {losses['val']:.4f} | {step_ms:.1f} ms/step | {elapsed:.1f} s elapsed")
                writer.writerow([step, lr, losses["train"], losses["val"], step_ms])
                f.flush()

    if args.save_ckpt:
        torch.save({"model": model.state_dict(), "cfg": vars(cfg)}, out / "ckpt.pt")
        print(f"saved checkpoint to {out / 'ckpt.pt'}")
    else:
        print("done (ckpt not saved; pass --save_ckpt to keep)")


if __name__ == "__main__":
    main()
