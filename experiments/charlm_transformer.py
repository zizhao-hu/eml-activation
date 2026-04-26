"""Train a TinyGPT on TinyShakespeare under one of three attention variants.

usage:
    python experiments/charlm_transformer.py --attn softmax     --iters 1500
    python experiments/charlm_transformer.py --attn eml-norm-v1 --iters 1500
    python experiments/charlm_transformer.py --attn eml-norm-v2 --iters 1500
    python experiments/charlm_transformer.py --attn eml-norm-v1 --sample "ROMEO:" --max_new_tokens 200
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import urllib.request

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from eml.transformer import GPTConfig, TinyGPT  # noqa: E402

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def ensure_data(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading TinyShakespeare to {path} ...")
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, "rb") as f:
        return f.read().decode("utf-8")


def get_batch(data: np.ndarray, block_size: int, batch_size: int, device: torch.device):
    ix = np.random.randint(0, len(data) - block_size - 1, size=(batch_size,))
    x = np.stack([data[i : i + block_size] for i in ix])
    y = np.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x_t = torch.from_numpy(x).long().to(device, non_blocking=True)
    y_t = torch.from_numpy(y).long().to(device, non_blocking=True)
    return x_t, y_t


def cosine_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: TinyGPT, data: np.ndarray, cfg: dict, device: torch.device) -> float:
    model.eval()
    losses = []
    for _ in range(cfg["eval_iters"]):
        xb, yb = get_batch(data, cfg["block_size"], cfg["batch_size"], device)
        _, loss = model(xb, yb)
        losses.append(float(loss.detach()))
    model.train()
    return float(np.mean(losses))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", choices=["softmax", "eml-norm-v1", "eml-norm-v2"], default="softmax")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--config", default=os.path.join("configs", "charlm.yaml"))
    ap.add_argument("--data", default=os.path.join("data", "tinyshakespeare.txt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--sample", type=str, default=None,
                    help="if set, load latest checkpoint and generate from this prompt")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=50)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.iters is not None:
        cfg["iters"] = args.iters

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device(args.device)

    # Data — byte-level tokens.
    text = ensure_data(args.data)
    raw = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    split = int(0.9 * len(raw))
    train_data = raw[:split].astype(np.int64)
    val_data = raw[split:].astype(np.int64)
    print(f"train tokens: {len(train_data):,}   val tokens: {len(val_data):,}")

    gpt_cfg = GPTConfig(
        vocab_size=256,
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        d_model=cfg["d_model"],
        dropout=cfg["dropout"],
        attn=args.attn,
    )
    model = TinyGPT(gpt_cfg).to(device)
    print(f"model params: {model.num_params():,}   attn={args.attn}")

    os.makedirs(args.out_dir, exist_ok=True)
    run_tag = f"{args.attn}"
    ckpt_path = os.path.join(args.out_dir, f"{run_tag}.pt")
    csv_path = os.path.join(args.out_dir, f"{run_tag}.csv")

    if args.sample is not None:
        if not os.path.exists(ckpt_path):
            raise SystemExit(f"no checkpoint at {ckpt_path}; train first.")
        sd = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(sd["model"])
        prompt = args.sample.encode("utf-8")
        idx = torch.tensor([list(prompt)], dtype=torch.long, device=device)
        out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k)
        text_out = bytes(out[0].tolist()).decode("utf-8", errors="replace")
        print("\n--- sample ---\n" + text_out)
        return

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=tuple(cfg["betas"]),
        weight_decay=cfg["weight_decay"],
    )

    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["step", "train_loss", "val_loss", "lr"])

    model.train()
    pbar = tqdm(range(cfg["iters"]), desc=f"train {args.attn}")
    t0 = time.time()
    last_train = float("nan")
    for step in pbar:
        lr = cosine_lr(step, cfg["warmup_iters"], cfg["iters"], cfg["lr"])
        for g in optim.param_groups:
            g["lr"] = lr

        xb, yb = get_batch(train_data, cfg["block_size"], cfg["batch_size"], device)
        _, loss = model(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optim.step()
        last_train = float(loss.detach())

        if step % cfg["eval_interval"] == 0 or step == cfg["iters"] - 1:
            val_loss = evaluate(model, val_data, cfg, device)
            csv_w.writerow([step, last_train, val_loss, lr])
            csv_f.flush()
            pbar.set_postfix(train=f"{last_train:.3f}", val=f"{val_loss:.3f}", lr=f"{lr:.2e}")

        if step > 0 and step % cfg["sample_interval"] == 0:
            seed = torch.tensor([[ord("\n")]], dtype=torch.long, device=device)
            out = model.generate(seed, max_new_tokens=120, temperature=1.0, top_k=50)
            txt = bytes(out[0].tolist()).decode("utf-8", errors="replace")
            print(f"\n[step {step} sample] {txt!r}\n")

    csv_f.close()
    elapsed = time.time() - t0
    val_loss = evaluate(model, val_data, cfg, device)
    print(f"done. final val_loss = {val_loss:.4f}   elapsed = {elapsed:.1f}s")

    torch.save({"model": model.state_dict(), "cfg": cfg, "attn": args.attn}, ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")
    print(f"loss log: {csv_path}")


if __name__ == "__main__":
    main()
