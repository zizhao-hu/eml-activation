"""Multi-seed comparison: softmax vs eml-norm-v1 vs eml-norm-v2.

Trains each attention variant under the same budget (iters from configs/charlm_cpu.yaml)
across N seeds, then reports val_loss mean ± std and wall-clock per run.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from eml.transformer import GPTConfig, TinyGPT  # noqa: E402

DATA_PATH = os.path.join("data", "tinyshakespeare.txt")
CONFIG_PATH = os.path.join("configs", "charlm_cpu.yaml")
SEEDS = (1337, 42, 7)


def load_data() -> tuple[np.ndarray, np.ndarray]:
    with open(DATA_PATH) as f:
        text = f.read()
    raw = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    split = int(0.9 * len(raw))
    return raw[:split].astype(np.int64), raw[split:].astype(np.int64)


def get_batch(data: np.ndarray, block: int, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    ix = np.random.randint(0, len(data) - block - 1, size=(batch,))
    x = np.stack([data[i : i + block] for i in ix])
    y = np.stack([data[i + 1 : i + block + 1] for i in ix])
    return torch.from_numpy(x).long(), torch.from_numpy(y).long()


def cosine_lr(step: int, warmup: int, total: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model: TinyGPT, data: np.ndarray, cfg: dict, n: int = 30) -> float:
    model.eval()
    losses = []
    for _ in range(n):
        xb, yb = get_batch(data, cfg["block_size"], cfg["batch_size"])
        _, loss = model(xb, yb)
        losses.append(float(loss))
    model.train()
    return float(np.mean(losses))


def train_once(attn: str, seed: int, cfg: dict, train_data: np.ndarray, val_data: np.ndarray):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gpt_cfg = GPTConfig(
        vocab_size=256,
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        d_model=cfg["d_model"],
        dropout=cfg["dropout"],
        attn=attn,
    )
    model = TinyGPT(gpt_cfg)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=tuple(cfg["betas"]),
        weight_decay=cfg["weight_decay"],
    )
    t0 = time.time()
    model.train()
    for step in range(cfg["iters"]):
        lr = cosine_lr(step, cfg["warmup_iters"], cfg["iters"], cfg["lr"])
        for g in optim.param_groups:
            g["lr"] = lr
        xb, yb = get_batch(train_data, cfg["block_size"], cfg["batch_size"])
        _, loss = model(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optim.step()
    dt = time.time() - t0
    return evaluate(model, val_data, cfg, 30), dt, model.num_params()


def main() -> None:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    train_data, val_data = load_data()
    results: dict[str, tuple[list[float], list[float], int]] = {}
    for attn in ("softmax", "eml-norm-v1", "eml-norm-v2"):
        vs, ts = [], []
        params = 0
        for seed in SEEDS:
            v, t, p = train_once(attn, seed, cfg, train_data, val_data)
            vs.append(v)
            ts.append(t)
            params = p
            print(f"  {attn:<14s} seed={seed}  val={v:.4f}  t={t:.1f}s")
        results[attn] = (vs, ts, params)
        print(
            f"  {attn:<14s} mean val={np.mean(vs):.4f} ± {np.std(vs):.4f}   "
            f"mean time={np.mean(ts):.1f}s"
        )

    print()
    header = f"{'attn':<14s}  {'params':>9s}  {'val_mean':>10s}   {'std':>6s}  {'time_mean':>9s}"
    print(header)
    print("-" * len(header))
    for k, (vs, ts, p) in results.items():
        print(
            f"{k:<14s}  {p:>9,d}  {np.mean(vs):>10.4f} ± {np.std(vs):>6.4f}  "
            f"{np.mean(ts):>8.1f}s"
        )


if __name__ == "__main__":
    main()
