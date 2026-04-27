"""Tiny Shakespeare character-level dataset.

Downloads from karpathy/char-rnn on first use, then caches to ./data/.
Simple deterministic train/val split (90/10).
"""
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class CharDataset:
    def __init__(self, data_dir: str | Path = "data", val_frac: float = 0.1):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        text_path = self.data_dir / "input.txt"
        if not text_path.exists():
            print(f"downloading shakespeare to {text_path} ...")
            urllib.request.urlretrieve(SHAKESPEARE_URL, text_path)
        text = text_path.read_text(encoding="utf-8")
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        ids = np.array([self.stoi[c] for c in text], dtype=np.int64)
        n_train = int(len(ids) * (1.0 - val_frac))
        self.train = ids[:n_train]
        self.val = ids[n_train:]
        print(f"vocab_size={self.vocab_size}, train tokens={len(self.train)}, val tokens={len(self.val)}")

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def get_batch(self, split: str, batch_size: int, block_size: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train if split == "train" else self.val
        ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + block_size]) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size]) for i in ix])
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
