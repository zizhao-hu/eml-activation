# EML-Attention: Exp-Minus-Log as a Softmax Replacement in Transformer Attention

Research project that replaces the softmax in transformer attention with normalization
built from a single binary primitive — the **Exp-Minus-Log (EML)** operator
introduced by Odrzywolek (arXiv 2603.21852, 2026):

```
eml(x, y) = exp(x) − ln(y)
```

By the universality of `{eml, 1}`, every elementary function is a finite binary tree
of identical EML nodes. Examples (exact, not approximate):

| Function | EML expression | Tree depth |
|---|---|---|
| `exp(x)` | `eml(x, 1)` | 1 |
| `ln(x)` | `eml(1, eml(eml(1, x), 1))` | 3 |
| `a − b` (a > 0) | `eml(ln(a), exp(b))` | 4 |

## What this project does

A separate paper (arXiv 2604.13871) embeds EML in a *neuro-symbolic head* aimed at
FPGA / analog hardware. **We do something different**: we put EML on a GPU, in the
spot it actually fits — the **binary normalization slot** of attention — and study
training dynamics.

Specifically, in a small char-level transformer (TinyGPT-style on TinyShakespeare),
we replace `softmax(QKᵀ/√d)` with one of:

- `softmax` — baseline.
- `eml-norm-v1` — single-projection EML normalization
  `r_ij = exp(L_ij) − log(Σ_k softplus(L_ik) + ε)`, then softplus + row-normalize.
- `eml-norm-v2` — **twin-projection** EML, a strict generalization of softmax: two
  separate K projections feed the two legs.

The FFN keeps its standard GeLU; only attention normalization is swapped.

## Why softmax, not a unary activation

EML is binary. Softmax is binary in structure (per-cell logit + per-row aggregate);
ReLU/GeLU are unary. The slots line up at softmax. Wiring EML into a unary slot
underuses its structure.

## Layout

```
src/eml/
  operator.py       # eml() with positivity / clamp / fp32 fence
  attention.py      # CausalSelfAttention with --attn flag
  transformer.py    # TinyGPT (~6 layers, d=384, n_head=6, block=128, byte vocab)
  expressivity.py   # Mode-A closed-form refs + Mode-B parameterized EML tree
experiments/
  expressivity_demo.py
  charlm_transformer.py
configs/charlm.yaml
tests/test_operator.py
```

## Quickstart

```
pip install -e .[dev]
pytest -q tests/

python experiments/expressivity_demo.py
python experiments/charlm_transformer.py --attn softmax     --iters 1500
python experiments/charlm_transformer.py --attn eml-norm-v1 --iters 1500
python experiments/charlm_transformer.py --attn eml-norm-v2 --iters 1500
python experiments/charlm_transformer.py --attn eml-norm-v1 --sample "ROMEO:" --max_new_tokens 200
```

TinyShakespeare is auto-downloaded on first run into `data/tinyshakespeare.txt`.

## Results

Char-level TinyGPT on TinyShakespeare, CPU, `configs/charlm_cpu.yaml`
(d=128, 4 layers × 4 heads, block=64, batch=32, AdamW + cosine LR, 500 iters,
mean ± std over 3 seeds {1337, 42, 7}, val_loss = mean cross-entropy on the
held-out 10% split). Reproduce with `python experiments/multiseed.py`.

### Iso-iter (same number of training steps)

| `--attn`       |   params | val_loss             | wall_clock |
|----------------|---------:|---------------------|-----------:|
| `softmax`      |  799,488 | 2.5033 ± 0.0013     |      23.3s |
| `eml-norm-v1`  |  799,488 | **2.3931 ± 0.0086** |      27.5s |
| `eml-norm-v2`  |  865,024 | **2.3875 ± 0.0104** |      31.7s |

EML beats softmax by ~0.11 nats — about 10× the inter-seed std, so the gap
is well outside noise.

### Iso-wall-clock (softmax given more iters to match EML's wall-clock)

| `--attn`       | iters | wall_clock | val_loss        |
|----------------|------:|-----------:|----------------:|
| `softmax`      |   500 |      22.5s | 2.5033 ± 0.0013 |
| `softmax`      |   600 |      27.3s | 2.4514 ± 0.0058 |
| `softmax`      |   700 |      33.0s | 2.4144 ± 0.0001 |
| `eml-norm-v1`  |   500 |      29.3s | **2.3931 ± 0.0086** |
| `eml-norm-v2`  |   500 |      32.0s | **2.3875 ± 0.0104** |

The shrunk-but-still-positive gap: **+0.058 nats** for v1 vs softmax at ~27s,
**+0.027 nats** for v2 vs softmax at ~33s. EML wins at compute parity.

### Speed (does NOT win)

EML is **structurally slower** on this CPU/GPU stack: the operator carries
extra `softplus / clamp / exp / log / softplus / sum / divide` ops and an
fp32 fence. Per-iter cost: **v1 ≈ +18%**, **v2 ≈ +36%** vs softmax.
A speed win would require dedicated EML hardware (FPGA / analog cells —
the focus of arXiv 2604.13871) which is out of scope here.

## References

- Odrzywolek, *All elementary functions from a single binary operator*, arXiv 2603.21852 (2026).
- *Hardware-Efficient Neuro-Symbolic Networks with the Exp-Minus-Log Operator*, arXiv 2604.13871 (2026).
