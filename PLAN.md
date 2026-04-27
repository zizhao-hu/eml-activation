# EML in the Transformer MLP — Sanity Check

## Context

The repo `C:\Users\zizha\development\eml-activation` was just cloned/renamed; it currently holds only a stale README and `.gitignore`. The new direction: experiment with **EML (Exponential Minus Log)** — `eml(x, y) = exp(x) − ln(y)`, the recent universal binary operator from [Odrzywołek 2026](https://arxiv.org/abs/2603.21852) — as the **nonlinearity inside the transformer MLP (FFN) block**, replacing the standard activation.

> Two design clarifications from the user, both correct:
> 1. **Tree depth comes from layer stacking, not from a hand-built tree.** EML is one binary node. A stack of `[Linear → eml → Linear → eml → …]` across L transformer blocks composes a depth-L EML tree along each forward path — the same way ReLU + linear stacking gives MLP universality. So we use a single EML node as the activation primitive and let the network do the composition.
> 2. **Target the MLP, not attention.** EML is binary; SwiGLU-style MLPs are binary (two parallel projections, gated combination). The slot is already there. Attention is the smaller block in modern transformers, has a softmax bottleneck, and the paper itself never applies EML to similarity functions — only to feature vectors from MLP trunks. The MLP target is structurally and conceptually a better fit.

Goal: quick sanity check (~1–2 days), local hardware only, train tiny transformers from scratch from {ReLU, GELU, SwiGLU, **EML-GLU**} variants of the FFN, compare loss curves and forward/backward speed.

## What EML is, per the actual paper

After reading [Odrzywołek 2026](https://arxiv.org/abs/2603.21852) (the original) and [Ipek 2026](https://arxiv.org/abs/2604.13871) (implementation-focused follow-up):

- **Operator (Eq. 1):** `eml(x, y) = exp(x) − ln(y)`. Formally over `ℂ²` because some constants (`i`, `π`) require `ln(-1)`, but for our purposes we restrict to reals and accept some loss of universality (the paper concedes a real-only Sheffer is impossible).
- **Universality (Theorem 1):** Paired with the constant 1, the grammar `S → 1 | x | eml(S, S)` generates every standard elementary function as a binary tree of identical nodes. Worked examples from the paper:
  - `exp(x) = eml(x, 1)` (trivial, depth 1)
  - `ln(z) = eml(1, eml(eml(1, z), 1))` (depth 3 — the constants `e` introduced by inner `exp(1)` cancel against outer `ln`)
  - From `exp` and `ln` alone, all algebraic + transcendental functions follow.
- **Practical depth:** Empirical recovery rates: 100% at D=2, ~25% at D=3–4, <1% at D=5. **In our setting we don't build an explicit tree** — the transformer's layer stack provides the composition depth, and each EML node sees a fresh affine projection from the previous block's output.
- **Numerical caveats (paper's own Sec 6):** `exp` overflows easily in compositions; `ln(y)` requires `y > 0`. Clamping is required and acknowledged to disrupt exact gradient flow. The paper uses `complex128` internally; we restrict to real and add `softplus + ε` to enforce `y > 0`.
- **Per-node FLOPs (Eq. 12):** ~111 vs ~1 for ReLU, ~30 for GELU. The paper itself states EML *cannot* accelerate inference or training on CPU/GPU; the hardware story is FPGA/analog-only. We bench honestly and frame results as quality-at-equal-FLOPs, not wall-clock parity.

## How we use EML in the MLP

Standard transformer FFN block (per layer):
```
ReLU  variant:  out = W_o · ReLU(W_i · h)                          # GPT-1 style
GELU  variant:  out = W_o · GELU(W_i · h)                          # GPT-2 style
SwiGLU variant: out = W_o · ( SiLU(W_g · h) ⊙ (W_i · h) )           # LLaMA/Mistral style
```

The SwiGLU pattern is two parallel projections from the residual `h`, combined via a gating function. EML, being a binary operator `(x, y) → exp(x) − ln(y)`, slots into exactly this pattern as a new gating function — call it **EML-GLU**:

```
EML-GLU variant:  X = W_x · h
                  Y = W_y · h
                  out = W_o · eml(X, Y)
                       = W_o · ( exp(X) − log(softplus(Y) + ε) )    # element-wise on [B, T, d_ff]
```

Both `W_x` and `W_y` map `d → d_ff`; `W_o` maps `d_ff → d`. Same parameter shape as SwiGLU (which also has two `d → d_ff` projections + one `d_ff → d`). To keep total FFN params matched against the ReLU/GELU baselines (which only have one `d → d_ff` projection), we'll halve `d_ff` for the GLU variants per the SwiGLU paper's convention (e.g. `d_ff = 4d/2 = 2d` for the gated variants vs `d_ff = 4d` for the ungated — gives equal param count).

### Three points on the design

1. **Single EML node, element-wise.** No hand-built tree; stacking transformer layers does the composition. This is the right reading of "EML is universal via tree composition" — let the depth fall out of the architecture.
2. **`softplus(Y) + ε` for the `ln` argument.** Paper uses complex arithmetic for `y < 0`; we restrict to real and pay the price (some functions in EML's universal family aren't reachable). For a learned activation, this is fine — the gradient of `softplus` is `sigmoid`, which is well-behaved.
3. **Clamp `exp(X)` input to `[-10, 10]`.** Paper warns about overflow; this caps `exp(X) ≤ e¹⁰ ≈ 22000`, comfortably finite in fp32. Applied via a `.clamp(-10, 10)` before the exp.

### Recommended runs

| Run | FFN variant | Activation | Purpose |
|---|---|---|---|
| 1 | ReLU MLP    | `ReLU(W·h)`                          | Vanilla baseline |
| 2 | GELU MLP    | `GELU(W·h)`                          | GPT-2 baseline |
| 3 | SwiGLU      | `SiLU(W₁·h) ⊙ (W₂·h)`                 | Modern LLM baseline (LLaMA-style) |
| 4 | **EML-GLU** | `exp(W₁·h.clamp(±10)) − log(softplus(W₂·h)+ε)` | **Our proposal** |

Four runs at ~1M params, T=128, shakespeare_char, 2k steps. ~20–40 min each on CPU; faster on GPU.

## Project structure (greenfield, src layout, uv-managed)

```
eml-activation/
├── pyproject.toml
├── README.md                   # rewrite — kill the stale proof-walkthrough sketch
├── src/eml_attn/               # keep package name "eml_attn" — repo is named eml-activation, the package can be renamed later
│   ├── ops/eml.py              # eml(x, y, eps=1e-6) primitive with input clamping
│   ├── ffn.py                  # MLP variants: relu / gelu / swiglu / emlglu, all selectable by name
│   ├── attention.py            # standard CausalSelfAttention (uses F.scaled_dot_product_attention; identical across runs)
│   ├── model.py                # nanoGPT-style decoder; ffn_kind config flag threaded through
│   └── data.py                 # shakespeare_char loader + simple batcher
├── scripts/
│   ├── train.py                # CLI: --ffn relu|gelu|swiglu|emlglu --steps N --out runs/...
│   ├── bench.py                # forward + backward timing per variant
│   └── viz.py                  # plot loss curves and FFN activation distributions
├── runs/                       # per-experiment output: config.json, loss.csv, ckpt.pt
└── tests/
    ├── test_eml_op.py          # numerics, gradients, y → 0 edge case
    └── test_ffn_shapes.py      # all variants produce same output shape/dtype
```

Setup (run once on local Windows):
```bash
cd C:\Users\zizha\development\eml-activation
uv venv .venv --python 3.11
uv pip install torch numpy tiktoken pytest matplotlib
```

## Critical files (in dependency order)

- [eml-activation/src/eml_attn/ops/eml.py](eml-activation/src/eml_attn/ops/eml.py) — `eml(x, y, eps=1e-6)`: clamps `x` to `[-10, 10]` before `exp`, applies `softplus(y) + ε` before `ln`, returns `exp(x_clamped) - log(softplus(y) + ε)`. Element-wise; supports broadcasting; works with autograd.
- [eml-activation/src/eml_attn/ffn.py](eml-activation/src/eml_attn/ffn.py) — four FFN modules sharing a common `forward(self, h) → out` interface: `ReLUMLP`, `GELUMLP`, `SwiGLU`, `EMLGLU`. The GLU variants halve `d_ff` to match parameter count against the ungated variants. Factory: `build_ffn(kind, d_model, d_ff_mult=4)`.
- [eml-activation/src/eml_attn/model.py](eml-activation/src/eml_attn/model.py) — minimal nanoGPT (4 layers, 128 hidden, 4 heads, ctx 128). Single `ffn_kind` config flag passed into all blocks. Attention is unchanged across runs.
- [eml-activation/scripts/train.py](eml-activation/scripts/train.py) — CLI with `--ffn`, `--steps`, `--lr`, `--out`. Saves `runs/<id>/{config.json, loss.csv, ckpt.pt}`.
- [eml-activation/scripts/bench.py](eml-activation/scripts/bench.py) — for each `ffn_kind`, time 100 forward + 100 forward-backward passes at fixed `(B, T)` after 20-step warmup; report median ms/step and tokens/sec. Same model, same data, same hardware.

## Implementation phases

1. **Phase 0 — Scaffold (~1 hr).** Init `pyproject.toml`, src tree, README rewrite. Verify `uv venv` + `uv pip install torch` works on local Windows. Confirm CUDA availability (`torch.cuda.is_available()`) and pick fp32 (CPU) or bf16 (GPU) accordingly. Decide on tokenizer (tiktoken `gpt2` or character-level) — character-level is simpler, ~65 vocab on shakespeare.
2. **Phase 1 — EML op + ReLU baseline (~2 hrs).** Implement `ops/eml.py` with unit tests: finite-difference gradient check, `y → −∞` clamps via softplus, `x = 50` clamps before exp. Implement `ffn.py` with `ReLUMLP` and `GELUMLP` first. Implement `model.py`. Train Run 1 (ReLU) for 2k steps; confirm val loss reaches nanoGPT-typical range (~1.5–1.8 bpc).
3. **Phase 2 — Add GLU variants (~2 hrs).** Add `SwiGLU` and `EMLGLU` to `ffn.py`. Match params via halved `d_ff`. Run Run 2 (GELU), Run 3 (SwiGLU), Run 4 (EML-GLU) for 2k steps each.
4. **Phase 3 — Speed bench (~1 hr).** `bench.py` across all four variants at `T ∈ {64, 128, 256}`, batch=8, fixed seed, 100-step median after 20-step warmup. Report ms/step (forward), ms/step (forward+backward), tokens/sec, peak memory.
5. **Phase 4 — Writeup (~1 hr).** Loss curves overlaid (4 lines on one chart), per-variant final val loss table, throughput bar chart at T=128, brief README "Findings" section honestly stating what worked, what was slower, what was within noise.

Total: ~1–1.5 days of active work, plus training wall-clock (likely 1–3 hours total across runs).

## Verification

- **EML op unit tests:** finite-difference gradient check vs autograd (rtol 1e-3 in fp32); `eml(x, y)` with `y = -100` clamps via softplus rather than NaN; `exp(x)` for `x = 50` clamps rather than overflowing.
- **FFN shape parity:** for random `h` of shape `[2, 16, 128]`, all four FFN variants produce `[2, 16, 128]` output, identical dtype/device.
- **Param-count parity:** total params for each FFN variant within ±5% — print at startup. SwiGLU and EML-GLU should match each other exactly; ReLU/GELU should match each other; the gated pair should approximately match the ungated pair (via halved `d_ff`).
- **ReLU baseline sanity:** shakespeare_char val loss at 2k steps should land near nanoGPT numbers (~1.5 bpc); if not, the harness is broken and no comparison is meaningful.
- **EML-GLU initial-step sanity:** before launching the full 2k-step run, do a 50-step run and confirm loss decreases monotonically. If it NaNs, lower LR 10× and retry once before declaring blocked.
- **Speed-bench fairness:** same model, same batch, same dtype, same device, identical warmup, median over 100 timed steps not mean. Be transparent that ReLU/GELU each call one matmul + one elementwise op while SwiGLU/EML-GLU each do two parallel matmuls — that's the architecture, not a measurement artifact.

## Risk register (top 3)

1. **EML-GLU NaNs early in training.** The paper itself flags numerical fragility (Sec 6.2: "Compositions of exp overflow easily; clamping is required and breaks exact gradient flow"). Mitigation: input clamp `[-10, 10]` before exp; `softplus + 1e-6` before log; LayerNorm on the residual before the FFN block (already standard); fp32 inside the EML op even if model is bf16. If still NaN within first 100 steps, halve LR; if still NaN, drop `d_ff` multiplier from 4× to 2×.
2. **EML-GLU is just a different parameterization of SwiGLU and learns nothing new.** This is a *positive* outcome to write up — "EML used as an MLP activation behaves equivalently to SwiGLU at this scale, despite its theoretical universality" is a legitimate, honest finding. The interesting question is whether it diverges from SwiGLU in any direction (worse, better, similar trajectory). Plot the loss curves overlaid; if they sit on top of each other within noise, that's the result.
3. **CPU is too slow / no GPU available.** A ~1M nanoGPT trains shakespeare_char on CPU in ~30 min for 2k steps, manageable. Confirm in Phase 0; if CPU-only and too slow, drop to ~500K params and ctx 64, or accept a ~1 hr per-run wall-clock.

## Out of scope (explicitly)

- **Attention modifications.** The user's first instinct (replace softmax / replace QK) is set aside in favor of MLP. Could revisit later if MLP results are clean and there's appetite.
- **EML tree with hand-built topology** (the depth-2 master formula from Odrzywołek Sec 4.3). Replaced by "single EML node + layer stacking does the composition."
- **Symbolic snapping** (Gumbel-softmax + hardening on `(α, β, γ)` to recover closed-form expressions). Paper's interpretability use case, not ours.
- **Complex-domain EML.** We restrict to real; loss of universality is acceptable.
- **CARC / SLURM, pretrained-model finetuning, lm-eval-harness.** Skipped per the user's quick-sanity-check + local-only constraints.
