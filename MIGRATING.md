# Migrating to local development

This project was developed inside an ephemeral web sandbox. Everything that
matters is committed and pushed to GitHub — to develop locally you just clone
the branch and install the deps.

## Prerequisites

- Python 3.10+
- `git`
- Optional: `pdflatex` + `bibtex` if you want to recompile the paper
  (`brew install --cask mactex-no-gui` on macOS, `apt-get install texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra` on Debian/Ubuntu)
- Optional: a CUDA GPU. PyTorch will auto-detect it.

## One-shot setup

```bash
git clone https://github.com/zizhao-hu/llm-proof-walkthrough.git
cd llm-proof-walkthrough
git checkout claude/test-exp-minus-log-activation-wm1a5

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
make install                        # = pip install -e ".[dev]"

make test                           # pytest, ~2 s
make expressivity                   # sanity checks, writes expressivity.png
make multiseed                      # reproduces the paper's tables (~3 min on CPU)
```

If you don't have `make`, the equivalent direct commands are listed in the
Makefile (run `cat Makefile` or read the file in your editor).

## GPU vs CPU

The training script auto-detects CUDA. The `configs/charlm.yaml` config is
sized for a single GPU (d=384, 6 layers, block=128, 1500 iters);
`configs/charlm_cpu.yaml` is the small CPU-friendly version that produced the
paper's tables. Override per run:

```bash
python experiments/charlm_transformer.py --attn eml-norm-v1 \
    --config configs/charlm.yaml --iters 1500
```

## Where to develop

`src/eml/` is the package. The four files that matter:
- `operator.py` — the EML primitive itself (numerics live here).
- `attention.py` — the `--attn` dispatch; add new variants here.
- `transformer.py` — TinyGPT scaffolding.
- `expressivity.py` — Mode-A closed-form refs and Mode-B parameterized trees.

`experiments/` holds runnable scripts; `configs/` holds the YAML configs;
`tests/` holds pytest. New experiments are usually a new file in
`experiments/` plus, optionally, a new config.

## Branch and contribution flow

The current development branch is `claude/test-exp-minus-log-activation-wm1a5`
(main is empty placeholder). Open a PR from feature branches into this branch,
or rename and push to `main` once you're happy.

```bash
git checkout -b feature/my-change
# ... edit ...
make test
git commit -am "..."
git push -u origin feature/my-change
```

## Submitting the paper from your machine

If you want to recompile or edit:

```bash
cd paper
# edit paper.tex (e.g. de-anonymize the author block)
make -C .. paper        # or: pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper

# rebuild the arXiv tarball
tar --exclude='*.aux' --exclude='*.log' --exclude='*.out' --exclude='*.blg' \
    -czf eml_attention_arxiv.tar.gz paper.tex refs.bib paper.bbl neurips_2024.sty
```

Then upload `paper/eml_attention_arxiv.tar.gz` to <https://arxiv.org/submit>.
Full guide in `paper/SUBMISSION.md`.
