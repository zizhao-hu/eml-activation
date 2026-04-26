.PHONY: help venv install test expressivity baseline eml1 eml2 multiseed sample paper clean clean-runs distclean

PY ?= python

help:
	@echo "Targets:"
	@echo "  make venv           Create .venv (python -m venv .venv)"
	@echo "  make install        Install package + deps into the active env"
	@echo "  make test           Run pytest"
	@echo "  make expressivity   Run Mode-A/Mode-B sanity checks; writes expressivity.png"
	@echo "  make baseline       Train softmax baseline (CPU config)"
	@echo "  make eml1           Train --attn eml-norm-v1"
	@echo "  make eml2           Train --attn eml-norm-v2"
	@echo "  make multiseed      Reproduce the paper's iso-iter table (3 seeds, ~3 min)"
	@echo "  make sample         Sample text from the trained eml-norm-v1 checkpoint"
	@echo "  make paper          Compile paper/paper.pdf (needs pdflatex)"
	@echo "  make clean          Remove __pycache__/.pytest_cache"
	@echo "  make clean-runs     Remove training checkpoints/logs in runs/"
	@echo "  make distclean      clean + clean-runs + remove .venv and downloaded data"

venv:
	$(PY) -m venv .venv
	@echo "Now run: source .venv/bin/activate && make install"

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q tests/

expressivity:
	$(PY) experiments/expressivity_demo.py

baseline:
	$(PY) experiments/charlm_transformer.py --attn softmax --config configs/charlm_cpu.yaml

eml1:
	$(PY) experiments/charlm_transformer.py --attn eml-norm-v1 --config configs/charlm_cpu.yaml

eml2:
	$(PY) experiments/charlm_transformer.py --attn eml-norm-v2 --config configs/charlm_cpu.yaml

multiseed:
	$(PY) experiments/multiseed.py

sample:
	$(PY) experiments/charlm_transformer.py --attn eml-norm-v1 --config configs/charlm_cpu.yaml \
	    --sample "ROMEO:" --max_new_tokens 200

paper:
	cd paper && pdflatex -interaction=nonstopmode paper.tex \
	    && bibtex paper \
	    && pdflatex -interaction=nonstopmode paper.tex \
	    && pdflatex -interaction=nonstopmode paper.tex
	@echo "Built paper/paper.pdf"

clean:
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-runs:
	rm -rf runs/

distclean: clean clean-runs
	rm -rf .venv data/tinyshakespeare.txt eml.egg-info src/eml.egg-info
