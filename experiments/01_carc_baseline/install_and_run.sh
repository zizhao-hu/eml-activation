#!/bin/bash
#SBATCH --job-name=eml_install_run
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/zizhaoh/eml-activation/logs/%x_%j.log
#SBATCH --error=/project2/jessetho_1732/zizhaoh/eml-activation/logs/%x_%j.log

set -eo pipefail
module purge
module load gcc/13.3.0 cuda/12.6.3
export CUDA_HOME=$CUDA_ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch1/$USER/.cache/huggingface

VENV=/scratch1/zizhaoh/envs/eml-activation

echo "=== node info ==="
hostname
nvidia-smi -L
date

echo
echo "=== ensure venv exists ==="
if [ ! -f "$VENV/bin/activate" ]; then
    /home1/zizhaoh/.local/bin/uv venv "$VENV" --python 3.11
fi
source "$VENV/bin/activate"

echo
echo "=== install torch (CUDA 12.6) + project deps ==="
/home1/zizhaoh/.local/bin/uv pip install --index-url https://download.pytorch.org/whl/cu126 torch
/home1/zizhaoh/.local/bin/uv pip install numpy matplotlib tqdm pytest

echo
echo "=== install eml_attn package (no deps; deps already installed above) ==="
cd /project2/jessetho_1732/zizhaoh/eml-activation
/home1/zizhaoh/.local/bin/uv pip install -e . --no-deps

echo
echo "=== verify torch + cuda ==="
python -c "import torch; print(f'torch={torch.__version__}'); print(f'cuda={torch.cuda.is_available()}'); print(f'device={torch.cuda.get_device_name(0)}')"

echo
echo "=== run pytest ==="
python -m pytest tests/ -v

echo
echo "=== training runs (5 layers, d_model 384, ctx 256, 5k steps each) ==="
mkdir -p runs_carc
COMMON="--steps 5000 --batch_size 64 --block_size 256 --n_layer 6 --n_head 6 --d_model 384 --eval_interval 250 --eval_iters 50"

for ffn in relu gelu swiglu emlglu; do
    echo
    echo ">>> training $ffn"
    python scripts/train.py --ffn $ffn $COMMON --out runs_carc/$ffn --seed 42
done

echo
echo "=== speed bench ==="
python scripts/bench.py --Ts 128 256 512 1024 --batch 16 --out runs_carc/bench.csv

echo
echo "=== make plots ==="
python scripts/viz.py --runs_dir runs_carc --out_dir runs_carc/figs

echo
echo "DONE  $(date)"
