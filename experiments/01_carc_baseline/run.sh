#!/bin/bash
#SBATCH --job-name=eml_carc
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/zizhaoh/eml-activation/logs/%x_%j.log
#SBATCH --error=/project2/jessetho_1732/zizhaoh/eml-activation/logs/%x_%j.log

set -eo pipefail
module purge
module load gcc/13.3.0 cuda/12.6.3
export CUDA_HOME=$CUDA_ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /scratch1/zizhaoh/envs/eml-activation/bin/activate
cd /project2/jessetho_1732/zizhaoh/eml-activation

echo "=== device info ==="
nvidia-smi -L
python -c "import torch; print(f'torch={torch.__version__}  cuda={torch.cuda.is_available()}  device={torch.cuda.get_device_name(0)}')"

# Larger model + more steps than the local sanity check
# d_model 384, 6 layers, 6 heads, ctx 256 ≈ 11M params
COMMON="--steps 5000 --batch_size 64 --block_size 256 --n_layer 6 --n_head 6 --d_model 384 --eval_interval 250 --eval_iters 50"

echo
echo "=== ReLU baseline ==="
python scripts/train.py --ffn relu $COMMON --out runs/carc_relu --seed 42

echo
echo "=== GELU baseline ==="
python scripts/train.py --ffn gelu $COMMON --out runs/carc_gelu --seed 42

echo
echo "=== SwiGLU baseline ==="
python scripts/train.py --ffn swiglu $COMMON --out runs/carc_swiglu --seed 42

echo
echo "=== EML-GLU ==="
python scripts/train.py --ffn emlglu $COMMON --out runs/carc_emlglu --seed 42

echo
echo "=== speed bench ==="
python scripts/bench.py --Ts 128 256 512 1024 --batch 16 --out runs/carc_bench.csv

echo "DONE"
