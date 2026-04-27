#!/bin/bash
#SBATCH --job-name=eml_sweep
#SBATCH --partition=nlp_hiprio
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --account=jessetho_1732
#SBATCH --output=/project2/jessetho_1732/zizhaoh/eml-activation/logs/%x_%j.log
#SBATCH --error=/project2/jessetho_1732/zizhaoh/eml-activation/logs/%x_%j.log

set -eo pipefail
module purge
module load gcc/13.3.0 cuda/12.6.3
export CUDA_HOME=$CUDA_ROOT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_CACHE_DIR=/scratch1/$USER/.cache/uv

source /scratch1/zizhaoh/envs/eml-activation/bin/activate
cd /project2/jessetho_1732/zizhaoh/eml-activation

echo "=== node info ==="
hostname; nvidia-smi -L; date

OUTROOT=runs_sweep
mkdir -p "$OUTROOT"

# ----- model-size sweep -----
# Each entry: name d_model n_layer n_head
SIZES=(
    "small  256 4 4"
    "medium 384 6 6"
    "large  512 8 8"
)
VARIANTS=(relu gelu swiglu emlglu)

run_one() {
    local tag=$1; shift
    local args="$@"
    local out="$OUTROOT/$tag"
    if [ -f "$out/loss.csv" ]; then
        echo "[skip] $tag (already exists)"
        return
    fi
    echo
    echo ">>> $tag"
    python scripts/train.py $args --out "$out"
}

# Stage A: model-size sweep, seed=42
echo
echo "########### STAGE A: model size sweep ###########"
for SIZE in "${SIZES[@]}"; do
    read -r name dm nl nh <<<"$SIZE"
    for ffn in "${VARIANTS[@]}"; do
        run_one "${name}_${ffn}_s42" \
            --ffn $ffn --steps 5000 --batch_size 64 --block_size 256 \
            --n_layer $nl --n_head $nh --d_model $dm \
            --eval_interval 250 --eval_iters 50 \
            --weight_decay 0.1 --seed 42
    done
done

# Stage B: seed sweep at medium (11M), seeds 123 and 456
echo
echo "########### STAGE B: seed sweep at medium ###########"
for seed in 123 456; do
    for ffn in "${VARIANTS[@]}"; do
        run_one "medium_${ffn}_s${seed}" \
            --ffn $ffn --steps 5000 --batch_size 64 --block_size 256 \
            --n_layer 6 --n_head 6 --d_model 384 \
            --eval_interval 250 --eval_iters 50 \
            --weight_decay 0.1 --seed $seed
    done
done

# Stage C: weight-decay ablation at medium (11M), seed=42
echo
echo "########### STAGE C: weight-decay ablation at medium ###########"
for wd in 0.5 1.0; do
    for ffn in "${VARIANTS[@]}"; do
        run_one "medium_${ffn}_s42_wd${wd}" \
            --ffn $ffn --steps 5000 --batch_size 64 --block_size 256 \
            --n_layer 6 --n_head 6 --d_model 384 \
            --eval_interval 250 --eval_iters 50 \
            --weight_decay $wd --seed 42
    done
done

# Stage D: speed bench (already in runs_carc, but redo for completeness)
echo
echo "########### STAGE D: speed bench ###########"
python scripts/bench.py --Ts 128 256 512 1024 2048 --batch 16 --out "$OUTROOT/bench.csv"

echo
echo "DONE  $(date)"
