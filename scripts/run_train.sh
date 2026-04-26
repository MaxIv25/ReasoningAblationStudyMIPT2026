#!/bin/bash
# ============================================================
# Universal training launcher
# Auto-detects GPU count: 1 GPU → python, 2+ GPU → torchrun DDP
# ============================================================
#
# Usage:
#   bash scripts/run_train.sh --config configs/exp1_1_deepseek_traces.yaml \
#       --data-dir data/openr1_20k --output-dir outputs/exp1_1
#
# Force single GPU:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_train.sh ...
#
# Force 2 GPUs:
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_train.sh ...
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Count GPUs: respect CUDA_VISIBLE_DEVICES if set
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # Count comma-separated GPU IDs
    NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
fi

echo "Using $NGPU GPU(s) (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all})"

if [ "$NGPU" -gt 1 ]; then
    echo "Launching with torchrun (DDP, $NGPU GPUs)..."
    torchrun --nproc_per_node=$NGPU \
        "$PROJECT_ROOT/src/train_sft.py" "$@"
else
    echo "Launching single-GPU training..."
    python "$PROJECT_ROOT/src/train_sft.py" "$@"
fi
