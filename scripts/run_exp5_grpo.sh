#!/bin/bash
# ============================================================
# GRPO Ablation Experiments — Phase 5
#
# Runs 4 experiments sequentially on a single H200 GPU:
#   exp5_1: Vanilla GRPO on base model (control)
#   exp5_2: Vanilla GRPO on SFT model
#   exp5_3: DAPO on SFT model
#   exp5_4: Dr. GRPO on SFT model
#
# Prerequisites:
#   1. Prepare data:
#      python scripts/prepare_grpo_data.py --output data/grpo_20k
#   2. Ensure SFT model exists:
#      ls outputs/exp2_1_full_ft/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_exp5_grpo.sh
#
# To run a single experiment:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_exp5_grpo.sh exp5_2
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data/grpo_20k"
RESULTS_DIR="${PROJECT_ROOT}/results"

# Check data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: GRPO dataset not found at $DATA_DIR"
    echo "Run: python scripts/prepare_grpo_data.py --output $DATA_DIR"
    exit 1
fi

# Optional: run only specific experiment
RUN_ONLY="${1:-all}"

mkdir -p "$RESULTS_DIR"

# ── Helper function ────────────────────────────────────────
run_experiment() {
    local exp_name="$1"
    local config="$2"
    local output_dir="$3"

    echo ""
    echo "============================================"
    echo "  Training: $exp_name"
    echo "  Config:   $config"
    echo "  Output:   $output_dir"
    echo "============================================"
    echo ""

    python "${PROJECT_ROOT}/src/train_grpo.py" \
        --config "$config" \
        --data-dir "$DATA_DIR" \
        --output-dir "$output_dir" \
        2>&1 | tee "${PROJECT_ROOT}/train_${exp_name}.log"

    echo ""
    echo "  Training complete: $exp_name"
    echo ""
}

run_eval() {
    local exp_name="$1"
    local model_path="$2"

    echo ""
    echo "  Evaluating: $exp_name..."
    echo ""

    python "${PROJECT_ROOT}/src/evaluate.py" \
        --model "$model_path" \
        --chat-template \
        --benchmarks gsm8k math500 math_hard \
        --gpu-mem 0.9 \
        --output "${RESULTS_DIR}/${exp_name}.json"

    echo "  Eval saved: ${RESULTS_DIR}/${exp_name}.json"
}

# ── Exp 5.1: Vanilla GRPO on base model ───────────────────
if [ "$RUN_ONLY" = "all" ] || [ "$RUN_ONLY" = "exp5_1" ]; then
    run_experiment "exp5_1_grpo_base" \
        "${PROJECT_ROOT}/configs/exp5_1_grpo_base.yaml" \
        "${PROJECT_ROOT}/outputs/exp5_1_grpo_base"

    # Base model: no chat template for eval (it wasn't SFT-trained)
    echo "  Evaluating: exp5_1 (no chat template for base)..."
    python "${PROJECT_ROOT}/src/evaluate.py" \
        --model "${PROJECT_ROOT}/outputs/exp5_1_grpo_base" \
        --benchmarks gsm8k math500 math_hard \
        --gpu-mem 0.9 \
        --output "${RESULTS_DIR}/exp5_1_grpo_base.json"
fi

# ── Exp 5.2: Vanilla GRPO on SFT model ────────────────────
if [ "$RUN_ONLY" = "all" ] || [ "$RUN_ONLY" = "exp5_2" ]; then
    # Check SFT model exists
    if [ ! -d "${PROJECT_ROOT}/outputs/exp2_1_full_ft" ]; then
        echo "ERROR: SFT model not found at outputs/exp2_1_full_ft"
        echo "Run SFT training first (exp2_1)"
        exit 1
    fi

    run_experiment "exp5_2_grpo_sft" \
        "${PROJECT_ROOT}/configs/exp5_2_grpo_sft.yaml" \
        "${PROJECT_ROOT}/outputs/exp5_2_grpo_sft"

    run_eval "exp5_2_grpo_sft" "${PROJECT_ROOT}/outputs/exp5_2_grpo_sft"
fi

# ── Exp 5.3: DAPO on SFT model ────────────────────────────
if [ "$RUN_ONLY" = "all" ] || [ "$RUN_ONLY" = "exp5_3" ]; then
    run_experiment "exp5_3_dapo_sft" \
        "${PROJECT_ROOT}/configs/exp5_3_dapo_sft.yaml" \
        "${PROJECT_ROOT}/outputs/exp5_3_dapo_sft"

    run_eval "exp5_3_dapo_sft" "${PROJECT_ROOT}/outputs/exp5_3_dapo_sft"
fi

# ── Exp 5.4: Dr. GRPO on SFT model ────────────────────────
if [ "$RUN_ONLY" = "all" ] || [ "$RUN_ONLY" = "exp5_4" ]; then
    run_experiment "exp5_4_dr_grpo_sft" \
        "${PROJECT_ROOT}/configs/exp5_4_dr_grpo_sft.yaml" \
        "${PROJECT_ROOT}/outputs/exp5_4_dr_grpo_sft"

    run_eval "exp5_4_dr_grpo_sft" "${PROJECT_ROOT}/outputs/exp5_4_dr_grpo_sft"
fi

echo ""
echo "============================================"
echo "  All GRPO experiments complete!"
echo "  Results: ${RESULTS_DIR}/exp5_*.json"
echo "============================================"
echo ""
echo "TensorBoard:"
echo "  tensorboard --logdir ${PROJECT_ROOT}/outputs/"
echo ""
echo "Key metrics to compare in TensorBoard:"
echo "  - reward (mean accuracy reward)"
echo "  - reward/accuracy_reward/mean"
echo "  - reward/format_reward/mean"
echo "  - completions/mean_length"
echo "  - entropy"
echo "  - clip_ratio/region_mean"
echo "  - frac_reward_zero_std"
echo "  - Custom: accuracy, format_compliance"
