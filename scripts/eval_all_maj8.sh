#!/bin/bash
# ============================================================
# Batch evaluation script for all experiments (maj@8)
# Runs sequentially on a single GPU to avoid OOM.
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/eval_all_maj8.sh
# ============================================================

set -e

BASE_MODEL="Qwen/Qwen3.5-0.8B-Base"
BENCHMARKS="gsm8k math500 math_hard"
NUM_SAMPLES=8
RESULTS_DIR="results"

mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "  Batch maj@${NUM_SAMPLES} evaluation"
echo "========================================"

# ── Base model (no chat template) ──────────────────────────
echo ""
echo "[1/8] Evaluating base model..."
python src/evaluate.py \
    --model "$BASE_MODEL" \
    --benchmarks $BENCHMARKS \
    --num-samples $NUM_SAMPLES \
    --output "$RESULTS_DIR/base_model_maj${NUM_SAMPLES}.json"

# ── Exp 2.1: Full FT ──────────────────────────────────────
echo ""
echo "[2/8] Evaluating exp2_1_full_ft..."
python src/evaluate.py \
    --model outputs/exp2_1_full_ft \
    --chat-template \
    --benchmarks $BENCHMARKS \
    --num-samples $NUM_SAMPLES \
    --output "$RESULTS_DIR/exp2_1_full_ft_maj${NUM_SAMPLES}.json"

# ── Exp 2.2: LoRA ─────────────────────────────────────────
echo ""
echo "[3/8] Evaluating exp2_2_lora..."
python src/evaluate.py \
    --model "$BASE_MODEL" \
    --lora-path outputs/exp2_2_lora \
    --chat-template \
    --benchmarks $BENCHMARKS \
    --num-samples $NUM_SAMPLES \
    --output "$RESULTS_DIR/exp2_2_lora_maj${NUM_SAMPLES}.json"

# ── Exp 2.3: DoRA ─────────────────────────────────────────
echo ""
echo "[4/8] Evaluating exp2_3_dora..."
python src/evaluate.py \
    --model "$BASE_MODEL" \
    --lora-path outputs/exp2_3_dora \
    --chat-template \
    --benchmarks $BENCHMARKS \
    --num-samples $NUM_SAMPLES \
    --output "$RESULTS_DIR/exp2_3_dora_maj${NUM_SAMPLES}.json"

# ── Exp 2.4: PiSSA ────────────────────────────────────────
echo ""
echo "[5/8] Evaluating exp2_4_pissa..."
python src/evaluate.py \
    --model "$BASE_MODEL" \
    --lora-path outputs/exp2_4_pissa \
    --chat-template \
    --benchmarks $BENCHMARKS \
    --num-samples $NUM_SAMPLES \
    --output "$RESULTS_DIR/exp2_4_pissa_maj${NUM_SAMPLES}.json"

# ── Exp 3.1: Curriculum easy→hard ──────────────────────────
echo ""
echo "[6/8] Evaluating exp3_1_easy2hard..."
python src/evaluate.py \
    --model outputs/exp3_1_easy2hard \
    --chat-template \
    --benchmarks $BENCHMARKS \
    --num-samples $NUM_SAMPLES \
    --output "$RESULTS_DIR/exp3_1_easy2hard_maj${NUM_SAMPLES}.json"

# ── Exp 3.2: Curriculum random ─────────────────────────────
echo ""
echo "[7/8] Evaluating exp3_2_random..."
python src/evaluate.py \
    --model outputs/exp3_2_random \
    --chat-template \
    --benchmarks $BENCHMARKS \
    --num-samples $NUM_SAMPLES \
    --output "$RESULTS_DIR/exp3_2_random_maj${NUM_SAMPLES}.json"

# ── Exp 3.3: Curriculum hard→easy ──────────────────────────
echo ""
echo "[8/8] Evaluating exp3_3_hard2easy..."
python src/evaluate.py \
    --model outputs/exp3_3_hard2easy \
    --chat-template \
    --benchmarks $BENCHMARKS \
    --num-samples $NUM_SAMPLES \
    --output "$RESULTS_DIR/exp3_3_hard2easy_maj${NUM_SAMPLES}.json"

echo ""
echo "========================================"
echo "  All evaluations complete!"
echo "  Results saved to $RESULTS_DIR/*_maj${NUM_SAMPLES}.json"
echo "========================================"
echo ""
echo "Run 'python3 src/visualize_results.py' to build comparison tables."
