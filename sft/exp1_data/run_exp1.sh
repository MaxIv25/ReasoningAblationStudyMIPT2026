#!/bin/bash
# ============================================================
# Phase 1: Data source ablation
# ============================================================
# Exp 1.1: SFT on DeepSeek-R1 traces (OpenR1-Math-220K)
# Exp 1.2: SFT on Qwen3.5-35B-A3B traces (generated)
# Exp 1.3: SFT on R1 traces + KL-constraint with Qwen3.5-35B-A3B
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_DIR="sft/exp1_data/results"
DATA_DIR="data"
mkdir -p "$RESULTS_DIR" "$DATA_DIR"

echo "=========================================="
echo "PHASE 1: DATA SOURCE ABLATION"
echo "=========================================="

# ── Step 0: Prepare data ──────────────────────────────────────

echo "[Step 0] Preparing OpenR1-Math data (20K samples)..."
python src/data_utils.py \
    --config configs/base.yaml \
    --output "$DATA_DIR/openr1_20k" \
    --curriculum random

# ── Step 1: Generate Qwen traces ──────────────────────────────

echo "[Step 1] Generating traces from Qwen3.5-35B-A3B..."
python src/generate_traces.py \
    --teacher "Qwen/Qwen3.5-35B-A3B" \
    --output "$DATA_DIR/qwen35_traces.json" \
    --num-samples 20000 \
    --config configs/base.yaml

# ── Exp 1.1: SFT on DeepSeek-R1 traces ───────────────────────

echo "[Exp 1.1] SFT on DeepSeek-R1 traces..."
python src/train_sft.py \
    --config configs/exp1_1_deepseek_traces.yaml \
    --data-dir "$DATA_DIR/openr1_20k" \
    --output-dir "outputs/exp1_1_deepseek_traces"

echo "[Exp 1.1] Evaluating..."
python src/evaluate.py \
    --model "outputs/exp1_1_deepseek_traces" \
    --output "$RESULTS_DIR/exp1_1_results.json" \
    --chat-template

# ── Exp 1.2: SFT on Qwen traces ──────────────────────────────

echo "[Exp 1.2] SFT on Qwen3.5-35B-A3B traces..."
python src/train_sft.py \
    --config configs/exp1_2_qwen_traces.yaml \
    --data-dir "$DATA_DIR/qwen35_traces_hf" \
    --output-dir "outputs/exp1_2_qwen_traces"

echo "[Exp 1.2] Evaluating..."
python src/evaluate.py \
    --model "outputs/exp1_2_qwen_traces" \
    --output "$RESULTS_DIR/exp1_2_results.json" \
    --chat-template

# ── Exp 1.3: SFT + KL-constraint ─────────────────────────────

echo "[Exp 1.3] SFT + KL-constraint with Qwen teacher..."
python src/train_sft.py \
    --config configs/exp1_3_kl_constraint.yaml \
    --data-dir "$DATA_DIR/openr1_20k" \
    --output-dir "outputs/exp1_3_kl_constraint"

echo "[Exp 1.3] Evaluating..."
python src/evaluate.py \
    --model "outputs/exp1_3_kl_constraint" \
    --output "$RESULTS_DIR/exp1_3_results.json" \
    --chat-template

echo "=========================================="
echo "PHASE 1 COMPLETE"
echo "=========================================="
