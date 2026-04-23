#!/bin/bash
# ============================================================
# Phase 2: Training method ablation
# ============================================================
# Exp 2.1: Full Fine-Tuning
# Exp 2.2: LoRA r=64
# Exp 2.3: DoRA r=64
# Exp 2.4: PiSSA r=64
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_DIR="sft/exp2_method/results"
DATA_DIR="data/best_phase1"  # Use best data from Phase 1
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "PHASE 2: TRAINING METHOD ABLATION"
echo "=========================================="

for METHOD in full_ft lora dora pissa; do
    EXP_NUM="2_$(echo $METHOD | tr '_' '-')"
    echo "[Exp $EXP_NUM] Training with $METHOD..."
    
    python src/train_sft.py \
        --config "configs/exp2_${METHOD//-/_}.yaml" \
        --data-dir "$DATA_DIR" \
        --output-dir "outputs/exp2_${METHOD}" \
        --method "$METHOD"
    
    echo "[Exp $EXP_NUM] Evaluating..."
    python src/evaluate.py \
        --model "outputs/exp2_${METHOD}" \
        --output "$RESULTS_DIR/exp2_${METHOD}_results.json" \
        --chat-template
done

echo "=========================================="
echo "PHASE 2 COMPLETE"
echo "=========================================="
