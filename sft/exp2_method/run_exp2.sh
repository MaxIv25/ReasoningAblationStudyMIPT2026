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

declare -A EXP_CONFIGS=(
    [full_ft]="configs/exp2_1_full_ft.yaml"
    [lora]="configs/exp2_2_lora.yaml"
    [dora]="configs/exp2_3_dora.yaml"
    [pissa]="configs/exp2_4_pissa.yaml"
)

for METHOD in full_ft lora dora pissa; do
    CONFIG="${EXP_CONFIGS[$METHOD]}"
    echo "[Exp 2 — $METHOD] Training..."
    
    python src/train_sft.py \
        --config "$CONFIG" \
        --data-dir "$DATA_DIR" \
        --output-dir "outputs/exp2_${METHOD}" \
        --method "$METHOD"
    
    echo "[Exp 2 — $METHOD] Evaluating..."
    python src/evaluate.py \
        --model "outputs/exp2_${METHOD}" \
        --output "$RESULTS_DIR/exp2_${METHOD}_results.json" \
        --chat-template
done

echo "=========================================="
echo "PHASE 2 COMPLETE"
echo "=========================================="
