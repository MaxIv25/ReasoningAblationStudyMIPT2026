#!/bin/bash
# ============================================================
# Exp 3: Curriculum Learning — Easy→Hard / Random / Hard→Easy
# ============================================================
#
# Phase 1: Data preparation (run ONCE, ~15-30 min)
#   bash scripts/run_exp3_curriculum.sh prep
#
# Phase 2: Training (run each on a separate GPU)
#   CUDA_VISIBLE_DEVICES=4 bash scripts/run_exp3_curriculum.sh train easy_to_hard
#   CUDA_VISIBLE_DEVICES=5 bash scripts/run_exp3_curriculum.sh train random
#   CUDA_VISIBLE_DEVICES=6 bash scripts/run_exp3_curriculum.sh train hard_to_easy
#
# Phase 3: Evaluation (run after training completes)
#   CUDA_VISIBLE_DEVICES=4 bash scripts/run_exp3_curriculum.sh eval easy_to_hard
#   CUDA_VISIBLE_DEVICES=5 bash scripts/run_exp3_curriculum.sh eval random
#   CUDA_VISIBLE_DEVICES=6 bash scripts/run_exp3_curriculum.sh eval hard_to_easy
#
# Or run all 3 training jobs in parallel with nohup:
#   bash scripts/run_exp3_curriculum.sh train_all
#
# ============================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
export PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF"

DATA_BASE="data/curriculum"
OUTPUT_BASE="outputs"

# ── Phase 1: Prepare data ───────────────────────────────────
prep_data() {
    echo "=========================================="
    echo "Preparing curriculum datasets..."
    echo "=========================================="
    python scripts/prepare_curriculum_data.py --output-base "$DATA_BASE"
    echo ""
    echo "Data prepared in $DATA_BASE/"
    echo "  easy_to_hard/ — ascending difficulty"
    echo "  random/       — shuffled (control)"
    echo "  hard_to_easy/ — descending difficulty"
}

# ── Phase 2: Train one variant ──────────────────────────────
train_variant() {
    local VARIANT="$1"

    case "$VARIANT" in
        easy_to_hard)
            CONFIG="configs/exp3_1_curriculum_easy2hard.yaml"
            OUTPUT="$OUTPUT_BASE/exp3_1_easy2hard"
            ;;
        random)
            CONFIG="configs/exp3_2_curriculum_random.yaml"
            OUTPUT="$OUTPUT_BASE/exp3_2_random"
            ;;
        hard_to_easy)
            CONFIG="configs/exp3_3_curriculum_hard2easy.yaml"
            OUTPUT="$OUTPUT_BASE/exp3_3_hard2easy"
            ;;
        *)
            echo "Unknown variant: $VARIANT"
            echo "Usage: $0 train {easy_to_hard|random|hard_to_easy}"
            exit 1
            ;;
    esac

    DATA_DIR="$DATA_BASE/$VARIANT"

    if [ ! -d "$DATA_DIR/train" ]; then
        echo "ERROR: Data not found at $DATA_DIR/train"
        echo "Run '$0 prep' first to prepare the data."
        exit 1
    fi

    echo "=========================================="
    echo "Training: $VARIANT"
    echo "  Config:  $CONFIG"
    echo "  Data:    $DATA_DIR"
    echo "  Output:  $OUTPUT"
    echo "  GPU:     ${CUDA_VISIBLE_DEVICES:-all}"
    echo "=========================================="

    python src/train_sft.py \
        --config "$CONFIG" \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT"
}

# ── Phase 2b: Launch all 3 in parallel with nohup ──────────
train_all() {
    echo "Launching all 3 curriculum training jobs..."

    for gpu_variant in "4:easy_to_hard" "5:random" "6:hard_to_easy"; do
        IFS=':' read -r gpu variant <<< "$gpu_variant"
        
        case "$variant" in
            easy_to_hard) config="configs/exp3_1_curriculum_easy2hard.yaml"; output="$OUTPUT_BASE/exp3_1_easy2hard" ;;
            random)       config="configs/exp3_2_curriculum_random.yaml";    output="$OUTPUT_BASE/exp3_2_random"   ;;
            hard_to_easy) config="configs/exp3_3_curriculum_hard2easy.yaml"; output="$OUTPUT_BASE/exp3_3_hard2easy" ;;
        esac

        log_file="train_exp3_${variant}.log"

        echo "  GPU $gpu → $variant (log: $log_file)"
        CUDA_VISIBLE_DEVICES=$gpu \
        PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF" \
        nohup python src/train_sft.py \
            --config "$config" \
            --data-dir "$DATA_BASE/$variant" \
            --output-dir "$output" \
            > "$log_file" 2>&1 &
    done

    echo ""
    echo "All 3 jobs launched! Monitor with:"
    echo "  tail -f train_exp3_easy_to_hard.log"
    echo "  tail -f train_exp3_random.log"
    echo "  tail -f train_exp3_hard_to_easy.log"
    echo ""
    echo "  nvidia-smi  # check GPU usage"
}

# ── Phase 3: Evaluate one variant ───────────────────────────
eval_variant() {
    local VARIANT="$1"

    case "$VARIANT" in
        easy_to_hard) OUTPUT="$OUTPUT_BASE/exp3_1_easy2hard"; RUN_NAME="exp3_1_easy2hard" ;;
        random)       OUTPUT="$OUTPUT_BASE/exp3_2_random";    RUN_NAME="exp3_2_random"    ;;
        hard_to_easy) OUTPUT="$OUTPUT_BASE/exp3_3_hard2easy"; RUN_NAME="exp3_3_hard2easy" ;;
        *)
            echo "Unknown variant: $VARIANT"
            exit 1
            ;;
    esac

    if [ ! -d "$OUTPUT" ]; then
        echo "ERROR: Model not found at $OUTPUT"
        exit 1
    fi

    echo "=========================================="
    echo "Evaluating: $VARIANT"
    echo "  Model:  $OUTPUT"
    echo "  GPU:    ${CUDA_VISIBLE_DEVICES:-all}"
    echo "=========================================="

    # Full FT: model is saved directly (not a LoRA adapter)
    python src/evaluate.py \
        --model "$OUTPUT" \
        --benchmarks gsm8k math500 math_hard \
        --chat-template \
        --run-name "$RUN_NAME" \
        --output "results/${RUN_NAME}_results.json"
}

# ── Main ────────────────────────────────────────────────────
case "${1:-help}" in
    prep)
        prep_data
        ;;
    train)
        train_variant "${2:?Usage: $0 train {easy_to_hard|random|hard_to_easy}}"
        ;;
    train_all)
        train_all
        ;;
    eval)
        eval_variant "${2:?Usage: $0 eval {easy_to_hard|random|hard_to_easy}}"
        ;;
    *)
        echo "Usage: $0 {prep|train|train_all|eval} [variant]"
        echo ""
        echo "Commands:"
        echo "  prep                     — Prepare 3 curriculum datasets"
        echo "  train {variant}          — Train one variant"
        echo "  train_all                — Launch all 3 on GPUs 4,5,6 with nohup"
        echo "  eval {variant}           — Evaluate one variant"
        echo ""
        echo "Variants: easy_to_hard, random, hard_to_easy"
        exit 1
        ;;
esac
