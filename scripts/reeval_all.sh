#!/bin/bash
# =============================================================================
# Re-evaluate ALL models with fixed LaTeX matching
# Supports parallel execution on multiple GPUs
#
# Usage:
#   bash scripts/reeval_all.sh                    # auto-detect all GPUs
#   bash scripts/reeval_all.sh 0,1,2,3            # use specific GPUs
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/reeval_all.sh  # also works
# =============================================================================

# --- Configuration ---
BENCHMARKS="gsm8k math500 math_hard"
COMMON="--benchmarks $BENCHMARKS --chat-template --temperature 0.6 --top-p 0.95 --top-k 20"
LOGDIR="logs/reeval"
mkdir -p "$LOGDIR" results

# --- GPU detection ---
if [ -n "$1" ]; then
    IFS=',' read -ra GPUS <<< "$1"
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
    NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
    GPUS=($(seq 0 $((NUM_GPUS - 1))))
fi

NUM_GPUS=${#GPUS[@]}
echo "Using $NUM_GPUS GPUs: ${GPUS[*]}"

# --- Job queue ---
# Format: "MODEL_PATH|OUTPUT_PATH|NUM_SAMPLES|EXTRA_ARGS|DESCRIPTION"
# Priority order: P1 (critical) → P2 (important) → P3 (nice-to-have)
JOBS=()

# ──────────────────────────────────────────────────────────────
# P1: Full FT baseline + RL models (most critical)
# ──────────────────────────────────────────────────────────────

# SFT baseline — all comparisons depend on this
JOBS+=("outputs/exp2_1_full_ft|results/exp2_1_full_ft_maj8.json|8||P1: SFT Full FT baseline (maj@8)")
JOBS+=("outputs/exp2_1_full_ft|results/exp2_1_full_ft_pass1.json|1||P1: SFT Full FT baseline (pass@1)")

# RL models (may not exist on this machine — will skip gracefully)
JOBS+=("outputs/exp5_2_grpo_sft_easy|results/exp5_2_grpo_sft_easy_maj8.json|8||P1: GRPO (maj@8)")
JOBS+=("outputs/exp5_2_grpo_sft_easy|results/exp5_2_grpo_sft_easy_pass1.json|1||P1: GRPO (pass@1)")
JOBS+=("outputs/exp5_3_dapo_sft_easy|results/exp5_3_dapo_sft_easy_maj8.json|8||P1: DAPO (maj@8)")
JOBS+=("outputs/exp5_3_dapo_sft_easy|results/exp5_3_dapo_sft_easy_pass1.json|1||P1: DAPO (pass@1)")
JOBS+=("outputs/exp5_4_dr_grpo_sft_easy|results/exp5_4_dr_grpo_sft_easy_maj8.json|8||P1: DR-GRPO (maj@8)")
JOBS+=("outputs/exp5_4_dr_grpo_sft_easy|results/exp5_4_dr_grpo_sft_easy_pass1.json|1||P1: DR-GRPO (pass@1)")
JOBS+=("outputs/exp5_5_prime_sft_easy|results/exp5_5_prime_sft_easy_maj8.json|8||P1: PRIME (maj@8)")
JOBS+=("outputs/exp5_5_prime_sft_easy|results/exp5_5_prime_sft_easy_pass1.json|1||P1: PRIME (pass@1)")

# ──────────────────────────────────────────────────────────────
# P2: Data ordering experiments
# ──────────────────────────────────────────────────────────────

JOBS+=("outputs/exp3_1_easy2hard|results/exp3_1_easy2hard_maj8.json|8||P2: Easy→Hard (maj@8)")
JOBS+=("outputs/exp3_1_easy2hard|results/exp3_1_easy2hard_pass1.json|1||P2: Easy→Hard (pass@1)")
JOBS+=("outputs/exp3_2_random|results/exp3_2_random_maj8.json|8||P2: Random order (maj@8)")
JOBS+=("outputs/exp3_2_random|results/exp3_2_random_pass1.json|1||P2: Random order (pass@1)")
JOBS+=("outputs/exp3_3_hard2easy|results/exp3_3_hard2easy_maj8.json|8||P2: Hard→Easy (maj@8)")
JOBS+=("outputs/exp3_3_hard2easy|results/exp3_3_hard2easy_pass1.json|1||P2: Hard→Easy (pass@1)")

# ──────────────────────────────────────────────────────────────
# P3: LoRA variants + prompt loss + no decay
# ──────────────────────────────────────────────────────────────

JOBS+=("outputs/exp2_2_lora|results/exp2_2_lora_maj8.json|8||P3: LoRA (maj@8)")
JOBS+=("outputs/exp2_2_lora|results/exp2_2_lora_pass1.json|1||P3: LoRA (pass@1)")
JOBS+=("outputs/exp2_3_dora|results/exp2_3_dora_maj8.json|8||P3: DoRA (maj@8)")
JOBS+=("outputs/exp2_3_dora|results/exp2_3_dora_pass1.json|1||P3: DoRA (pass@1)")
JOBS+=("outputs/exp2_4_pissa|results/exp2_4_pissa_maj8.json|8||P3: PiSSA (maj@8)")
JOBS+=("outputs/exp2_4_pissa|results/exp2_4_pissa_pass1.json|1||P3: PiSSA (pass@1)")
JOBS+=("outputs/exp4_1_no_weight_decay|results/exp4_1_no_weight_decay_maj8.json|8||P3: No WD (maj@8)")
JOBS+=("outputs/exp4_1_no_weight_decay|results/exp4_1_no_weight_decay_pass1.json|1||P3: No WD (pass@1)")
JOBS+=("outputs/exp4_2_prompt_loss|results/exp4_2_prompt_loss_maj8.json|8||P3: Prompt loss (maj@8)")
JOBS+=("outputs/exp4_2_prompt_loss|results/exp4_2_prompt_loss_pass1.json|1||P3: Prompt loss (pass@1)")

# --- Worker function ---
run_eval() {
    local GPU_ID=$1
    local MODEL=$2
    local OUTPUT=$3
    local NSAMPLES=$4
    local EXTRA=$5
    local DESC=$6

    local LOGFILE="$LOGDIR/$(basename $OUTPUT .json).log"

    # Skip if model doesn't exist
    if [ ! -d "$MODEL" ]; then
        echo "[GPU $GPU_ID] SKIP: $DESC — model not found: $MODEL"
        return 0
    fi

    echo "[GPU $GPU_ID] START: $DESC"
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m src.evaluate \
        --model "$MODEL" \
        --output "$OUTPUT" \
        --num-samples "$NSAMPLES" \
        $EXTRA \
        $COMMON \
        > "$LOGFILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "[GPU $GPU_ID] DONE:  $DESC ✓"
    else
        echo "[GPU $GPU_ID] FAIL:  $DESC ✗ (see $LOGFILE)"
    fi
}

# --- Dispatch jobs across GPUs ---
echo ""
echo "============================================"
echo "Starting ${#JOBS[@]} evaluation jobs on $NUM_GPUS GPUs"
echo "============================================"
echo ""

# Track PIDs per GPU slot
declare -A GPU_PIDS

JOB_IDX=0
for JOB in "${JOBS[@]}"; do
    IFS='|' read -r MODEL OUTPUT NSAMPLES EXTRA DESC <<< "$JOB"

    # Wait for a free GPU
    while true; do
        for i in "${!GPUS[@]}"; do
            GPU=${GPUS[$i]}
            PID=${GPU_PIDS[$i]:-0}
            # If no PID or PID finished, this GPU is free
            if [ "$PID" -eq 0 ] || ! kill -0 "$PID" 2>/dev/null; then
                GPU_PIDS[$i]=0
                # Launch job on this GPU
                run_eval "$GPU" "$MODEL" "$OUTPUT" "$NSAMPLES" "$EXTRA" "$DESC" &
                GPU_PIDS[$i]=$!
                break 2  # break both loops
            fi
        done
        # All GPUs busy, wait a bit
        sleep 5
    done
done

# Wait for all remaining jobs
echo ""
echo "Waiting for remaining jobs to finish..."
wait

echo ""
echo "============================================"
echo "All evaluations complete!"
echo "============================================"

# --- Print results table ---
python3 -c "
import json, glob, os

print('\n' + '='*100)
print('RESULTS COMPARISON (fixed LaTeX matching)')
print('='*100)

# Collect all result files
files = {}
for f in sorted(glob.glob('results/*.json')):
    name = os.path.basename(f).replace('.json', '')
    files[name] = f

# Group by experiment
experiments = {}
for name, path in files.items():
    # Strip _maj8 or _pass1 suffix for grouping
    base = name.replace('_maj8', '').replace('_pass1', '')
    if base not in experiments:
        experiments[base] = {}
    if '_maj8' in name:
        experiments[base]['maj8'] = path
    elif '_pass1' in name:
        experiments[base]['pass1'] = path
    else:
        experiments[base]['other'] = path

# Print table
print(f'\n{\"Experiment\":<30} {\"Metric\":<7} {\"GSM8K\":>7} {\"MATH500\":>8} {\"M_Hard\":>8} {\"Fmt_G\":>7} {\"Fmt_M\":>7} {\"Fmt_H\":>7}')
print('-' * 90)

for exp in sorted(experiments.keys()):
    data = experiments[exp]
    for metric_key in ['maj8', 'pass1', 'other']:
        if metric_key not in data:
            continue
        try:
            d = json.load(open(data[metric_key]))
            g = d.get('gsm8k', {})
            m = d.get('math500', {})
            h = d.get('math_hard', {})
            label = metric_key if metric_key != 'other' else 'p@1'
            print(f'{exp:<30} {label:<7} {g.get(\"accuracy\",0):>7.2f} {m.get(\"accuracy\",0):>8.2f} {h.get(\"accuracy\",0):>8.2f} {g.get(\"format_compliance\",0):>7.1f} {m.get(\"format_compliance\",0):>7.1f} {h.get(\"format_compliance\",0):>7.1f}')
        except Exception as e:
            print(f'{exp:<30} {metric_key:<7} ERROR: {e}')
"
