#!/bin/bash
# =============================================================================
# Re-evaluate ALL models with fixed LaTeX matching
# Supports parallel execution on multiple GPUs
#
# Usage:
#   bash scripts/reeval_all.sh 0,1,2,3     # use GPUs 0,1,2,3
#   bash scripts/reeval_all.sh 2,5          # use GPUs 2 and 5
#   bash scripts/reeval_all.sh 0            # single GPU
#   bash scripts/reeval_all.sh              # auto-detect all GPUs
# =============================================================================

# --- Configuration ---
BENCHMARKS="gsm8k math500 math_hard"
BASE_MODEL="Qwen/Qwen3.5-0.8B-Base"
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
# Format: "TYPE|MODEL_PATH|OUTPUT_PATH|NUM_SAMPLES|DESCRIPTION"
# TYPE: full = full weights, lora = LoRA adapter, dora = DoRA (needs --merge-lora)
# Priority order: P1 → P2 → P3
JOBS=()

# P1: Full FT baseline + RL models
JOBS+=("full|outputs/exp2_1_full_ft|results/exp2_1_full_ft_maj8.json|8|P1: SFT Full FT (maj@8)")
JOBS+=("full|outputs/exp2_1_full_ft|results/exp2_1_full_ft_pass1.json|1|P1: SFT Full FT (pass@1)")
JOBS+=("full|outputs/exp5_2_grpo_sft_easy|results/exp5_2_grpo_sft_easy_maj8.json|8|P1: GRPO (maj@8)")
JOBS+=("full|outputs/exp5_2_grpo_sft_easy|results/exp5_2_grpo_sft_easy_pass1.json|1|P1: GRPO (pass@1)")
JOBS+=("full|outputs/exp5_3_dapo_sft_easy|results/exp5_3_dapo_sft_easy_maj8.json|8|P1: DAPO (maj@8)")
JOBS+=("full|outputs/exp5_3_dapo_sft_easy|results/exp5_3_dapo_sft_easy_pass1.json|1|P1: DAPO (pass@1)")
JOBS+=("full|outputs/exp5_4_dr_grpo_sft_easy|results/exp5_4_dr_grpo_sft_easy_maj8.json|8|P1: DR-GRPO (maj@8)")
JOBS+=("full|outputs/exp5_4_dr_grpo_sft_easy|results/exp5_4_dr_grpo_sft_easy_pass1.json|1|P1: DR-GRPO (pass@1)")
JOBS+=("full|outputs/exp5_5_prime_sft_easy|results/exp5_5_prime_sft_easy_maj8.json|8|P1: PRIME (maj@8)")
JOBS+=("full|outputs/exp5_5_prime_sft_easy|results/exp5_5_prime_sft_easy_pass1.json|1|P1: PRIME (pass@1)")

# P2: Data ordering experiments (all full weights)
JOBS+=("full|outputs/exp3_1_easy2hard|results/exp3_1_easy2hard_maj8.json|8|P2: Easy→Hard (maj@8)")
JOBS+=("full|outputs/exp3_1_easy2hard|results/exp3_1_easy2hard_pass1.json|1|P2: Easy→Hard (pass@1)")
JOBS+=("full|outputs/exp3_2_random|results/exp3_2_random_maj8.json|8|P2: Random (maj@8)")
JOBS+=("full|outputs/exp3_2_random|results/exp3_2_random_pass1.json|1|P2: Random (pass@1)")
JOBS+=("full|outputs/exp3_3_hard2easy|results/exp3_3_hard2easy_maj8.json|8|P2: Hard→Easy (maj@8)")
JOBS+=("full|outputs/exp3_3_hard2easy|results/exp3_3_hard2easy_pass1.json|1|P2: Hard→Easy (pass@1)")

# P3: LoRA / DoRA / PiSSA / training tweaks
JOBS+=("lora|outputs/exp2_2_lora|results/exp2_2_lora_maj8.json|8|P3: LoRA (maj@8)")
JOBS+=("lora|outputs/exp2_2_lora|results/exp2_2_lora_pass1.json|1|P3: LoRA (pass@1)")
JOBS+=("dora|outputs/exp2_3_dora|results/exp2_3_dora_maj8.json|8|P3: DoRA (maj@8)")
JOBS+=("dora|outputs/exp2_3_dora|results/exp2_3_dora_pass1.json|1|P3: DoRA (pass@1)")
JOBS+=("lora|outputs/exp2_4_pissa|results/exp2_4_pissa_maj8.json|8|P3: PiSSA (maj@8)")
JOBS+=("lora|outputs/exp2_4_pissa|results/exp2_4_pissa_pass1.json|1|P3: PiSSA (pass@1)")
JOBS+=("full|outputs/exp4_1_no_weight_decay|results/exp4_1_no_weight_decay_maj8.json|8|P3: No WD (maj@8)")
JOBS+=("full|outputs/exp4_1_no_weight_decay|results/exp4_1_no_weight_decay_pass1.json|1|P3: No WD (pass@1)")
JOBS+=("full|outputs/exp4_2_prompt_loss|results/exp4_2_prompt_loss_maj8.json|8|P3: Prompt loss (maj@8)")
JOBS+=("full|outputs/exp4_2_prompt_loss|results/exp4_2_prompt_loss_pass1.json|1|P3: Prompt loss (pass@1)")

# --- Worker function ---
run_eval() {
    local GPU_ID=$1
    local TYPE=$2
    local MODEL=$3
    local OUTPUT=$4
    local NSAMPLES=$5
    local DESC=$6
    local LOGFILE="$LOGDIR/$(basename $OUTPUT .json).log"

    # Skip if model doesn't exist
    if [ ! -d "$MODEL" ]; then
        echo "[GPU $GPU_ID] SKIP: $DESC — not found: $MODEL"
        return 0
    fi

    # Auto-detect: if adapter_config.json exists, it's a PEFT adapter
    # even if marked as "full" (safety check)
    local EVAL_ARGS=""
    if [ "$TYPE" = "lora" ] || ([ -f "$MODEL/adapter_config.json" ] && [ "$TYPE" != "dora" ]); then
        # LoRA/PiSSA: load base model + adapter
        EVAL_ARGS="--model $BASE_MODEL --lora-path $MODEL"
    elif [ "$TYPE" = "dora" ]; then
        # DoRA: must merge adapter into base (vLLM doesn't support DoRA natively)
        if [ -f "$MODEL/adapter_config.json" ]; then
            EVAL_ARGS="--model $BASE_MODEL --lora-path $MODEL --merge-lora"
        else
            # Already merged to full weights
            EVAL_ARGS="--model $MODEL"
        fi
    else
        # Full weights: direct load
        EVAL_ARGS="--model $MODEL"
    fi

    echo "[GPU $GPU_ID] START: $DESC"
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m src.evaluate \
        $EVAL_ARGS \
        --output "$OUTPUT" \
        --num-samples "$NSAMPLES" \
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
echo "Jobs will continue even if individual evals fail"
echo "============================================"
echo ""

declare -A GPU_PIDS

for JOB in "${JOBS[@]}"; do
    IFS='|' read -r TYPE MODEL OUTPUT NSAMPLES DESC <<< "$JOB"

    # Wait for a free GPU
    while true; do
        for i in "${!GPUS[@]}"; do
            GPU=${GPUS[$i]}
            PID=${GPU_PIDS[$i]:-0}
            if [ "$PID" -eq 0 ] || ! kill -0 "$PID" 2>/dev/null; then
                GPU_PIDS[$i]=0
                run_eval "$GPU" "$TYPE" "$MODEL" "$OUTPUT" "$NSAMPLES" "$DESC" &
                GPU_PIDS[$i]=$!
                break 2
            fi
        done
        sleep 5
    done
done

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

files = {}
for f in sorted(glob.glob('results/*.json')):
    name = os.path.basename(f).replace('.json', '')
    files[name] = f

experiments = {}
for name, path in files.items():
    base = name.replace('_maj8', '').replace('_pass1', '')
    if base not in experiments:
        experiments[base] = {}
    if '_maj8' in name:
        experiments[base]['maj8'] = path
    elif '_pass1' in name:
        experiments[base]['pass1'] = path
    else:
        experiments[base]['other'] = path

print(f'\n{\"Experiment\":<35} {\"Metric\":<7} {\"GSM8K\":>7} {\"MATH500\":>8} {\"M_Hard\":>8} {\"Fmt_G\":>7} {\"Fmt_M\":>7} {\"Fmt_H\":>7}')
print('-' * 90)
for exp in sorted(experiments.keys()):
    for mk in ['maj8', 'pass1', 'other']:
        if mk not in experiments[exp]: continue
        try:
            d = json.load(open(experiments[exp][mk]))
            g, m, h = d.get('gsm8k',{}), d.get('math500',{}), d.get('math_hard',{})
            label = mk if mk != 'other' else 'p@1'
            print(f'{exp:<35} {label:<7} {g.get(\"accuracy\",0):>7.2f} {m.get(\"accuracy\",0):>8.2f} {h.get(\"accuracy\",0):>8.2f} {g.get(\"format_compliance\",0):>7.1f} {m.get(\"format_compliance\",0):>7.1f} {h.get(\"format_compliance\",0):>7.1f}')
        except Exception as e:
            print(f'{exp:<35} {mk:<7} ERROR: {e}')
"
