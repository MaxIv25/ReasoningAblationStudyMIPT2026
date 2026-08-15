#!/usr/bin/env bash
set -euo pipefail

# Reproducible same-stack evaluation of completed Qwen3.5-0.8B experiments.
# Launch with CUDA_VISIBLE_DEVICES set explicitly by the caller.

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/data/users/maxiv25/opt_project/ReasoningAblationStudyMIPT2026}"
PYTHON="${PYTHON:-/data/users/maxiv25/opt_project/venv/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-${ARTIFACT_ROOT}/results/maj8_test_20260815}"
SFT_MODEL="${ARTIFACT_ROOT}/outputs/sft_lora_r64_16k_two_epochs_merged"

mkdir -p "${RESULTS_DIR}"

COMMON=(
  --benchmarks gsm8k math500
  --num-samples 8
  --max-new-tokens 16384
  --temperature 1.0
  --top-p 1.0
  --top-k 0
  --seed 42
  --gpu-mem 0.30
)

run_eval() {
  local name="$1"
  shift
  local metrics="${RESULTS_DIR}/${name}.json"
  local traces="${RESULTS_DIR}/${name}.traces.jsonl.gz"
  if [[ -s "${metrics}" && -s "${traces}" ]]; then
    echo "SKIP complete: ${name}"
    return
  fi
  echo "START: ${name}"
  "${PYTHON}" -u -m src.evaluate \
    "${COMMON[@]}" \
    --output "${metrics}" \
    --traces-output "${traces}" \
    "$@"
  echo "DONE: ${name}"
}

cd "${CODE_ROOT}"

run_eval base \
  --model Qwen/Qwen3.5-0.8B-Base

run_eval sft_two_epochs \
  --model "${SFT_MODEL}" \
  --chat-template

run_eval grpo_vanilla_decay_lr1p5e6 \
  --model "${SFT_MODEL}" \
  --lora-path "${ARTIFACT_ROOT}/outputs/grpo_vanilla_full_gpu1_val20_postsync_r4" \
  --chat-template

run_eval grpo_vanilla_constant_lr1p5e6 \
  --model "${SFT_MODEL}" \
  --lora-path "${ARTIFACT_ROOT}/outputs/grpo_vanilla_constant_full_gpu1_val20_postsync_r1" \
  --chat-template

run_eval grpo_dpo_z_cosine_lr1p5e6 \
  --model "${SFT_MODEL}" \
  --lora-path "${ARTIFACT_ROOT}/outputs/grpo_dpo_z_full_gpu7_val20_postsync_r4" \
  --chat-template
