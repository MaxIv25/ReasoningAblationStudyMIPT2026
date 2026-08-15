#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 {vanilla|dpo_z|prime} GPU_ID [DATA_DIR]" >&2
  exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || usage

variant=$1
gpu_id=$2
data_dir=${3:-data/grpo_prime_calibrated_v3}

[[ $gpu_id =~ ^[0-9]+$ ]] || {
  echo "GPU_ID must be a non-negative integer, got: $gpu_id" >&2
  exit 2
}

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-"$project_root/../venv/bin/python"}

case "$variant" in
  vanilla)
    module=src.train_grpo
    config=configs/grpo_vanilla_sft_lora_smoke.yaml
    output_dir=outputs/grpo_vanilla_from_sft_lora_smoke
    ;;
  dpo_z)
    module=src.train_grpo
    config=configs/grpo_dpo_z_smoke.yaml
    output_dir=outputs/grpo_dpo_z_from_sft_lora_smoke
    ;;
  prime)
    module=src.train_prime
    config=configs/prime_research_16k_smoke.yaml
    output_dir=outputs/prime_research_16k_smoke
    ;;
  *)
    usage
    ;;
esac

cd "$project_root"

[[ -x $python_bin ]] || {
  echo "Python environment not found or not executable: $python_bin" >&2
  exit 1
}
[[ -d $data_dir ]] || {
  echo "Dataset directory not found: $data_dir" >&2
  exit 1
}

model_dir=outputs/sft_lora_r64_16k_two_epochs_merged
[[ -d $model_dir ]] || {
  echo "Merged SFT model not found: $model_dir" >&2
  exit 1
}

mkdir -p logs
