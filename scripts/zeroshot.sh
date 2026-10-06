#!/usr/bin/env bash
set -euo pipefail

model="${1:?Usage: bash scripts/zeroshot.sh MODEL SOURCE TARGET CHECKPOINT SPLIT_SEED TRAIN_SEED [VARIANT]}"
source_dataset="${2:?Specify the source dataset}"
target="${3:?Specify the target dataset}"
checkpoint="${4:?Specify the source checkpoint}"
split_seed="${5:?Specify the source split seed}"
train_seed="${6:?Specify the source training seed}"
variant_args=()
if [[ -n "${7:-}" ]]; then variant_args=(--variant "$7"); fi

python -m scripts.run_experiment --config configs/experiments/zeroshot.yaml \
  --model "$model" --source-dataset "$source_dataset" --dataset-name "$target" \
  --checkpoint "$checkpoint" --split-seed "$split_seed" --seed "$train_seed" "${variant_args[@]}"
