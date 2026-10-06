#!/usr/bin/env bash
set -euo pipefail

method="${1:?Usage: bash scripts/approximate.sh METHOD DATASET}"
dataset="${2:?Specify a dataset}"
read -r -a split_seeds <<< "${SPLIT_SEEDS:-1729 3407 9679}"

for split_seed in "${split_seeds[@]}"; do
  python -m scripts.run_experiment --config configs/experiments/approximate.yaml \
    --model "$method" --dataset-name "$dataset" --split-seed "$split_seed" --seed 0
done
