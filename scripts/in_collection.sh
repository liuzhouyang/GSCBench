#!/usr/bin/env bash
set -euo pipefail

model="${1:?Usage: bash scripts/in_collection.sh MODEL DATASET [VARIANT]}"
dataset="${2:?Specify a dataset}"
variant_args=()
if [[ -n "${3:-}" ]]; then variant_args=(--variant "$3"); fi
read -r -a seed_pairs <<< "${SEED_PAIRS:-1729:0 1729:1 1729:2 3407:0 9679:0}"

for pair in "${seed_pairs[@]}"; do
  python -m scripts.run_experiment --config configs/experiments/in_collection.yaml \
    --model "$model" --dataset-name "$dataset" "${variant_args[@]}" \
    --split-seed "${pair%:*}" --seed "${pair#*:}"
done
