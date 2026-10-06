#!/usr/bin/env bash
set -euo pipefail

model="${1:?Usage: bash scripts/cross_collection.sh MODEL [VARIANT]}"
variant_args=()
if [[ -n "${2:-}" ]]; then variant_args=(--variant "$2"); fi

python -m scripts.run_pretrain --config configs/experiments/cross_collection.yaml \
  --model "$model" "${variant_args[@]}" --split-seed 1729 --seed 0
