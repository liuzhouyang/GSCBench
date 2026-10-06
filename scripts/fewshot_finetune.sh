#!/usr/bin/env bash
set -euo pipefail

model="${1:?Usage: bash scripts/fewshot_finetune.sh MODEL TARGET CHECKPOINT [VARIANT]}"
target="${2:?Specify a target dataset}"
checkpoint="${3:?Specify the pretrained checkpoint}"
variant_args=()
if [[ -n "${4:-}" ]]; then variant_args=(--variant "$4"); fi
read -r -a seed_pairs <<< "${SEED_PAIRS:-1729:0 1729:1 1729:2}"
read -r -a budgets <<< "${PAIR_BUDGETS:-100 500 2000}"

for budget in "${budgets[@]}"; do
  for pair in "${seed_pairs[@]}"; do
    python -m scripts.run_pretrain --config configs/experiments/fewshot_finetune.yaml \
      --model "$model" --train-datasets "$target" "${variant_args[@]}" \
      --init-from "$checkpoint" --train-pair-budget "$budget" \
      --split-seed "${pair%:*}" --seed "${pair#*:}"
  done
done
