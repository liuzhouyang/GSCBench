#!/usr/bin/env bash
set -euo pipefail

model="${1:?Usage: bash scripts/fewshot_scratch.sh MODEL [DATASET] [VARIANT]}"
datasets=(tu_AIDS tu_BZR tu_COX2 tu_DHFR tu_ENZYMES tu_IMDB-BINARY tu_MUTAG tu_NCI1 tu_PROTEINS tu_PTC_MR ogb_ogbg-code2)
if [[ -n "${2:-}" ]]; then datasets=("$2"); fi
variant_args=()
if [[ -n "${3:-}" ]]; then variant_args=(--variant "$3"); fi
read -r -a seed_pairs <<< "${SEED_PAIRS:-1729:0 1729:1 1729:2 3407:0 9679:0}"
read -r -a budgets <<< "${PAIR_BUDGETS:-100 500 2000}"

for dataset in "${datasets[@]}"; do
  for budget in "${budgets[@]}"; do
    for pair in "${seed_pairs[@]}"; do
      python -m scripts.run_pretrain --config configs/experiments/fewshot_scratch.yaml \
        --model "$model" --train-datasets "$dataset" "${variant_args[@]}" \
        --train-pair-budget "$budget" --split-seed "${pair%:*}" --seed "${pair#*:}"
    done
  done
done
