#!/usr/bin/env bash
set -euo pipefail

CONFIGS=("n1" "n2" "n3" "n4" "n2-4" "n3-5" "n4-6")

for CFG in "${CONFIGS[@]}"; do
    ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_smcalflow-add-remove-${CFG}-p20"

    echo "=== Training: ${CFG} ==="
    uv run python src/train.py \
        --train_path "data/smcalflow/train_add_remove_${CFG}_p20.json" \
        --hub_model_id "$ADAPTER" \
        "$@"
done
