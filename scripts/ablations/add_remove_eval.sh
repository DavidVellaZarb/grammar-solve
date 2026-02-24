#!/usr/bin/env bash
set -euo pipefail

# Evaluate 7 add_remove ablation models on gold + 7 perturbed test sets each.
# 7 models x 8 test sets = 56 eval runs.

CONFIGS=("n1" "n2" "n3" "n4" "n2-4" "n3-5" "n4-6")
TEST_SETS=("test" "test_add_remove_n1" "test_add_remove_n2" "test_add_remove_n3" "test_add_remove_n4" "test_add_remove_n2-4" "test_add_remove_n3-5" "test_add_remove_n4-6")

for CFG in "${CONFIGS[@]}"; do
    ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_smcalflow-add-remove-${CFG}-p20"
    RESULTS_DIR="results/ablations_add_remove/add_remove_${CFG}_p20"

    echo "=== Evaluating: ${CFG} ==="

    for TEST in "${TEST_SETS[@]}"; do
        uv run python src/eval.py \
            --adapter "$ADAPTER" \
            --test_path "data/smcalflow/${TEST}.json" \
            --output_path "$RESULTS_DIR/${TEST}.json" \
            "$@"
    done
done
