#!/usr/bin/env bash
set -euo pipefail

CONFIGS=("n1" "n2" "n3" "n4" "n2-4" "n3-5" "n4-6")
TEST_SETS=("test" "test_add_remove_n1" "test_add_remove_n2" "test_add_remove_n3" "test_add_remove_n4" "test_add_remove_n2-4" "test_add_remove_n3-5" "test_add_remove_n4-6")
RESULTS_BASE="results/ablations_add_remove"

TEST_LABELS='{"test": "Gold", "test_add_remove_n1": "n=1", "test_add_remove_n2": "n=2", "test_add_remove_n3": "n=3", "test_add_remove_n4": "n=4", "test_add_remove_n2-4": "n=[2,4)", "test_add_remove_n3-5": "n=[3,5)", "test_add_remove_n4-6": "n=[4,6)"}'

for CFG in "${CONFIGS[@]}"; do
    ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_smcalflow-add-remove-${CFG}-p20"
    RESULTS_DIR="${RESULTS_BASE}/add_remove_${CFG}_p20"

    echo "=== Evaluating: ${CFG} ==="

    for TEST in "${TEST_SETS[@]}"; do
        uv run python src/eval.py \
            --adapter "$ADAPTER" \
            --test_path "data/smcalflow/${TEST}.json" \
            --output_path "$RESULTS_DIR/${TEST}.json" \
            "$@"
    done

    uv run python src/plot.py \
        --results_dir "$RESULTS_BASE" \
        --models "[\"add_remove_${CFG}_p20\"]" \
        --output_path "$RESULTS_DIR/ablation_add_remove_${CFG}_p20.png" \
        --title "Add+Remove Model (n_ops=${CFG}, 20%)" \
        --model_labels "{\"add_remove_${CFG}_p20\": \"Add+Remove (${CFG})\"}" \
        --test_labels "$TEST_LABELS"
done
