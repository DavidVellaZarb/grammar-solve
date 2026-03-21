#!/usr/bin/env bash
set -euo pipefail

MODEL="gpt-5.4"
K=64
RESULTS_DIR="results/icl_${MODEL}"
CACHE="cache/icl_${MODEL}_cache.json"

uv run python src/icl.py evaluate_gpt \
    --mode standard --model "$MODEL" --k "$K" \
    --cache_path "$CACHE"

uv run python src/icl.py evaluate_gpt \
    --mode knn --model "$MODEL" --k "$K" \
    --cache_path "$CACHE"

uv run python src/icl.py evaluate_gpt \
    --mode oracle --model "$MODEL" --k "$K" \
    --cache_path "$CACHE"

uv run python src/plot.py plot_bar_chart \
    --result_files "[\"${RESULTS_DIR}/standard_k${K}.json\", \"${RESULTS_DIR}/knn_k${K}.json\", \"${RESULTS_DIR}/oracle_k${K}.json\"]" \
    --labels '["Standard", "kNN", "Oracle"]' \
    --reference_lines '[{"value_from": "results/rag_ablated/test_k64.json", "metric": "accuracy", "label": "RAG (k=64)", "style": "dotted", "color": "gray"}, {"value_from": "results/ablations/p20/add_remove_rule_p20/test.json", "metric": "accuracy", "label": "Gold grammar", "style": "dashed", "color": "green"}]' \
    --output_path "${RESULTS_DIR}/icl_results.png" \
    --title "ICL Evaluation — ${MODEL} (k=${K})"
