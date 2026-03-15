#!/usr/bin/env bash
set -euo pipefail

ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_smcalflow-add-remove-rule-p20"

for k in 8 16 32 64 128 256; do
    uv run python src/eval.py \
        --adapter "$ADAPTER" \
        --grammar_file outputs/predicted_grammars/rag/test_k${k}.json \
        --test_path data/smcalflow/test.json \
        --output_path results/rag_ablated/test_k${k}.json
done

uv run python src/plot.py plot_bar_chart \
    --result_files '["results/rag_ablated/test_k8.json", "results/rag_ablated/test_k16.json", "results/rag_ablated/test_k32.json", "results/rag_ablated/test_k64.json", "results/rag_ablated/test_k128.json", "results/rag_ablated/test_k256.json"]' \
    --labels '["k=8", "k=16", "k=32", "k=64", "k=128", "k=256"]' \
    --reference_lines '[{"value_from": "results/baseline/baseline.json", "metric": "accuracy", "label": "Baseline (no grammar)", "style": "dotted", "color": "gray"}, {"value_from": "results/ablations_p20/add_remove_rule_p20/test.json", "metric": "accuracy", "label": "Gold grammar", "style": "dashed", "color": "green"}]' \
    --output_path results/rag_ablated/rag_ablated_accuracy.png \
    --title "RAG Ablated — Program Accuracy by k" \
    --ylabel Accuracy
