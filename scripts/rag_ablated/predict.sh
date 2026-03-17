#!/usr/bin/env bash
set -euo pipefail

for k in 8 16 32 64 128 256; do
    uv run python src/rag_grammar.py predict \
        --test_path data/smcalflow/test.json \
        --train_path data/smcalflow/train.json \
        --k $k \
        --output_path outputs/predicted_grammars/rag/test_k${k}.json

    uv run python src/eval_grammar.py \
        --predicted_path outputs/predicted_grammars/rag/test_k${k}.json \
        --gold_path data/smcalflow/test.json \
        --write
done

uv run python src/plot.py plot_lines \
    --result_files '["outputs/predicted_grammars/rag/test_k8.json", "outputs/predicted_grammars/rag/test_k16.json", "outputs/predicted_grammars/rag/test_k32.json", "outputs/predicted_grammars/rag/test_k64.json", "outputs/predicted_grammars/rag/test_k128.json", "outputs/predicted_grammars/rag/test_k256.json"]' \
    --x_values '[8, 16, 32, 64, 128, 256]' \
    --metrics '["metrics.exact_match", "metrics.relaxed_match"]' \
    --metric_labels '{"metrics.exact_match": "Exact Match", "metrics.relaxed_match": "Relaxed Match"}' \
    --output_path results/rag_ablated/rag_ablated_grammar_quality.png \
    --title "RAG Ablated — Grammar Quality by k" \
    --xlabel k \
    --ylabel Score
