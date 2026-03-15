#!/usr/bin/env bash
set -euo pipefail

ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_smcalflow"
BALANCED_ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_smcalflow_balanced"

for k in 8 16 32; do
    uv run python src/eval_grammar.py \
        --predicted_path outputs/predicted_grammars/rag/test_k${k}.json \
        --gold_path data/smcalflow/test.json \
        --write

    uv run python src/eval_grammar.py \
        --predicted_path outputs/predicted_grammars/rag/test_balanced_k${k}.json \
        --gold_path data/smcalflow/test_balanced.json \
        --write
done

for k in 8 16 32; do
    uv run python src/eval.py \
        --adapter "$ADAPTER" \
        --grammar_file outputs/predicted_grammars/rag/test_k${k}.json \
        --test_path data/smcalflow/test.json \
        --output_path results/rag/test_k${k}.json
done

for k in 8 16 32; do
    uv run python src/eval.py \
        --adapter "$BALANCED_ADAPTER" \
        --grammar_file outputs/predicted_grammars/rag/test_balanced_k${k}.json \
        --test_path data/smcalflow/test_balanced.json \
        --output_path results/rag_balanced/test_balanced_k${k}.json
done

uv run python src/plot.py plot_accuracies \
    --results_dir results \
    --models '["rag"]' \
    --model_labels '{"rag": "RAG"}' \
    --test_labels '{"test_k8": "k=8", "test_k16": "k=16", "test_k32": "k=32"}' \
    --output_path results/rag/rag_accuracy.png

uv run python src/plot.py plot_accuracies \
    --results_dir results \
    --models '["rag_balanced"]' \
    --model_labels '{"rag_balanced": "RAG (Balanced)"}' \
    --test_labels '{"test_balanced_k8": "k=8", "test_balanced_k16": "k=16", "test_balanced_k32": "k=32"}' \
    --output_path results/rag_balanced/rag_balanced_accuracy.png
