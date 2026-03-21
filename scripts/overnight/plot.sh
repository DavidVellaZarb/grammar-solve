#!/usr/bin/env bash
set -euo pipefail

uv run python src/plot.py plot_multi_metrics \
    --result_files '["results/overnight/baseline/test.json", "results/overnight/rag/test.json", "results/overnight/grammar/test.json"]' \
    --labels '["Baseline", "RAG", "Gold Grammar"]' \
    --metrics '["execution_accuracy", "exact_match", "bleu"]' \
    --metric_labels '{"execution_accuracy": "Execution Acc", "exact_match": "Exact Match", "bleu": "BLEU"}' \
    --output_path results/overnight/comparison.png \
    --title "Overnight-Blocks: Baseline vs RAG vs Gold Grammar"
