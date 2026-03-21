#!/usr/bin/env bash
set -euo pipefail

uv run python src/plot.py plot_multi_metrics \
    --result_files '["results/geoquery/baseline/test.json", "results/geoquery/rag/test.json", "results/geoquery/grammar/test.json"]' \
    --labels '["Baseline", "RAG", "Gold Grammar"]' \
    --metrics '["execution_accuracy", "exact_match", "bleu"]' \
    --metric_labels '{"execution_accuracy": "Execution Acc", "exact_match": "Exact Match", "bleu": "BLEU"}' \
    --output_path results/geoquery/comparison.png \
    --title "GeoQuery: Baseline vs RAG vs Gold Grammar"
