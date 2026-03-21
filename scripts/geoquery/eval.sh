#!/usr/bin/env bash
set -euo pipefail

uv run python src/eval_geoquery.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_geoquery-baseline" \
    --test_path data/geoquery/test.json \
    --noinclude_grammar \
    --output_path results/geoquery/baseline/test.json \
    "$@"

uv run python src/eval_geoquery.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_geoquery" \
    --test_path data/geoquery/test.json \
    --include_grammar \
    --output_path results/geoquery/grammar/test.json \
    "$@"

uv run python src/plot.py plot_multi_metrics \
    --result_files '["results/geoquery/baseline/test.json", "results/geoquery/grammar/test.json"]' \
    --labels '["Baseline", "Gold Grammar"]' \
    --metrics '["execution_accuracy", "exact_match", "bleu"]' \
    --metric_labels '{"execution_accuracy": "Execution Acc", "exact_match": "Exact Match", "bleu": "BLEU"}' \
    --output_path results/geoquery/baseline_vs_grammar.png \
    --title "GeoQuery: Baseline vs Grammar-Guided"
