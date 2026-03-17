#!/usr/bin/env bash
set -euo pipefail

uv run python src/eval_openscad.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_openscad" \
    --grammar_file outputs/predicted_grammars/rag/openscad_test_k64.json \
    --test_path data/openscad/test.json \
    --output_path results/rag_openscad/test.json

uv run python src/plot.py plot_multi_metrics \
    --result_files '["results/openscad/baseline/test.json", "results/rag_openscad/test.json", "results/openscad/grammar/test.json"]' \
    --metrics '["syntax_validity", "iou"]' \
    --labels '["Baseline", "RAG", "Gold Grammar"]' \
    --metric_labels '{"syntax_validity": "Syntax Validity", "iou": "Volumetric IoU"}' \
    --output_path results/rag_openscad/comparison.png \
    --title "OpenSCAD — RAG Grammar Prediction"
