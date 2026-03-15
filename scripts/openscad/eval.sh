#!/usr/bin/env bash
set -euo pipefail

# Requires: apt-get update && apt-get install -y openscad xvfb

uv run python src/eval_openscad.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_openscad-baseline" \
    --test_path data/openscad/test.json \
    --noinclude_grammar \
    --output_path results/openscad/baseline/test.json \

uv run python src/eval_openscad.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_openscad" \
    --test_path data/openscad/test.json \
    --include_grammar \
    --output_path results/openscad/grammar/test.json

uv run python src/plot.py plot_multi_metrics \
    --result_files '["results/openscad/baseline/test.json", "results/openscad/grammar/test.json"]' \
    --metrics '["syntax_validity", "chamfer_distance", "iou"]' \
    --labels '["Baseline", "Grammar-Guided (Ours)"]' \
    --metric_labels '{"syntax_validity": "Syntax Validity", "chamfer_distance": "Chamfer Distance", "iou": "Volumetric IoU"}' \
    --output_path results/openscad/comparison.png \
    --title "OpenSCAD Code Generation"
