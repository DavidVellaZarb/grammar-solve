#!/usr/bin/env bash
set -euo pipefail

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
    --metrics '["iou", "syntax_validity", "bleu"]' \
    --labels '["Baseline", "Grammar-Guided (Ours)"]' \
    --metric_labels '{"iou": "Volumetric IoU", "syntax_validity": "Syntax Validity", "bleu": "BLEU"}' \
    --output_path results/openscad/comparison.png \
    --title "OpenSCAD Code Generation"
