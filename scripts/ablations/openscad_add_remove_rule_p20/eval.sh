#!/usr/bin/env bash
set -euo pipefail

ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_openscad-add-remove-rule-p20"
RESULTS_DIR="results/ablations/openscad_add_remove_rule_p20"

uv run python src/eval_openscad.py \
    --adapter "$ADAPTER" \
    --test_path data/openscad/test.json \
    --include_grammar \
    --output_path "$RESULTS_DIR/gold_grammar.json"

uv run python src/eval_openscad.py \
    --adapter "$ADAPTER" \
    --test_path data/openscad/test.json \
    --grammar_file outputs/predicted_grammars/rag/openscad_test_k64.json \
    --output_path "$RESULTS_DIR/rag_grammar.json"

uv run python src/plot.py plot_multi_metrics \
    --result_files '["results/openscad/baseline/test.json", "'"$RESULTS_DIR"'/rag_grammar.json", "'"$RESULTS_DIR"'/gold_grammar.json"]' \
    --metrics '["iou", "syntax_validity", "bleu"]' \
    --labels '["Baseline", "RAG", "Gold Grammar"]' \
    --metric_labels '{"iou": "Volumetric IoU", "syntax_validity": "Syntax Validity", "bleu": "BLEU"}' \
    --output_path "$RESULTS_DIR/comparison.png" \
    --title "OpenSCAD — Ablated Model (add_remove p=20%)"
