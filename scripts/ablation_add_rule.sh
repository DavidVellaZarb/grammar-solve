#!/usr/bin/env bash
set -euo pipefail

HUB_MODEL_ID="${HF_NAMESPACE:+${HF_NAMESPACE}/qwen2.5-7b_smcalflow-add-rule}"
ADAPTER="${HUB_MODEL_ID:-outputs/qwen2.5-7b-lora-add-rule}"
RESULTS_DIR="results/add_rule"

uv run python src/train.py \
    --train_path "data/smcalflow/train_add_rule.json" \
    --output_dir "outputs/qwen2.5-7b-lora-add-rule" \
    --hub_model_id "$HUB_MODEL_ID" \
    "$@"

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path data/smcalflow/test.json \
    --output_path "$RESULTS_DIR/test.json"

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path data/smcalflow/test_add_rule.json \
    --output_path "$RESULTS_DIR/test_add_rule.json"

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path data/smcalflow/test_remove_rule.json \
    --output_path "$RESULTS_DIR/test_remove_rule.json"

uv run python src/plot.py \
    --results_dir results \
    --models '["add_rule"]' \
    --output_path figures/ablation_add_rule.png \
    --title "Add Rule Model" \
    --model_labels '{"add_rule": "Add Rule"}' \
    --test_labels '{"test": "Original", "test_add_rule": "Added Rules", "test_remove_rule": "Removed Rules"}'
