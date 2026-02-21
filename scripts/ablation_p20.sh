#!/usr/bin/env bash
set -euo pipefail

ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_smcalflow-add-rule-p20"
RESULTS_DIR="results/add_rule_p20"

uv run python src/train.py \
    --train_path "data/smcalflow/train_add_rule_p20.json" \
    --hub_model_id "$ADAPTER" \
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

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path data/smcalflow/test_add_remove_rule.json \
    --output_path "$RESULTS_DIR/test_add_remove_rule.json"

ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_smcalflow-remove-rule-p20"
RESULTS_DIR="results/remove_rule_p20"

uv run python src/train.py \
    --train_path "data/smcalflow/train_remove_rule_p20.json" \
    --hub_model_id "$ADAPTER" \
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

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path data/smcalflow/test_add_remove_rule.json \
    --output_path "$RESULTS_DIR/test_add_remove_rule.json"

ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_smcalflow-add-remove-rule-p20"
RESULTS_DIR="results/add_remove_rule_p20"

uv run python src/train.py \
    --train_path "data/smcalflow/train_add_remove_rule_p20.json" \
    --hub_model_id "$ADAPTER" \
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

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path data/smcalflow/test_add_remove_rule.json \
    --output_path "$RESULTS_DIR/test_add_remove_rule.json"

uv run python src/plot.py \
    --results_dir results \
    --models '["add_rule_p20", "remove_rule_p20", "add_remove_rule_p20"]' \
    --output_path results/ablation_p20.png \
    --title "Ablation p=0.2" \
    --model_labels '{"add_rule_p20": "Add Rule (p=0.2)", "remove_rule_p20": "Remove Rule (p=0.2)", "add_remove_rule_p20": "Add+Remove Rule (p=0.2)"}' \
    --test_labels '{"test": "Original", "test_add_rule": "Added Rules", "test_remove_rule": "Removed Rules", "test_add_remove_rule": "Added+Removed Rules"}'
