#!/usr/bin/env bash
set -euo pipefail

# --- Add Rule (10%) ---

ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_smcalflow-add-rule-p10"
RESULTS_DIR="results/ablations_p10/add_rule_p10"

uv run python src/train.py \
    --train_path "data/smcalflow/train_add_rule_p10.json" \
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
    --results_dir results/ablations_p10 \
    --models '["add_rule_p10"]' \
    --output_path results/ablations_p10/add_rule_p10/ablation_add_rule_p10.png \
    --title "Add Rule Model (10%)" \
    --model_labels '{"add_rule_p10": "Add Rule (10%)"}' \
    --test_labels '{"test": "Original", "test_add_rule": "Added Rules", "test_remove_rule": "Removed Rules", "test_add_remove_rule": "Added+Removed Rules"}'

# --- Remove Rule (10%) ---

ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_smcalflow-remove-rule-p10"
RESULTS_DIR="results/ablations_p10/remove_rule_p10"

uv run python src/train.py \
    --train_path "data/smcalflow/train_remove_rule_p10.json" \
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
    --results_dir results/ablations_p10 \
    --models '["remove_rule_p10"]' \
    --output_path results/ablations_p10/remove_rule_p10/ablation_remove_rule_p10.png \
    --title "Remove Rule Model (10%)" \
    --model_labels '{"remove_rule_p10": "Remove Rule (10%)"}' \
    --test_labels '{"test": "Original", "test_add_rule": "Added Rules", "test_remove_rule": "Removed Rules", "test_add_remove_rule": "Added+Removed Rules"}'

# --- Add+Remove Rule (10%) ---

ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_smcalflow-add-remove-rule-p10"
RESULTS_DIR="results/ablations_p10/add_remove_rule_p10"

uv run python src/train.py \
    --train_path "data/smcalflow/train_add_remove_rule_p10.json" \
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
    --results_dir results/ablations_p10 \
    --models '["add_remove_rule_p10"]' \
    --output_path results/ablations_p10/add_remove_rule_p10/ablation_add_remove_rule_p10.png \
    --title "Add+Remove Rule Model (10%)" \
    --model_labels '{"add_remove_rule_p10": "Add+Remove Rule (10%)"}' \
    --test_labels '{"test": "Original", "test_add_rule": "Added Rules", "test_remove_rule": "Removed Rules", "test_add_remove_rule": "Added+Removed Rules"}'
