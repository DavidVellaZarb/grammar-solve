#!/usr/bin/env bash
set -euo pipefail

# --- Add Rule (20%) ---

ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_smcalflow-add-rule-p20"
RESULTS_DIR="results/ablations_p20/add_rule_p20"

uv run python src/train.py \
    --train_path "data/smcalflow/ablations/train_add_rule_p20.json" \
    --hub_model_id "$ADAPTER" \
    "$@"

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path data/smcalflow/test.json \
    --output_path "$RESULTS_DIR/test.json"

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path data/smcalflow/ablations/test_add_rule.json \
    --output_path "$RESULTS_DIR/test_add_rule.json"

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path data/smcalflow/ablations/test_remove_rule.json \
    --output_path "$RESULTS_DIR/test_remove_rule.json"

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path data/smcalflow/ablations/test_add_remove_rule.json \
    --output_path "$RESULTS_DIR/test_add_remove_rule.json"

uv run python src/plot.py \
    --results_dir results/ablations_p20 \
    --models '["add_rule_p20"]' \
    --output_path results/ablations_p20/add_rule_p20/ablation_add_rule_p20.png \
    --title "Add Rule Model (20%)" \
    --model_labels '{"add_rule_p20": "Add Rule (20%)"}' \
    --test_labels '{"test": "Original", "test_add_rule": "Added Rules", "test_remove_rule": "Removed Rules", "test_add_remove_rule": "Added+Removed Rules"}'

# --- Remove Rule (20%) ---

ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_smcalflow-remove-rule-p20"
RESULTS_DIR="results/ablations_p20/remove_rule_p20"

uv run python src/train.py \
    --train_path "data/smcalflow/ablations/train_remove_rule_p20.json" \
    --hub_model_id "$ADAPTER" \
    "$@"

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path data/smcalflow/test.json \
    --output_path "$RESULTS_DIR/test.json"

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path data/smcalflow/ablations/test_add_rule.json \
    --output_path "$RESULTS_DIR/test_add_rule.json"

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path data/smcalflow/ablations/test_remove_rule.json \
    --output_path "$RESULTS_DIR/test_remove_rule.json"

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path data/smcalflow/ablations/test_add_remove_rule.json \
    --output_path "$RESULTS_DIR/test_add_remove_rule.json"

uv run python src/plot.py \
    --results_dir results/ablations_p20 \
    --models '["remove_rule_p20"]' \
    --output_path results/ablations_p20/remove_rule_p20/ablation_remove_rule_p20.png \
    --title "Remove Rule Model (20%)" \
    --model_labels '{"remove_rule_p20": "Remove Rule (20%)"}' \
    --test_labels '{"test": "Original", "test_add_rule": "Added Rules", "test_remove_rule": "Removed Rules", "test_add_remove_rule": "Added+Removed Rules"}'

# --- Add+Remove Rule (20%) ---

ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_smcalflow-add-remove-rule-p20"
RESULTS_DIR="results/ablations_p20/add_remove_rule_p20"

uv run python src/train.py \
    --train_path "data/smcalflow/ablations/train_add_remove_rule_p20.json" \
    --hub_model_id "$ADAPTER" \
    "$@"

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path data/smcalflow/test.json \
    --output_path "$RESULTS_DIR/test.json"

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path data/smcalflow/ablations/test_add_rule.json \
    --output_path "$RESULTS_DIR/test_add_rule.json"

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path data/smcalflow/ablations/test_remove_rule.json \
    --output_path "$RESULTS_DIR/test_remove_rule.json"

uv run python src/eval.py \
    --adapter "$ADAPTER" \
    --test_path data/smcalflow/ablations/test_add_remove_rule.json \
    --output_path "$RESULTS_DIR/test_add_remove_rule.json"

uv run python src/plot.py \
    --results_dir results/ablations_p20 \
    --models '["add_remove_rule_p20"]' \
    --output_path results/ablations_p20/add_remove_rule_p20/ablation_add_remove_rule_p20.png \
    --title "Add+Remove Rule Model (20%)" \
    --model_labels '{"add_remove_rule_p20": "Add+Remove Rule (20%)"}' \
    --test_labels '{"test": "Original", "test_add_rule": "Added Rules", "test_remove_rule": "Removed Rules", "test_add_remove_rule": "Added+Removed Rules"}'
