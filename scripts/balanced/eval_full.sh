#!/usr/bin/env bash
set -euo pipefail

BASELINE_ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_smcalflow_balanced-baseline"
NORMAL_ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_smcalflow_balanced"
ABLATED_ADAPTER="dv347/qwen2.5-7b_smcalflow_balanced-ablated"
GRAMMAR_ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_smcalflow_balanced-grammar"
CLASSIFIER="${HF_NAMESPACE}/deberta-v3-base_smcalflow_balanced-classifier"

TEST_PATH="data/smcalflow/test_balanced.json"
TEST_GENERIC_PATH="data/smcalflow/test_balanced_generic.json"

GENERATIVE_OUT="outputs/predicted_grammars/balanced_generative.json"
CLASSIFIER_GENERIC_OUT="outputs/predicted_grammars/balanced_classifier_t0.5_generic.json"
CLASSIFIER_SPECIALIZED_OUT="outputs/predicted_grammars/balanced_classifier_t0.5_specialized.json"

# --- Step 1: Generate grammars with generative model ---

uv run python src/generate_grammar.py \
    --adapter "$GRAMMAR_ADAPTER" \
    --test_path "$TEST_PATH" \
    --output_path "$GENERATIVE_OUT"

# --- Step 2: Predict grammars with classifier ---

uv run python src/classifier.py predict \
    --test_path "$TEST_GENERIC_PATH" \
    --classifier "$CLASSIFIER" \
    --threshold 0.5 \
    --output_path "$CLASSIFIER_GENERIC_OUT"

# --- Step 3: Specialize classifier predictions ---

uv run python src/specialize_grammar.py \
    --test_path "$CLASSIFIER_GENERIC_OUT" \
    --train_path "data/smcalflow/train_balanced.json" \
    --train_generic_path "data/smcalflow/train_balanced_generic.json" \
    --cache_path "cache/specialize_balanced_cache.json" \
    --output_path "$CLASSIFIER_SPECIALIZED_OUT"

# --- Step 4: Evaluate grammar quality ---

uv run python src/eval_grammar.py \
    --predicted_path "$CLASSIFIER_GENERIC_OUT" \
    --gold_path "$TEST_GENERIC_PATH" \
    --write

uv run python src/eval_grammar.py \
    --predicted_path "$CLASSIFIER_SPECIALIZED_OUT" \
    --gold_path "$TEST_PATH" \
    --write

# --- Step 5: Evaluate normal model (4 conditions) ---

uv run python src/eval.py \
    --adapter "$BASELINE_ADAPTER" \
    --noinclude_grammar \
    --test_path "$TEST_PATH" \
    --output_path results/balanced_normal/baseline.json

uv run python src/eval.py \
    --adapter "$NORMAL_ADAPTER" \
    --grammar_file "$CLASSIFIER_SPECIALIZED_OUT" \
    --test_path "$TEST_PATH" \
    --output_path results/balanced_normal/classifier.json

uv run python src/eval.py \
    --adapter "$NORMAL_ADAPTER" \
    --grammar_file "$GENERATIVE_OUT" \
    --test_path "$TEST_PATH" \
    --output_path results/balanced_normal/generative.json

uv run python src/eval.py \
    --adapter "$NORMAL_ADAPTER" \
    --test_path "$TEST_PATH" \
    --output_path results/balanced_normal/gold.json

# --- Step 6: Evaluate ablated model (4 conditions) ---

uv run python src/eval.py \
    --adapter "$BASELINE_ADAPTER" \
    --noinclude_grammar \
    --test_path "$TEST_PATH" \
    --output_path results/balanced_ablated/baseline.json

uv run python src/eval.py \
    --adapter "$ABLATED_ADAPTER" \
    --grammar_file "$CLASSIFIER_SPECIALIZED_OUT" \
    --test_path "$TEST_PATH" \
    --output_path results/balanced_ablated/classifier.json

uv run python src/eval.py \
    --adapter "$ABLATED_ADAPTER" \
    --grammar_file "$GENERATIVE_OUT" \
    --test_path "$TEST_PATH" \
    --output_path results/balanced_ablated/generative.json

uv run python src/eval.py \
    --adapter "$ABLATED_ADAPTER" \
    --test_path "$TEST_PATH" \
    --output_path results/balanced_ablated/gold.json

# --- Step 7: Plot results ---

uv run python src/plot.py plot_accuracies \
    --results_dir results \
    --models '["balanced_normal"]' \
    --model_labels '{"balanced_normal": "Normal"}' \
    --test_labels '{"baseline": "Without Grammar", "classifier": "Classifier Grammar", "generative": "Generated Grammar", "gold": "Gold Grammar"}' \
    --output_path results/balanced_normal/balanced_normal.png

uv run python src/plot.py plot_accuracies \
    --results_dir results \
    --models '["balanced_ablated"]' \
    --model_labels '{"balanced_ablated": "Ablated"}' \
    --test_labels '{"baseline": "Without Grammar", "classifier": "Classifier Grammar", "generative": "Generated Grammar", "gold": "Gold Grammar"}' \
    --output_path results/balanced_ablated/balanced_ablated.png
