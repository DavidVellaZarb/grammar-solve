#!/usr/bin/env bash
set -euo pipefail

NORMAL_ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_smcalflow_balanced"
TEST_PATH="data/smcalflow/test_balanced.json"
COT_GRAMMARS="outputs/predicted_grammars/balanced_cot.json"

uv run python src/eval_grammar.py \
    --predicted_path "$COT_GRAMMARS" \
    --gold_path "$TEST_PATH" \
    --write

uv run python src/eval.py \
    --adapter "$NORMAL_ADAPTER" \
    --grammar_file "$COT_GRAMMARS" \
    --test_path "$TEST_PATH" \
    --output_path "results/grammar_CoT/test.json" \
    "$@"
