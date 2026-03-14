#!/usr/bin/env bash
set -euo pipefail

uv run python src/generate_cot.py \
    --input_path "data/smcalflow/train_balanced.json" \
    --output_path "data/smcalflow/train_balanced_cot.json" \
    "$@"

uv run python src/generate_cot.py \
    --input_path "data/smcalflow/valid_balanced.json" \
    --output_path "data/smcalflow/valid_balanced_cot.json" \
    "$@"
