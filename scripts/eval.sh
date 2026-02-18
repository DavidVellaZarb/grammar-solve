#!/usr/bin/env bash
set -euo pipefail

uv run python src/eval.py \
    --adapter dv347/qwen2.5-7b_smcalflow \
    --test_path data/smcalflow/test.json \
    --batch_size 8 \
    --max_new_tokens 512 \
    --output_path results/eval_output.json
