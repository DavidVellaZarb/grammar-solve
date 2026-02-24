#!/usr/bin/env bash
set -euo pipefail

uv run python src/train.py \
    --noinclude_grammar \
    --num_train_epochs 2 \
    --output_dir "outputs/qwen2.5-7b-lora-baseline-2epoch" \
    --hub_model_id "${HF_NAMESPACE}/qwen2.5-7b_smcalflow-baseline-2epoch"

uv run python src/eval.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_smcalflow-baseline-2epoch" \
    --noinclude_grammar \
    --output_path results/baseline_2epoch/baseline.json
