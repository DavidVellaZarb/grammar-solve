#!/usr/bin/env bash
set -euo pipefail

uv run python src/eval_smiles.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_smiles-baseline" \
    --test_path data/smiles/test.json \
    --noinclude_grammar \
    --output_path results/smiles/baseline/test.json

uv run python src/eval_smiles.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_smiles" \
    --test_path data/smiles/test.json \
    --include_grammar \
    --output_path results/smiles/grammar/test.json

uv run python src/plot.py plot_accuracies \
    --results_dir results/smiles \
    --models '["baseline", "grammar"]' \
    --model_labels '{"baseline": "Baseline", "grammar": "Grammar-Guided (Ours)"}' \
    --output_path results/smiles/comparison.png \
    --title "SMILES Generation Accuracy"
