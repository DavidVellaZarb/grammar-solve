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

uv run python src/plot.py plot_multi_metrics \
    --result_files '["results/smiles/baseline/test.json", "results/smiles/grammar/test.json"]' \
    --labels '["Baseline", "Grammar-Guided (Ours)"]' \
    --metrics '["canonical_exact_match", "validity", "fingerprint_similarity", "bleu"]' \
    --metric_labels '{"canonical_exact_match": "Exact Match", "validity": "Validity", "fingerprint_similarity": "FTS (Morgan)", "bleu": "BLEU"}' \
    --output_path results/smiles/comparison.png \
    --title "SMILES Generation"
