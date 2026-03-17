#!/usr/bin/env bash
set -euo pipefail

uv run python src/eval_smiles.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_smiles" \
    --grammar_file outputs/predicted_grammars/rag/smiles_test_k64.json \
    --test_path data/smiles/test.json \
    --output_path results/rag_smiles/test.json

uv run python src/plot.py plot_multi_metrics \
    --result_files '["results/smiles/baseline/test.json", "results/rag_smiles/test.json", "results/smiles/grammar/test.json"]' \
    --metrics '["canonical_exact_match", "validity", "fingerprint_similarity", "bleu"]' \
    --labels '["Baseline", "RAG", "Gold Grammar"]' \
    --metric_labels '{"canonical_exact_match": "Canonical Exact Match", "validity": "Validity", "fingerprint_similarity": "Fingerprint Similarity", "bleu": "BLEU"}' \
    --output_path results/rag_smiles/comparison.png \
    --title "SMILES — RAG Grammar Prediction"
