#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=$1
MODEL_ALIAS=$2

RESULT_DIR="results/smiles/${MODEL_ALIAS}"
PRED_DIR=outputs/predicted_grammars/rag_cot

BASELINE_ADAPTER="${HF_NAMESPACE}/${MODEL_ALIAS}_smiles-baseline"
OURS_ADAPTER="${HF_NAMESPACE}/${MODEL_ALIAS}_smiles-mixed-r0.1"

echo "=== Baseline (no grammar) ==="
uv run python src/eval_smiles.py \
    --adapter "$BASELINE_ADAPTER" \
    --test_path data/smiles/test.json \
    --noinclude_grammar \
    --output_path "${RESULT_DIR}/baseline.json"

echo "=== Ours (RAG grammar) ==="
uv run python src/eval_smiles.py \
    --adapter "$OURS_ADAPTER" \
    --test_path data/smiles/test.json \
    --include_grammar \
    --grammar_file "${PRED_DIR}/smiles_test_k64.json" \
    --output_path "${RESULT_DIR}/rag.json"

echo "=== Ours (gold grammar) ==="
uv run python src/eval_smiles.py \
    --adapter "$OURS_ADAPTER" \
    --test_path data/smiles/test.json \
    --include_grammar \
    --output_path "${RESULT_DIR}/gold.json"

echo "=== Plotting ==="
uv run python src/plot.py plot_paper_results \
    --result_files "[\"${RESULT_DIR}/baseline.json\", \"${RESULT_DIR}/rag.json\", \"${RESULT_DIR}/gold.json\"]" \
    --labels '["Baseline", "Ours (RAG)", "Gold Grammar"]' \
    --metrics '["fingerprint_similarity", "validity", "canonical_exact_match"]' \
    --metric_labels '{"fingerprint_similarity": "Fingerprint Similarity", "validity": "Validity", "canonical_exact_match": "Exact Match"}' \
    --per_example_fields '{"fingerprint_similarity": "fingerprint_similarity", "validity": "valid", "canonical_exact_match": "canonical_match"}' \
    --output_path "${RESULT_DIR}/comparison.png" \
    --title "SMILES (${MODEL_ALIAS})"
