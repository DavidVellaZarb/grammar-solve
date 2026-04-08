#!/usr/bin/env bash
set -euo pipefail

ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_smiles-mixed"
RESULT_DIR=results/mixed/smiles
PRED_DIR=outputs/predicted_grammars/rag_cot

echo "=== Mixed w/o Grammar ==="

uv run python src/eval_smiles.py \
    --adapter "$ADAPTER" \
    --test_path data/smiles/test.json \
    --noinclude_grammar \
    --output_path "${RESULT_DIR}/no_grammar/test.json"

echo "=== Mixed w/ Gold Grammar ==="

uv run python src/eval_smiles.py \
    --adapter "$ADAPTER" \
    --test_path data/smiles/test.json \
    --include_grammar \
    --output_path "${RESULT_DIR}/gold_grammar/test.json"

echo "=== Mixed w/ RAG CoT Grammar ==="

uv run python src/eval_smiles.py \
    --adapter "$ADAPTER" \
    --grammar_file "${PRED_DIR}/smiles_test_k64.json" \
    --test_path data/smiles/test.json \
    --output_path "${RESULT_DIR}/rag_cot/test.json"

echo "=== Plotting comparison ==="

uv run python src/plot.py plot_multi_metrics \
    --result_files '["results/smiles/baseline/test.json", "results/smiles/grammar/test.json", "results/rag_cot/standard/smiles/test.json", "'"${RESULT_DIR}"'/no_grammar/test.json", "'"${RESULT_DIR}"'/gold_grammar/test.json", "'"${RESULT_DIR}"'/rag_cot/test.json"]' \
    --metrics '["canonical_exact_match", "validity", "fingerprint_similarity", "bleu"]' \
    --labels '["Baseline", "Grammar-Guided\nw/ Gold", "Grammar-Guided\nw/ RAG CoT", "Mixed\nw/o Grammar", "Mixed\nw/ Gold", "Mixed\nw/ RAG CoT"]' \
    --metric_labels '{"canonical_exact_match": "Exact Match", "validity": "Validity", "fingerprint_similarity": "FTS (Morgan)", "bleu": "BLEU"}' \
    --output_path "${RESULT_DIR}/comparison.png" \
    --title "SMILES — Mixed Training Comparison"
