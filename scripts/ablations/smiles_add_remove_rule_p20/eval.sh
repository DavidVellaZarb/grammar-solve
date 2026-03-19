#!/usr/bin/env bash
set -euo pipefail

ADAPTER="${HF_NAMESPACE}/qwen2.5-7b_smiles-add-remove-rule-p20"
RESULTS_DIR="results/ablations/smiles_add_remove_rule_p20"

uv run python src/eval_smiles.py \
    --adapter "$ADAPTER" \
    --test_path data/smiles/test.json \
    --include_grammar \
    --output_path "$RESULTS_DIR/gold_grammar.json"

uv run python src/eval_smiles.py \
    --adapter "$ADAPTER" \
    --test_path data/smiles/test.json \
    --grammar_file outputs/predicted_grammars/rag/smiles_test_k64.json \
    --output_path "$RESULTS_DIR/rag_grammar.json"

uv run python src/plot.py plot_multi_metrics \
    --result_files '["results/smiles/baseline/test.json", "'"$RESULTS_DIR"'/rag_grammar.json", "'"$RESULTS_DIR"'/gold_grammar.json"]' \
    --metrics '["canonical_exact_match", "validity", "fingerprint_similarity", "bleu"]' \
    --labels '["Baseline", "RAG", "Gold Grammar"]' \
    --metric_labels '{"canonical_exact_match": "Canonical Exact Match", "validity": "Validity", "fingerprint_similarity": "Fingerprint Similarity", "bleu": "BLEU"}' \
    --output_path "$RESULTS_DIR/comparison.png" \
    --title "SMILES — Noisy Training Grammars (add_remove=20%)"
