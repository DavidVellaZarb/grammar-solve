#!/usr/bin/env bash
set -euo pipefail

PRED_DIR=outputs/predicted_grammars/rag_domain_specific
RESULT_DIR=results/rag_domain_specific

echo "=== Evaluating SMILES ==="

uv run python src/eval_smiles.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_smiles" \
    --grammar_file "${PRED_DIR}/smiles_test_k64.json" \
    --test_path data/smiles/test.json \
    --output_path "${RESULT_DIR}/smiles/test.json"

uv run python src/plot.py plot_multi_metrics \
    --result_files '["results/smiles/baseline/test.json", "'"${RESULT_DIR}"'/smiles/test.json", "results/smiles/grammar/test.json"]' \
    --metrics '["canonical_exact_match", "validity", "fingerprint_similarity", "bleu"]' \
    --labels '["Baseline", "RAG", "Gold Grammar"]' \
    --metric_labels '{"canonical_exact_match": "Canonical Exact Match", "validity": "Validity", "fingerprint_similarity": "Fingerprint Similarity", "bleu": "BLEU"}' \
    --output_path "${RESULT_DIR}/smiles/comparison.png" \
    --title "SMILES — RAG Grammar Prediction (Domain-Specific)"

echo "=== Evaluating SPICE ==="

uv run python src/eval_spice.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_spice" \
    --grammar_file "${PRED_DIR}/spice_test_k64.json" \
    --test_path data/spice/test.json \
    --output_path "${RESULT_DIR}/spice/test.json"

uv run python src/plot.py plot_multi_metrics \
    --result_files '["results/spice/baseline/test.json", "'"${RESULT_DIR}"'/spice/test.json", "results/spice/grammar/test.json"]' \
    --metrics '["ged_similarity", "syntax_validity", "exact_match", "bleu", "component_f1"]' \
    --labels '["Baseline", "RAG", "Gold Grammar"]' \
    --metric_labels '{"ged_similarity": "GED Similarity", "syntax_validity": "Syntax Validity", "exact_match": "Exact Match", "bleu": "BLEU", "component_f1": "Component F1"}' \
    --output_path "${RESULT_DIR}/spice/comparison.png" \
    --title "SPICE — RAG Grammar Prediction (Domain-Specific)"

echo "=== Evaluating OpenSCAD ==="

uv run python src/eval_openscad.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_openscad" \
    --grammar_file "${PRED_DIR}/openscad_test_k64.json" \
    --test_path data/openscad/test.json \
    --output_path "${RESULT_DIR}/openscad/test.json"

uv run python src/plot.py plot_multi_metrics \
    --result_files '["results/openscad/baseline/test.json", "'"${RESULT_DIR}"'/openscad/test.json", "results/openscad/grammar/test.json"]' \
    --metrics '["syntax_validity", "iou"]' \
    --labels '["Baseline", "RAG", "Gold Grammar"]' \
    --metric_labels '{"syntax_validity": "Syntax Validity", "iou": "Volumetric IoU"}' \
    --output_path "${RESULT_DIR}/openscad/comparison.png" \
    --title "OpenSCAD — RAG Grammar Prediction (Domain-Specific)"

echo "=== Evaluating Verilog ==="

if ! command -v iverilog &> /dev/null; then
    echo "Error: iverilog not found. Install: apt-get install -y iverilog"
    exit 1
fi

uv run python src/eval_verilog.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_mg-verilog" \
    --grammar_file "${PRED_DIR}/verilog_test_k64.json" \
    --problem_file data/verilog_eval/VerilogEval_Human.jsonl \
    --include_grammar \
    --n_samples 5 \
    --temperature 0.8 \
    --output_path "${RESULT_DIR}/verilog/test.json"

uv run python src/plot.py plot_pass_at_k \
    --result_files '["results/verilog/baseline.json", "'"${RESULT_DIR}"'/verilog/test.json", "results/verilog/grammar.json"]' \
    --labels '["Baseline", "RAG", "Gold Grammar"]' \
    --output_path "${RESULT_DIR}/verilog/pass_at_k.png" \
    --title "Verilog — RAG Grammar Prediction (Domain-Specific)"

echo "All evaluations complete."
