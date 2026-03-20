#!/usr/bin/env bash
set -euo pipefail

PRED_DIR=outputs/predicted_grammars/rag_cot
RESULT_DIR=results/rag_cot/standard
RAG_COMP_DIR=results/rag_cot/rag_comparison
RAG_STD_DIR=results/rag_domain_specific/standard

echo "=== Evaluating SMILES ==="

uv run python src/eval_smiles.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_smiles" \
    --grammar_file "${PRED_DIR}/smiles_test_k64.json" \
    --test_path data/smiles/test.json \
    --output_path "${RESULT_DIR}/smiles/test.json"

echo "=== Evaluating SPICE ==="

uv run python src/eval_spice.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_spice" \
    --grammar_file "${PRED_DIR}/spice_test_k64.json" \
    --test_path data/spice/test.json \
    --output_path "${RESULT_DIR}/spice/test.json"

echo "=== Evaluating OpenSCAD ==="

uv run python src/eval_openscad.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_openscad" \
    --grammar_file "${PRED_DIR}/openscad_test_k64.json" \
    --test_path data/openscad/test.json \
    --output_path "${RESULT_DIR}/openscad/test.json"

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

echo "=== Plotting standard comparisons (Baseline vs RAG CoT vs Gold Grammar) ==="

uv run python src/plot.py plot_multi_metrics \
    --result_files '["results/smiles/baseline/test.json", "'"${RESULT_DIR}"'/smiles/test.json", "results/smiles/grammar/test.json"]' \
    --metrics '["canonical_exact_match", "validity", "fingerprint_similarity", "bleu"]' \
    --labels '["Baseline", "RAG CoT", "Gold Grammar"]' \
    --metric_labels '{"canonical_exact_match": "Canonical Exact Match", "validity": "Validity", "fingerprint_similarity": "Fingerprint Similarity", "bleu": "BLEU"}' \
    --output_path "${RESULT_DIR}/smiles/comparison.png" \
    --title "SMILES — RAG CoT Grammar Prediction"

uv run python src/plot.py plot_multi_metrics \
    --result_files '["results/spice/baseline/test.json", "'"${RESULT_DIR}"'/spice/test.json", "results/spice/grammar/test.json"]' \
    --metrics '["ged_similarity", "syntax_validity", "exact_match", "bleu", "component_f1"]' \
    --labels '["Baseline", "RAG CoT", "Gold Grammar"]' \
    --metric_labels '{"ged_similarity": "GED Similarity", "syntax_validity": "Syntax Validity", "exact_match": "Exact Match", "bleu": "BLEU", "component_f1": "Component F1"}' \
    --output_path "${RESULT_DIR}/spice/comparison.png" \
    --title "SPICE — RAG CoT Grammar Prediction"

uv run python src/plot.py plot_multi_metrics \
    --result_files '["results/openscad/baseline/test.json", "'"${RESULT_DIR}"'/openscad/test.json", "results/openscad/grammar/test.json"]' \
    --metrics '["iou", "syntax_validity", "bleu"]' \
    --labels '["Baseline", "RAG CoT", "Gold Grammar"]' \
    --metric_labels '{"iou": "Volumetric IoU", "syntax_validity": "Syntax Validity", "bleu": "BLEU"}' \
    --output_path "${RESULT_DIR}/openscad/comparison.png" \
    --title "OpenSCAD — RAG CoT Grammar Prediction"

uv run python src/plot.py plot_pass_at_k \
    --result_files '["results/verilog/baseline.json", "'"${RESULT_DIR}"'/verilog/test.json", "results/verilog/grammar.json"]' \
    --labels '["Baseline", "RAG CoT", "Gold Grammar"]' \
    --output_path "${RESULT_DIR}/verilog/pass_at_k.png" \
    --title "Verilog — RAG CoT Grammar Prediction"

echo "=== Plotting RAG comparison (RAG vs RAG CoT) ==="

uv run python src/plot.py plot_multi_metrics \
    --result_files '["'"${RAG_STD_DIR}"'/smiles/test.json", "'"${RESULT_DIR}"'/smiles/test.json"]' \
    --metrics '["canonical_exact_match", "validity", "fingerprint_similarity", "bleu"]' \
    --labels '["RAG", "RAG CoT"]' \
    --metric_labels '{"canonical_exact_match": "Canonical Exact Match", "validity": "Validity", "fingerprint_similarity": "Fingerprint Similarity", "bleu": "BLEU"}' \
    --output_path "${RAG_COMP_DIR}/smiles/comparison.png" \
    --title "SMILES — RAG vs RAG CoT"

uv run python src/plot.py plot_multi_metrics \
    --result_files '["'"${RAG_STD_DIR}"'/spice/test.json", "'"${RESULT_DIR}"'/spice/test.json"]' \
    --metrics '["ged_similarity", "syntax_validity", "exact_match", "bleu", "component_f1"]' \
    --labels '["RAG", "RAG CoT"]' \
    --metric_labels '{"ged_similarity": "GED Similarity", "syntax_validity": "Syntax Validity", "exact_match": "Exact Match", "bleu": "BLEU", "component_f1": "Component F1"}' \
    --output_path "${RAG_COMP_DIR}/spice/comparison.png" \
    --title "SPICE — RAG vs RAG CoT"

uv run python src/plot.py plot_multi_metrics \
    --result_files '["'"${RAG_STD_DIR}"'/openscad/test.json", "'"${RESULT_DIR}"'/openscad/test.json"]' \
    --metrics '["iou", "syntax_validity", "bleu"]' \
    --labels '["RAG", "RAG CoT"]' \
    --metric_labels '{"iou": "Volumetric IoU", "syntax_validity": "Syntax Validity", "bleu": "BLEU"}' \
    --output_path "${RAG_COMP_DIR}/openscad/comparison.png" \
    --title "OpenSCAD — RAG vs RAG CoT"

uv run python src/plot.py plot_pass_at_k \
    --result_files '["'"${RAG_STD_DIR}"'/verilog/test.json", "'"${RESULT_DIR}"'/verilog/test.json"]' \
    --labels '["RAG", "RAG CoT"]' \
    --output_path "${RAG_COMP_DIR}/verilog/pass_at_k.png" \
    --title "Verilog — RAG vs RAG CoT"

echo "All evaluations complete."
