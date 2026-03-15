#!/usr/bin/env bash
set -euo pipefail

# apt-get install -y ngspice

uv run python src/eval_spice.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_spice-baseline" \
    --test_path data/spice/test.json \
    --noinclude_grammar \
    --output_path results/spice/baseline/test.json

uv run python src/eval_spice.py \
    --adapter "${HF_NAMESPACE}/qwen2.5-7b_spice" \
    --test_path data/spice/test.json \
    --include_grammar \
    --output_path results/spice/grammar/test.json

uv run python src/plot.py plot_multi_metrics \
    --result_files '["results/spice/baseline/test.json", "results/spice/grammar/test.json"]' \
    --metrics '["ged_similarity", "syntax_validity", "exact_match", "bleu", "component_f1"]' \
    --labels '["Baseline", "Grammar-Guided (Ours)"]' \
    --metric_labels '{"ged_similarity": "GED Similarity", "syntax_validity": "Syntax Validity", "exact_match": "Exact Match", "bleu": "BLEU", "component_f1": "Component F1"}' \
    --output_path results/spice/comparison.png \
    --title "SPICE Netlist Generation"
