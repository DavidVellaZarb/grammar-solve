#!/usr/bin/env bash
set -euo pipefail

# Requires ngspice for simulation: apt-get install -y ngspice

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
    --metrics '["ged_similarity", "simulation_success", "syntax_validity"]' \
    --labels '["Baseline", "Grammar-Guided (Ours)"]' \
    --metric_labels '{"ged_similarity": "GED Similarity", "simulation_success": "Simulation Success", "syntax_validity": "Syntax Validity"}' \
    --output_path results/spice/comparison.png \
    --title "SPICE Netlist Generation"
