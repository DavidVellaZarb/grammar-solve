#!/usr/bin/env bash
set -euo pipefail

DOMAINS=("geoquery" "overnight" "verilog" "openscad" "spice")
LABELS=("GeoQuery" "Overnight" "Verilog" "OpenSCAD" "SPICE")

for domain in "${DOMAINS[@]}"; do
    uv run python src/grammar_health.py analyze \
        --input_path "outputs/predicted_grammars/rag_cot/${domain}_test_k64.json" \
        --output_path "outputs/analysis/grammar_health/${domain}.json"
done

FILES=()
for domain in "${DOMAINS[@]}"; do
    FILES+=("outputs/analysis/grammar_health/${domain}.json")
done

uv run python src/grammar_health.py plot \
    --result_files "[$(printf '"%s",' "${FILES[@]}" | sed 's/,$//')]" \
    --labels "[$(printf '"%s",' "${LABELS[@]}" | sed 's/,$//')]"
