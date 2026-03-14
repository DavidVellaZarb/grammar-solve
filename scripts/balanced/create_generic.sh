#!/usr/bin/env bash
set -euo pipefail

for split in train valid test; do
    uv run python src/grammar_parser.py add_minimal_grammar \
        --input_path "data/smcalflow/${split}_balanced.json" \
        --output_path "data/smcalflow/${split}_balanced_generic.json" \
        --generic
done
