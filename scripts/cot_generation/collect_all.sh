#!/usr/bin/env bash
set -euo pipefail

for domain in verilog openscad smiles spice; do
    echo "=== Collecting ${domain} ==="
    uv run python src/generate_cot.py collect --task_name "cot_${domain}" "$@"
    echo
done
