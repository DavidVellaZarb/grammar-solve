#!/usr/bin/env bash
set -euo pipefail

for domain in verilog openscad smiles spice; do
    echo "=== ${domain} ==="
    uv run python src/generate_cot.py check --task_name "cot_${domain}"
    echo
done
