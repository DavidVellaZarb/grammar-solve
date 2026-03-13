#!/usr/bin/env bash
set -euo pipefail

uv run python src/rule_composition.py analyze --dataset smcalflow
uv run python src/rule_composition.py analyze --dataset verilog
