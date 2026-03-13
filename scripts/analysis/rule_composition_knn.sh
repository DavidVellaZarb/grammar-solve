#!/usr/bin/env bash
set -euo pipefail

uv run python src/rule_composition.py analyze_knn --dataset smcalflow
