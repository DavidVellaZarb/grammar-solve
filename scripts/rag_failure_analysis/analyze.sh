#!/usr/bin/env bash
set -euo pipefail
uv run python src/analyze_rag_failures.py analyze_all \
    --output_dir results/rag_failure_analysis "$@"
