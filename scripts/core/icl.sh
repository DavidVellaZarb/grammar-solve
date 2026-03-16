#!/usr/bin/env bash
set -euo pipefail

uv run python src/icl.py evaluate --mode standard "$@"
uv run python src/icl.py evaluate --mode oracle "$@"
