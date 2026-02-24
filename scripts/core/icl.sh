#!/usr/bin/env bash
set -euo pipefail

uv run python src/icl.py --mode standard
uv run python src/icl.py --mode oracle
