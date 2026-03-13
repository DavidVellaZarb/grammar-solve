#!/usr/bin/env bash
set -euo pipefail

uv run python src/specialize_grammar.py

uv run python src/eval_specialization.py
