#!/usr/bin/env bash
set -euo pipefail

uv run python src/classifier.py train
