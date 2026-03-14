#!/usr/bin/env bash
set -euo pipefail

uv run python src/load_spice.py load
