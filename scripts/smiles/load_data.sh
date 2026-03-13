#!/usr/bin/env bash
set -euo pipefail

uv run python src/load_smiles.py load
