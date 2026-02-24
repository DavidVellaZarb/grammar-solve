#!/usr/bin/env bash
set -euo pipefail

uv run python src/prune_grammar.py \
    grammars/smcalflow.lark \
    data/smcalflow/train.json \
    data/smcalflow/test.json \
    --dry_run=False \
    "$@"
