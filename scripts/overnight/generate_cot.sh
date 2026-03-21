#!/usr/bin/env bash
set -euo pipefail

uv run python src/generate_cot.py run \
    --input_path data/overnight/train.json \
    --output_path data/overnight/train_cot.json \
    --grammar_path grammars/overnight_blocks.lark \
    --cache_path cache/cot_overnight_cache.json \
    --task_name cot_overnight \
    --domain_description "a Lambda DCS language for querying block objects with spatial relationships (left, right, above, below) and properties (shape, color, length, width, height)" \
    --mode batch \
    "$@"
