#!/usr/bin/env bash
set -euo pipefail

uv run python src/generate_cot.py run \
    --input_path data/geoquery/train.json \
    --output_path data/geoquery/train_cot.json \
    --grammar_path grammars/geoquery.lark \
    --cache_path cache/cot_geoquery_cache.json \
    --task_name cot_geoquery \
    --domain_description "a FunQL geographic query language for querying US geography facts (states, cities, rivers, mountains, populations, areas)" \
    --mode batch \
    "$@"
