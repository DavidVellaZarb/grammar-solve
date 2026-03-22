#!/usr/bin/env bash
set -euo pipefail

uv run python src/generate_cot.py run \
    --input_path data/pddl_depot/train.json \
    --output_path data/pddl_depot/train_cot.json \
    --grammar_path grammars/pddl_depot.lark \
    --cache_path cache/cot_pddl_depot_cache.json \
    --task_name cot_pddl_depot \
    --domain_description "a Depot STRIPS planning domain where trucks transport crates between depots and distributors using hoists to lift, drop, load, and unload crates onto pallets and trucks" \
    --mode batch
