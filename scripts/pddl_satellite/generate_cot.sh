#!/usr/bin/env bash
set -euo pipefail

uv run python src/generate_cot.py run \
    --input_path data/pddl_satellite/train.json \
    --output_path data/pddl_satellite/train_cot.json \
    --grammar_path grammars/pddl_satellite.lark \
    --cache_path cache/cot_pddl_satellite_cache.json \
    --task_name cot_pddl_satellite \
    --domain_description "a Satellite STRIPS planning domain where satellites turn to point at directions, switch on/off instruments, calibrate instruments, and take images of targets in specific modes" \
    --mode batch
