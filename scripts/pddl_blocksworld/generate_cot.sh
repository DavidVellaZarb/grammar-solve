#!/usr/bin/env bash
set -euo pipefail

uv run python src/generate_cot.py run \
    --input_path data/pddl_blocksworld/train.json \
    --output_path data/pddl_blocksworld/train_cot.json \
    --grammar_path grammars/pddl_blocksworld.lark \
    --cache_path cache/cot_pddl_blocksworld_cache.json \
    --task_name cot_pddl_blocksworld \
    --domain_description "a Blocksworld STRIPS planning domain where the plan is a sequence of actions (pick-up, put-down, stack, unstack) to rearrange blocks from an initial configuration to a goal configuration" \
    --mode batch
