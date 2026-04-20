#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/../eval.sh" "google/gemma-4-31B-it" "gemma-4-31b" "$@"
