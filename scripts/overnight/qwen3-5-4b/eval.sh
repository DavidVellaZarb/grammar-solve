#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/../eval.sh" "Qwen/Qwen3.5-4B" "qwen3-5-4b" "$@"
