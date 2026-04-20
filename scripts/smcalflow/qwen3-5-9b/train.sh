#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/../train.sh" "Qwen/Qwen3.5-9B" "qwen3-5-9b" "$@"
