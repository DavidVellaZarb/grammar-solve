#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/../train.sh" "Qwen/Qwen3-4B-Instruct-2507" "qwen3-4b"
