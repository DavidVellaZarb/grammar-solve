#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/../train.sh" "Qwen/Qwen2.5-7B-Instruct" "qwen2.5-7b"
