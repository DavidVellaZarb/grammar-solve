#!/usr/bin/env bash
set -euo pipefail
MODEL_NAME=${1:-${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}}
MODEL_ALIAS=${2:-${MODEL_ALIAS:-qwen3-4b}}
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
"${REPO_ROOT}/smoke_test/_eval_domain.sh" graphql "${MODEL_NAME}" "${MODEL_ALIAS}" "${MAX_NEW_TOKENS:-512}"

