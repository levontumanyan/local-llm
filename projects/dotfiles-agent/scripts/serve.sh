#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?MODEL is not set}"
: "${PORT:?PORT is not set}"
: "${MAX_TOKENS:?MAX_TOKENS is not set}"
: "${PROMPT_CACHE_SIZE:?PROMPT_CACHE_SIZE is not set}"

THINKING_ARGS=""
[ "${ENABLE_THINKING:-1}" = "0" ] && THINKING_ARGS='--chat-template-args {"enable_thinking":false}'

echo "model          : $MODEL"
echo "port           : $PORT"
echo "max tokens     : $MAX_TOKENS"
echo "prompt cache   : $PROMPT_CACHE_SIZE"
echo "thinking       : ${ENABLE_THINKING:-1}"
echo "url            : http://localhost:$PORT/v1"
echo ""

# shellcheck disable=SC2086
uv run mlx_lm.server \
    --model "$MODEL" \
    --port "$PORT" \
    --max-tokens "$MAX_TOKENS" \
    --prompt-cache-size "$PROMPT_CACHE_SIZE" \
    $THINKING_ARGS
