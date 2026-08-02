#!/usr/bin/env bash
set -euo pipefail

# Parallel authenticated download of the two big Fable-Therapy MLX shards.
# Uses curl --retry-all-errors so CDN connection drops auto-resume via -C -.
# Token passed via -H (kept out of process args would need -K; kept simple here
# since the token already lives in ~/.cache/huggingface/token on this machine).

MODEL_DIR="${1:-$HOME/repos/local-llm/coach-model}"
REPO="mlx-community/Qwen3.5-9B-Fable-5-v1-oQ4"
BASE="https://huggingface.co/$REPO/resolve/main"
TOKEN_FILE="$HOME/.cache/huggingface/token"

mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

if [ ! -s "$TOKEN_FILE" ]; then
	echo "error: no HF token at $TOKEN_FILE — run: hf auth login" >&2
	exit 1
fi
TOKEN="$(cat "$TOKEN_FILE")"
AUTH="Authorization: Bearer $TOKEN"

SHARD1="model-00001-of-00002.safetensors"  # 5.0 GB
SHARD2="model-00002-of-00002.safetensors"  # 1.0 GB

echo "downloading both shards in parallel (auth + auto-retry on drops)..."
echo "  $SHARD1 (5.0 GB)"
echo "  $SHARD2 (1.0 GB)"
echo ""

curl -L --fail -C - --retry 999 --retry-all-errors --retry-delay 5 \
	-H "$AUTH" -o "$SHARD1" "$BASE/$SHARD1" 2>&1 | sed 's/^/[shard1] /' &
PID1=$!

curl -L --fail -C - --retry 999 --retry-all-errors --retry-delay 5 \
	-H "$AUTH" -o "$SHARD2" "$BASE/$SHARD2" 2>&1 | sed 's/^/[shard2] /' &
PID2=$!

wait $PID1 $PID2
echo ""
echo "=== done ==="
ls -lh "$SHARD1" "$SHARD2"
