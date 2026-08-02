-include .env
export MODEL PORT MAX_TOKENS PROMPT_CACHE_SIZE ENABLE_THINKING MAX_WORKERS

ADAPTER    := ./adapters
FUSED      := ./fused-model
FUSED_4BIT ?= ./fused-model-4bit
OLLAMA_TAG  := dotfiles-coder
MODEL_ALIAS := dotfiles-coder
ITERS      := 500
RAW_DATA   := $(HOME)/repos/home_directory/training
DATA_DIR   := ./data

.PHONY: download serve rapid-serve prepare train test-adapter fuse import clean coach-download coach-serve coach-webui coach-stop-webui coach-clean josie-download josie-serve webui

download:
	uv run hf download $(MODEL) --max-workers $(MAX_WORKERS)

serve:
	bash scripts/serve.sh

## rapid-serve: serve the model via Rapid-MLX (parsers are auto-detected per model family)
rapid-serve:
	rapid-mlx serve $(FUSED_4BIT) \
		--port $(PORT) \
		--served-model-name $(MODEL_ALIAS) \
		--enable-auto-tool-choice \
		--tool-call-parser qwen

## prepare: convert raw dotfiles data to tool-calling chat format
prepare:
	RAW_DATA=$(RAW_DATA) DATA_DIR=$(DATA_DIR) uv run python scripts/prepare.py

## train: run LoRA fine-tuning (run prepare first)
train:
	uv run mlx_lm.lora \
		--model $(MODEL) \
		--train \
		--data $(DATA_DIR) \
		--adapter-path $(ADAPTER) \
		--iters $(ITERS) \
		--learning-rate 1e-5 \
		--batch-size 2 \
		--val-batches 5

## run: start the interactive dotfiles agent
run:
	uv run python main.py

## test-adapter: test the adapter with proper chat template + tool schemas
test-adapter:
	ADAPTER_PATH=$(ADAPTER) uv run python scripts/test_adapter.py

## fuse: merge adapter into base model (MLX bf16 format)
fuse:
	uv run mlx_lm.fuse \
		--model $(MODEL) \
		--adapter-path $(ADAPTER) \
		--save-path $(FUSED)

## quantize: convert fused bf16 model to 4-bit for fast serving
quantize:
	uv run mlx_lm.convert \
		--hf-path $(FUSED) \
		--mlx-path $(FUSED)-4bit \
		-q --q-bits 4

## clean: remove adapter, fused model, and local data copy
clean:
	rm -rf $(ADAPTER) $(FUSED) $(DATA_DIR) Modelfile

# ─────────────────────────────────────────────────────────────────────────────
# Coach profile — Fable-Therapy-9B (MLX, 4-bit oQ4) served via rapid-mlx
# Personal life-coach / mentor. Chat UI = Open WebUI (pip, no container) → rapid-mlx API.
# ─────────────────────────────────────────────────────────────────────────────

COACH_DIR     := ./coach-model
COACH_PORT    := 8081
COACH_ALIAS   := coach
COACH_WEBUI_PORT := 3000
# Open WebUI data (accounts + chat history). Pinned here so `uv tool upgrade`
# can't wipe it — the default location is inside the uv tool's site-packages.
COACH_HISTORY := $(CURDIR)/coach-history
# Which rapid-mlx port Open WebUI talks to by default (8081=coach, 8082=josie).
WEBUI_API_PORT ?= $(COACH_PORT)

.PHONY: coach-download coach-serve coach-webui coach-stop-webui coach-clean

## coach-download: fetch the Fable-Therapy-9B MLX model (parallel, authenticated)
coach-download:
	bash scripts/coach-download.sh $(COACH_DIR)

## coach-serve: serve the coach via rapid-mlx on $(COACH_PORT) (OpenAI-compatible API)
## Notes:
##   - no --enable-prefix-cache: it ate 2.7GB and caused GPU OOM on long turns;
##     for a growing chat the only shared prefix is the system prompt, low hit rate.
##   - --kv-cache-quantization: halves KV cache memory so long coaching sessions
##     don't accumulate GPU pressure.
coach-serve:
	rapid-mlx serve $(COACH_DIR) \
		--host 127.0.0.1 \
		--port $(COACH_PORT) \
		--served-model-name $(COACH_ALIAS) \
		--no-mllm \
		--reasoning-parser qwen3 \
		--kv-cache-quantization \
		--pin-system-prompt \
		--default-temperature 0.7 \
		--default-top-p 0.9

## coach-webui: Open WebUI pointed at the coach API (alias for `make webui`)
coach-webui: webui

## webui: launch Open WebUI (pip). Override API with WEBUI_API_PORT=8082 for josie.
## First run: `uv tool install open-webui`. History lives in coach-history/ (shared).
## To use both models in one UI: Admin → Settings → Connections → add
## http://localhost:8081/v1 and http://localhost:8082/v1 (key: local).
webui:
	@command -v open-webui >/dev/null 2>&1 || { echo "open-webui not installed. Run: uv tool install open-webui"; exit 1; }
	OPENAI_API_BASE_URL=http://localhost:$(WEBUI_API_PORT)/v1 \
	OPENAI_API_KEY=local \
	DATA_DIR=$(COACH_HISTORY) \
	exec open-webui serve --port $(COACH_WEBUI_PORT)

## coach-stop-webui: stop the Open WebUI process
coach-stop-webui:
	pkill -f "open-webui serve" 2>/dev/null || true

## coach-clean: remove the local model and stop the webui
coach-clean:
	pkill -f "open-webui serve" 2>/dev/null || true
	rm -rf $(COACH_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Josie profile — Josiefied-Qwen2.5-3B abliterated (MLX 4-bit) via rapid-mlx
# Small uncensored chat model. Same Open WebUI as coach (`make webui`).
# ─────────────────────────────────────────────────────────────────────────────

JOSIE_DIR   := ./josie-model
JOSIE_PORT  := 8082
JOSIE_ALIAS := josie
JOSIE_REPO  := mlx-community/Josiefied-Qwen2.5-3B-Instruct-abliterated-v1-4-bit

.PHONY: josie-download josie-serve josie-clean

## josie-download: fetch the Josiefied Qwen2.5-3B MLX 4-bit weights
josie-download:
	hf download $(JOSIE_REPO) --local-dir $(JOSIE_DIR) --max-workers 8

## josie-serve: serve josie via rapid-mlx on $(JOSIE_PORT)
## --max-tokens 4096: Open WebUI often omits max_tokens; rapid-mlx's default
## is 32768 (= full context), so prompt+completion exceeds the 32k window.
josie-serve:
	rapid-mlx serve $(JOSIE_DIR) \
		--host 127.0.0.1 \
		--port $(JOSIE_PORT) \
		--served-model-name $(JOSIE_ALIAS) \
		--max-tokens 4096 \
		--default-temperature 0.7 \
		--default-top-p 0.9

## josie-clean: remove the local josie model weights
josie-clean:
	rm -rf $(JOSIE_DIR)
