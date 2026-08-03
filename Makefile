# local-llm — top-level orchestration.
#
# Three independent local-LLM projects live under projects/<name>/. Each has its
# own code, prompts, and scripts; this Makefile wires them to a single .env and
# exposes a consistent prefixed target namespace:
#
#   dotfiles-*  LoRA fine-tune of Qwen3 + tool-calling agent
#   coach-*     Fable-5 therapy chatbot (rapid-mlx + Open WebUI)
#   josie-*     Josiefied-Qwen2.5-3B abliterated chat model
#
# Model weights + session history stay at the repo root (gitignored) so the
# trackable tree is just code + config. Run `make help` for every target.

-include .env
# Only env-driven vars go here. RAW_DATA/DATA_DIR are passed explicitly in the
# dotfiles-prepare recipe and stay plain make variables with ?= defaults below.
export MODEL PORT MAX_TOKENS PROMPT_CACHE_SIZE ENABLE_THINKING MAX_WORKERS \
       ADAPTER_PATH FUSED_4BIT

# ─────────────────────────────────────────────────────────────────────────────
# Common paths / defaults
# ─────────────────────────────────────────────────────────────────────────────
ROOT       := $(CURDIR)
DOTFILES   := $(ROOT)/projects/dotfiles-agent
COACH      := $(ROOT)/coach-model
JOSIE      := $(ROOT)/josie-model

ADAPTER  ?= $(DOTFILES)/adapters
FUSED    ?= $(DOTFILES)/fused-model
# Default: derive the 4-bit path from FUSED. Override FUSED_4BIT in .env to serve
# a prebuilt community model straight from the HF cache (skips fuse + quantize).
FUSED_4BIT ?= $(FUSED)-4bit
ITERS    ?= 500
DATA_DIR ?= $(ROOT)/data
RAW_DATA ?= $(HOME)/repos/home_directory/training

# ─────────────────────────────────────────────────────────────────────────────
# Dotfiles agent — Qwen3 LoRA fine-tune + tool-calling CLI
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ALIAS := dotfiles-coder

.PHONY: dotfiles-download dotfiles-serve dotfiles-rapid-serve \
        dotfiles-prepare dotfiles-train dotfiles-test-adapter \
        dotfiles-fuse dotfiles-quantize dotfiles-run dotfiles-clean dotfiles-test

dotfiles-download: ## fetch the base model via `hf` (authenticated, parallel)
	uv run hf download $(MODEL) --max-workers $(MAX_WORKERS)

dotfiles-serve: ## serve via mlx_lm.server (uses .env: MODEL/PORT/MAX_TOKENS/...)
	bash $(DOTFILES)/scripts/serve.sh

dotfiles-rapid-serve: ## serve via rapid-mlx with auto tool-choice
	rapid-mlx serve $(FUSED_4BIT) \
		--port $(PORT) \
		--served-model-name $(MODEL_ALIAS) \
		--enable-auto-tool-choice \
		--tool-call-parser qwen

dotfiles-prepare: ## convert raw jsonl (RAW_DATA) -> tool-calling chat data (DATA_DIR)
	cd $(DOTFILES) && RAW_DATA=$(RAW_DATA) DATA_DIR=$(DATA_DIR) uv run python scripts/prepare.py

dotfiles-train: ## LoRA fine-tune (run prepare first) -> adapters/
	cd $(DOTFILES) && uv run mlx_lm.lora \
		--model $(MODEL) --train --data $(DATA_DIR) \
		--adapter-path $(ADAPTER) --iters $(ITERS) \
		--learning-rate 1e-5 --batch-size 2 --val-batches 5

dotfiles-test-adapter: ## smoke-test the adapter with real chat template + tools
	cd $(DOTFILES) && ADAPTER_PATH=$(ADAPTER) uv run python scripts/test_adapter.py

dotfiles-fuse: ## merge adapter into base model (MLX bf16) -> fused-model/
	cd $(DOTFILES) && uv run mlx_lm.fuse --model $(MODEL) --adapter-path $(ADAPTER) --save-path $(FUSED)

dotfiles-quantize: ## 4-bit convert fused bf16 -> $(FUSED)-4bit
	cd $(DOTFILES) && uv run mlx_lm.convert --hf-path $(FUSED) --mlx-path $(FUSED)-4bit -q --q-bits 4

dotfiles-run: ## interactive CLI agent (main.py)
	cd $(DOTFILES) && uv run python main.py

dotfiles-clean: ## remove adapter, fused model, and local data
	rm -rf $(ADAPTER) $(FUSED) $(DATA_DIR) Modelfile

dotfiles-test: ## run the pytest unit tests (no model/GPU needed)
	uv run pytest -v

# ─────────────────────────────────────────────────────────────────────────────
# Coach — Fable-5 therapy model (rapid-mlx) + Open WebUI
# ─────────────────────────────────────────────────────────────────────────────
COACH_DIR        := $(COACH)
COACH_PORT       := 8081
COACH_ALIAS      := coach
COACH_WEBUI_PORT := 3000
COACH_HISTORY    := $(ROOT)/coach-history
# Which rapid-mlx port Open WebUI talks to by default (8081=coach, 8082=josie).
WEBUI_API_PORT ?= $(COACH_PORT)

.PHONY: coach-download coach-serve coach-webui coach-stop-webui coach-clean

coach-download: ## fetch Fable-5 MLX model (parallel, authenticated, resumable)
	bash $(ROOT)/projects/coach/scripts/download.sh $(COACH_DIR)

# no --enable-prefix-cache: ate 2.7GB and OOM'd on long turns; for a growing
# chat the only shared prefix is the system prompt, low hit rate.
# --kv-cache-quantization: halves KV cache memory for long coaching sessions.
coach-serve: ## serve coach via rapid-mlx on $(COACH_PORT)
	rapid-mlx serve $(COACH_DIR) \
		--host 127.0.0.1 --port $(COACH_PORT) \
		--served-model-name $(COACH_ALIAS) \
		--no-mllm --reasoning-parser qwen3 \
		--kv-cache-quantization --pin-system-prompt \
		--default-temperature 0.7 --default-top-p 0.9

coach-webui: ## Open WebUI pointed at the coach API (alias for `make webui`)
	$(MAKE) webui

coach-stop-webui: ## stop the Open WebUI process
	pkill -f "open-webui serve" 2>/dev/null || true

coach-clean: ## stop the webui and delete the local model (history is kept)
	pkill -f "open-webui serve" 2>/dev/null || true
	rm -rf $(COACH_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Josie — Josiefied-Qwen2.5-3B abliterated (rapid-mlx), shares Open WebUI
# ─────────────────────────────────────────────────────────────────────────────
JOSIE_DIR   := $(JOSIE)
JOSIE_PORT  := 8082
JOSIE_ALIAS := josie
JOSIE_REPO  := mlx-community/Josiefied-Qwen2.5-3B-Instruct-abliterated-v1-4-bit

.PHONY: josie-download josie-serve josie-clean

josie-download: ## fetch the Josiefied Qwen2.5-3B MLX 4-bit weights
	hf download $(JOSIE_REPO) --local-dir $(JOSIE_DIR) --max-workers 8

# --max-tokens 4096: Open WebUI often omits max_tokens; rapid-mlx's default
# (32768 = full context) lets prompt+completion exceed the 32k window.
josie-serve: ## serve josie via rapid-mlx on $(JOSIE_PORT)
	rapid-mlx serve $(JOSIE_DIR) \
		--host 127.0.0.1 --port $(JOSIE_PORT) \
		--served-model-name $(JOSIE_ALIAS) \
		--max-tokens 4096 \
		--default-temperature 0.7 --default-top-p 0.9

josie-clean: ## remove the local josie model weights
	rm -rf $(JOSIE_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Shared UI + help
# ─────────────────────────────────────────────────────────────────────────────
# First run: `uv tool install open-webui`. History lives in coach-history/ (shared).
# To use both models in one UI: Admin -> Settings -> Connections -> add
# http://localhost:8081/v1 and http://localhost:8082/v1 (key: local).
.PHONY: webui help clean

webui: ## launch Open WebUI (override API with WEBUI_API_PORT=8082 for josie)
	@command -v open-webui >/dev/null 2>&1 || { echo "open-webui not installed. Run: uv tool install open-webui"; exit 1; }
	OPENAI_API_BASE_URL=http://localhost:$(WEBUI_API_PORT)/v1 \
	OPENAI_API_KEY=local \
	DATA_DIR=$(COACH_HISTORY) \
	exec open-webui serve --port $(COACH_WEBUI_PORT)

help: ## show this help
	@awk 'BEGIN {FS = ":.*##"; printf "targets:\n"} /^[a-zA-Z][a-zA-Z0-9_-]*:.*##/ {printf "  %-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

clean: dotfiles-clean coach-stop-webui ## stop webui + dotfiles-clean
