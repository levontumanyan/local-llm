# local-llm

Local LLMs on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). Three
independent projects live under [`projects/`](projects/), each self-contained
with its own code, prompts, and scripts. A single top-level
[`Makefile`](Makefile) wires them to one `.env` and exposes a consistent
prefixed target namespace.

```
local-llm/
├── projects/
│   ├── coach/            Fable-5 therapy chatbot  (prompt + download script)
│   ├── josie/            Josiefied-Qwen2.5-3B abliterated chat model
│   └── dotfiles-agent/  Qwen3 LoRA fine-tune + tool-calling agent
│       ├── dotfiles_agent/   shared config + tool parser/executor
│       ├── main.py           interactive CLI agent
│       ├── prompts/system.md system prompt (single source of truth)
│       ├── tools/dotfiles.json  OpenAI-style tool schema
│       └── scripts/          prepare.py · train · fuse · quantize · test
├── Makefile              top-level orchestration (one target namespace)
├── .env.example          copy to .env and edit (real .env is gitignored)
└── pyproject.toml        shared MLX deps
```

Model weights and session history (`coach-model/`, `josie-model/`,
`coach-history/`, `adapters/`, `fused-model*/`, `data/`) stay at the repo root
and are gitignored — the trackable tree is just code + config.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- `rapid-mlx` and the `hf` CLI: `uv tool install rapid-mlx huggingface-hub`
- A Hugging Face token: `hf auth login`

## Quick start

```sh
cp .env.example .env      # then edit MODEL / PORT / ...
make help                 # list every target

# Dotfiles agent (train + serve + run)
make dotfiles-download && make dotfiles-prepare && make dotfiles-train
make dotfiles-run

# Coach — private therapy chatbot
make coach-download && uv tool install open-webui
make coach-serve          # terminal 1: model API on :8081
make coach-webui          # terminal 2: chat UI on :3000

# Josie — small uncensored chat model
make josie-download && make josie-serve   # API on :8082
```

## Targets

Run `make help` for the live list. The three namespaces:

- **`dotfiles-*`** — `download`, `serve`, `rapid-serve`, `prepare`, `train`,
  `test-adapter`, `fuse`, `quantize`, `run`, `clean`
- **`coach-*`** — `download`, `serve`, `webui`, `stop-webui`, `clean`
- **`josie-*`** — `download`, `serve`, `clean`
- **shared** — `webui` (Open WebUI; `WEBUI_API_PORT=8082` for josie), `help`,
  `clean`

## Project notes

See each project's directory for details. Highlights:

- **Coach** — `mlx-community/Qwen3.5-9B-Fable-5-v1-oQ4` (~6 GB, 4-bit oQ,
  262k context). Served via `rapid-mlx` on `:8081` as `coach`. Persona lives in
  [`projects/coach/prompts/coach.md`](projects/coach/prompts/coach.md) — set it
  as the Open WebUI system prompt (per-chat or a saved model preset); the
  server's `--pin-system-prompt` then caches that prefix.
- **Josie** — `mlx-community/Josiefied-Qwen2.5-3B-Instruct-abliterated-v1-4-bit`
  (~1.6 GB) on `:8082` as `josie`. Shares one Open WebUI with coach.
- **Dotfiles agent** — LoRA fine-tune of Qwen3 that edits dotfiles via tool
  calls. Training format (`scripts/prepare.py`) and the inference loop
  (`main.py` + `dotfiles_agent/`) share one config and one system prompt so
  they can't drift apart.
