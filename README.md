# local-llm

Local LLMs on Apple Silicon via MLX. Two setups live here: a private
life-coach chatbot (the main one) and a dotfiles tool-calling fine-tune.

Requirements: [uv](https://docs.astral.sh/uv/), `rapid-mlx` and the `hf` CLI
(`uv tool install rapid-mlx huggingface-hub`), and a Hugging Face token
(`hf auth login`).

# Coach (therapy model)

A personal life-coach/mentor chatbot. Fully local: weights, chat history, and
accounts never leave the machine.

- **Model**: [`mlx-community/Qwen3.5-9B-Fable-5-v1-oQ4`](https://huggingface.co/mlx-community/Qwen3.5-9B-Fable-5-v1-oQ4) —
  Fable-5, a therapy-tuned Qwen3.5-9B, 4-bit oQ mixed-precision for MLX
  (~6 GB on disk), stored in `coach-model/`. 262k context.
- **Server**: `rapid-mlx` serves an OpenAI-compatible API on
  `http://127.0.0.1:8081/v1` under the alias `coach`.
- **UI**: Open WebUI (pip install, no container) on `http://localhost:3000`,
  pointed at the rapid-mlx API. Accounts and chat history live in
  `coach-history/` (pinned via `DATA_DIR`, gitignored).
- **Persona**: `prompts/coach.md` — disposition, cross-session memory
  behavior, style, and crisis boundaries (988/911 fallback). It is **not**
  auto-loaded; set it as the system prompt in Open WebUI (per-chat or a saved
  model preset). The server's `--pin-system-prompt` then caches that prefix.

## Setup

```sh
make coach-download          # fetches both model shards into coach-model/
uv tool install open-webui   # one time
```

## Use

```sh
make coach-serve   # terminal 1: model API on :8081
make coach-webui   # terminal 2: chat UI on :3000
```

Open http://localhost:3000, create a local account (first run), select the
`coach` model, paste `prompts/coach.md` as the system prompt.

Without the UI, straight against the API:

```sh
curl -s http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "$(jq -n --arg sys "$(cat prompts/coach.md)" \
    '{model:"coach", messages:[{role:"system",content:$sys},{role:"user",content:"I cannot focus lately."}]}')" \
  | jq -r '.choices[0].message.content'
```

## Serving notes

`make coach-serve` flags, and why:

- `--kv-cache-quantization` — halves KV cache memory so long sessions don't
  build GPU pressure.
- No `--enable-prefix-cache` — it ate 2.7 GB and OOM'd on long turns; in a
  growing chat only the system prompt is a shared prefix, so hit rate is low.
- `--reasoning-parser qwen3` — parses the model's think tags out of replies.
- `--default-temperature 0.7 --default-top-p 0.9` — coaching defaults.

`make coach-clean` stops the UI and deletes the model; `coach-history/` is
kept. `make coach-stop-webui` stops only the UI.

# Josie (abliterated Qwen2.5-3B)

Small uncensored chat model for quick local testing.

- **Model**: [`mlx-community/Josiefied-Qwen2.5-3B-Instruct-abliterated-v1-4-bit`](https://huggingface.co/mlx-community/Josiefied-Qwen2.5-3B-Instruct-abliterated-v1-4-bit)
  (~1.6 GB), stored in `josie-model/`.
- **Server**: `rapid-mlx` on `http://127.0.0.1:8082/v1` under the alias `josie`.
- **UI**: same Open WebUI as coach (`make webui`).

```sh
make josie-download          # already done if josie-model/ exists
make josie-serve             # terminal 1: API on :8082
make webui WEBUI_API_PORT=8082   # terminal 2: UI on :3000
```

Or keep one WebUI and add both backends under Admin → Settings → Connections:
`http://localhost:8081/v1` (coach) and `http://localhost:8082/v1` (josie),
API key `local`. Then pick `coach` or `josie` in the model dropdown (only the
one currently served will answer).

# Dotfiles agent

LoRA fine-tune of Qwen3 that edits dotfiles via tool calls
(`tools/dotfiles.json`). Config comes from `.env` (`MODEL`, `PORT`,
`MAX_TOKENS`, ...).

```sh
make prepare        # raw jsonl (~/repos/home_directory/training) → data/
make train          # LoRA fine-tune → adapters/
make test-adapter   # smoke-test the adapter with tool schemas
make fuse           # merge adapter into base model → fused-model/
make quantize       # 4-bit convert → fused-model-4bit/
make serve          # serve via mlx_lm.server (uses .env)
make rapid-serve    # serve via rapid-mlx with auto tool-choice
make run            # interactive CLI agent (main.py)
```
