"""Single source of truth for paths, env-driven config, and the system prompt.

main.py, scripts/prepare.py, and scripts/test_adapter.py all read from here so
the training format and the inference loop can never drift apart.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# projects/dotfiles-agent/
PROJECT_DIR = Path(__file__).resolve().parent.parent
TOOLS_PATH = PROJECT_DIR / "tools" / "dotfiles.json"
PROMPT_PATH = PROJECT_DIR / "prompts" / "system.md"

MODEL_PATH = os.getenv("MODEL", "mlx-community/Qwen3-14B-bf16")
ADAPTER_PATH = os.getenv("ADAPTER_PATH") or None
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))

# Cap on consecutive tool round-trips per user turn. Prevents an infinite loop
# if the model keeps emitting tool calls without resolving the request.
MAX_TOOL_ITERS = int(os.getenv("MAX_TOOL_ITERS", "10"))

_FALLBACK_PROMPT = (
	"You are a dotfiles assistant. Use the available tools to read, write, and "
	"manage dotfiles. Always use tools to make changes — never just describe "
	"what to do."
)


@lru_cache(maxsize=1)
def system_prompt() -> str:
	"""System prompt, loaded from prompts/system.md (cached)."""
	if PROMPT_PATH.exists():
		return PROMPT_PATH.read_text().strip()
	return _FALLBACK_PROMPT
