#!/usr/bin/env python3
"""Test the fine-tuned adapter with proper chat template and tool schemas."""

import json
import os
from pathlib import Path

from mlx_lm import load, generate

TOOLS_PATH = Path(__file__).parent.parent / "tools" / "dotfiles.json"
MODEL_PATH = os.getenv("MODEL", "mlx-community/Qwen3-14B-bf16")
ADAPTER_PATH = os.getenv("ADAPTER_PATH") or None
MAX_TOKENS = 300

SYSTEM_PROMPT = (
	"You are a dotfiles assistant. Use the available tools to read, write, and manage dotfiles. "
	"Always use tools to make changes — never just describe what to do."
)

PROMPTS = [
	"Write a guard that sources a file only if it exists",
	"Write python and pip aliases that point to python3/pip3",
	"Add a zsh history options block with deduplication",
]


def main():
	tools = json.loads(TOOLS_PATH.read_text())
	print(f"Loading {MODEL_PATH} (adapter: {ADAPTER_PATH})...\n")
	model, tokenizer = load(MODEL_PATH, adapter_path=ADAPTER_PATH)

	for prompt in PROMPTS:
		print(f"PROMPT: {prompt}")
		print("-" * 60)
		messages = [
			{"role": "system", "content": SYSTEM_PROMPT},
			{"role": "user", "content": prompt},
		]
		text = tokenizer.apply_chat_template(
			messages,
			tools=tools,
			add_generation_prompt=True,
			tokenize=False,
		)
		response = generate(model, tokenizer, prompt=text, max_tokens=MAX_TOKENS, verbose=False)
		print(response)
		print()


if __name__ == "__main__":
	main()
