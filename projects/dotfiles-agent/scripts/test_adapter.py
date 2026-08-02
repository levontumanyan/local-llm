#!/usr/bin/env python3
"""Smoke-test the fine-tuned adapter against a few prompts, with the real chat
template + tool schemas. Mirrors main.py's generation path."""

from __future__ import annotations

from pathlib import Path

from mlx_lm import generate, load

from dotfiles_agent.config import (
	ADAPTER_PATH,
	MAX_TOKENS,
	MODEL_PATH,
	system_prompt,
)
from dotfiles_agent.tools import load_tools

# Test prompts stay short; clamp tokens so the smoke test is fast.
TEST_MAX_TOKENS = min(int(__import__("os").getenv("TEST_MAX_TOKENS", "300")), MAX_TOKENS)

PROMPTS = [
	"Write a guard that sources a file only if it exists",
	"Write python and pip aliases that point to python3/pip3",
	"Add a zsh history options block with deduplication",
]


def main() -> None:
	tools = load_tools()
	print(f"Loading {MODEL_PATH} (adapter: {ADAPTER_PATH})...\n")
	model, tokenizer = load(MODEL_PATH, adapter_path=ADAPTER_PATH)

	for prompt in PROMPTS:
		print(f"PROMPT: {prompt}")
		print("-" * 60)
		messages = [
			{"role": "system", "content": system_prompt()},
			{"role": "user", "content": prompt},
		]
		text = tokenizer.apply_chat_template(
			messages,
			tools=tools,
			add_generation_prompt=True,
			tokenize=False,
		)
		response = generate(model, tokenizer, prompt=text, max_tokens=TEST_MAX_TOKENS, verbose=False)
		print(response)
		print()


if __name__ == "__main__":
	main()
