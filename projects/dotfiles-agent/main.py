#!/usr/bin/env python3
"""Interactive dotfiles agent.

Loads the (optionally adapter-augmented) Qwen3 model, renders the chat
template with the tool schema, and drives a tool-calling REPL. Tool calls are
parsed from the model output, executed locally, fed back, and the loop
continues until the model produces a plain reply or MAX_TOOL_ITERS is hit.
"""

from __future__ import annotations

from mlx_lm import generate, load

from dotfiles_agent.config import (
	ADAPTER_PATH,
	MAX_TOKENS,
	MAX_TOOL_ITERS,
	MODEL_PATH,
	system_prompt,
)
from dotfiles_agent.tools import (
	build_assistant_tool_message,
	build_tool_result_message,
	execute_tool,
	load_tools,
	parse_tool_calls,
)


def chat(model, tokenizer, messages: list[dict], tools: list[dict]) -> str:
	prompt = tokenizer.apply_chat_template(
		messages,
		tools=tools,
		add_generation_prompt=True,
		tokenize=False,
	)
	return generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, verbose=False)


def run() -> None:
	tools = load_tools()

	print(f"Loading {MODEL_PATH}...")
	model, tokenizer = load(MODEL_PATH, adapter_path=ADAPTER_PATH)

	messages: list[dict] = [{"role": "system", "content": system_prompt()}]
	print("Dotfiles agent ready. Ctrl+C to exit.\n")

	while True:
		try:
			user_input = input("you: ").strip()
		except (KeyboardInterrupt, EOFError):
			print("\nbye")
			break
		if not user_input:
			continue

		messages.append({"role": "user", "content": user_input})

		for _ in range(MAX_TOOL_ITERS):
			response = chat(model, tokenizer, messages, tools)
			calls = parse_tool_calls(response)

			if not calls:
				print(f"\nagent: {response}\n")
				messages.append({"role": "assistant", "content": response})
				break

			assistant_msg = build_assistant_tool_message(calls)
			messages.append(assistant_msg)

			for i, call in enumerate(calls):
				name = call["name"]
				args = call.get("arguments", call.get("args", {}))
				cid = call.get("id", f"call_{i}")
				preview = str(args)[:80]
				print(f"  -> {name}({preview})")
				result = execute_tool(name, args)
				messages.append(build_tool_result_message(cid, result))
		else:
			print("\nagent: (hit MAX_TOOL_ITERS without resolving)\n")


if __name__ == "__main__":
	run()
