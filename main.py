import json
import os
import re
import subprocess
from pathlib import Path

from mlx_lm import load, generate

TOOLS_PATH = Path(__file__).parent / "tools" / "dotfiles.json"
MODEL_PATH = os.getenv("MODEL", "mlx-community/Qwen3-14B-bf16")
ADAPTER_PATH = os.getenv("ADAPTER_PATH") or None
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))

SYSTEM_PROMPT = (
	"You are a dotfiles assistant. Use the available tools to read, write, and manage dotfiles. "
	"Always use tools to make changes — never just describe what to do."
)


def load_tools() -> list:
	return json.loads(TOOLS_PATH.read_text())


def execute_tool(name: str, args: dict) -> str:
	match name:
		case "read_file":
			path = Path(args["path"]).expanduser()
			return path.read_text() if path.exists() else f"Error: {path} not found"
		case "write_file":
			path = Path(args["path"]).expanduser()
			path.parent.mkdir(parents=True, exist_ok=True)
			path.write_text(args["content"])
			return f"Written {len(args['content'])} bytes to {path}"
		case "edit_file":
			path = Path(args["path"]).expanduser()
			content = path.read_text()
			if args["old"] not in content:
				return f"Error: string not found in {path}"
			path.write_text(content.replace(args["old"], args["new"], 1))
			return "Edited successfully"
		case "append_to_file":
			path = Path(args["path"]).expanduser()
			with open(path, "a") as f:
				f.write("\n" + args["content"])
			return f"Appended to {path}"
		case "list_files":
			path = Path(args["dir"]).expanduser()
			return "\n".join(p.name for p in sorted(path.iterdir()))
		case "run_command":
			result = subprocess.run(args["cmd"], shell=True, capture_output=True, text=True, timeout=30)
			out = (result.stdout + result.stderr).strip()
			return out or "(no output)"
		case "check_command_exists":
			r = subprocess.run(f"command -v {args['command']}", shell=True, capture_output=True)
			return "exists" if r.returncode == 0 else "not found"
		case _:
			return f"Unknown tool: {name}"


def parse_tool_calls(text: str) -> list[dict]:
	calls = []
	for match in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL):
		try:
			calls.append(json.loads(match.group(1)))
		except json.JSONDecodeError:
			pass
	return calls


def chat(model, tokenizer, messages: list, tools: list) -> str:
	prompt = tokenizer.apply_chat_template(
		messages,
		tools=tools,
		add_generation_prompt=True,
		tokenize=False,
	)
	return generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, verbose=False)


def run():
	tools = load_tools()

	print(f"Loading {MODEL_PATH}...")
	model, tokenizer = load(MODEL_PATH, adapter_path=ADAPTER_PATH)

	messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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

		while True:
			response = chat(model, tokenizer, messages, tools)
			tool_calls = parse_tool_calls(response)

			if not tool_calls:
				print(f"\nagent: {response}\n")
				messages.append({"role": "assistant", "content": response})
				break

			messages.append({"role": "assistant", "content": response})

			for call in tool_calls:
				name = call["name"]
				args = call.get("arguments", call.get("args", {}))
				print(f"  → {name}({json.dumps(args)[:80]})")
				result = execute_tool(name, args)
				messages.append({
					"role": "tool",
					"name": name,
					"content": result,
				})


if __name__ == "__main__":
	run()
