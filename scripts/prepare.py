#!/usr/bin/env python3
"""Convert raw prompt/completion dotfiles data to Qwen3 tool-calling chat format."""

import json
import os
from pathlib import Path

RAW_DIR = Path(os.getenv("RAW_DATA", Path.home() / "repos/home_directory/training"))
OUT_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent.parent / "data"))
TOOLS_PATH = Path(__file__).parent.parent / "tools" / "dotfiles.json"

SYSTEM_PROMPT = (
	"You are a dotfiles assistant. Use the available tools to read, write, and manage dotfiles. "
	"Always use tools to make changes — never just describe what to do."
)

PATH_HINTS = {
	"zsh": "~/.zshrc",
	"bash": "~/.bashrc",
	"git": "~/.gitconfig",
	"tmux": "~/.tmux.conf",
	"vim": "~/.vimrc",
	"nvim": "~/.config/nvim/init.lua",
	"starship": "~/.config/starship.toml",
	"brew": "~/.Brewfile",
	"ssh": "~/.ssh/config",
}


def infer_path(prompt: str, completion: str) -> str:
	text = (prompt + " " + completion).lower()
	for keyword, path in PATH_HINTS.items():
		if keyword in text:
			return path
	return "~/.zshrc"


def to_tool_call_conversation(row: dict) -> dict:
	prompt = row["prompt"]
	completion = row["completion"]
	path = infer_path(prompt, completion)

	return {
		"messages": [
			{"role": "system", "content": SYSTEM_PROMPT},
			{"role": "user", "content": prompt},
			{
				"role": "assistant",
				"content": "",
				"tool_calls": [{
					"id": "call_0",
					"type": "function",
					"function": {
						"name": "write_file",
						"arguments": json.dumps({"path": path, "content": completion}),
					},
				}],
			},
			{
				"role": "tool",
				"tool_call_id": "call_0",
				"content": f"Written to {path}",
			},
			{
				"role": "assistant",
				"content": f"Done. Written to `{path}`.",
			},
		]
	}


def load_jsonl(path: Path) -> list[dict]:
	return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]):
	path.write_text("\n".join(json.dumps(r) for r in rows))


def main():
	OUT_DIR.mkdir(parents=True, exist_ok=True)

	train_rows = []
	for fname in ["train.jsonl", "dotfiles_training.jsonl"]:
		src = RAW_DIR / fname
		if src.exists():
			rows = load_jsonl(src)
			train_rows.extend(rows)
			print(f"  loaded {len(rows):>3} rows from {fname}")

	train_converted = [to_tool_call_conversation(r) for r in train_rows]
	write_jsonl(OUT_DIR / "train.jsonl", train_converted)
	print(f"train.jsonl → {len(train_converted)} examples")

	valid_src = RAW_DIR / "valid.jsonl"
	if valid_src.exists():
		valid_rows = load_jsonl(valid_src)
		valid_converted = [to_tool_call_conversation(r) for r in valid_rows]
		write_jsonl(OUT_DIR / "valid.jsonl", valid_converted)
		print(f"valid.jsonl → {len(valid_converted)} examples")


if __name__ == "__main__":
	main()
