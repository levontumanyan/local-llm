#!/usr/bin/env python3
"""Convert raw prompt/completion dotfiles data to Qwen3 tool-calling chat format.

Output messages mirror the exact shape the inference loop (main.py) expects:
system -> user -> assistant(tool_calls) -> tool -> assistant. Keeping both in
sync is what lets the fine-tuned adapter generalize to the live agent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Run via `uv run python scripts/prepare.py` from the project dir, so the
# dotfiles_agent package is importable.
from dotfiles_agent.config import PROJECT_DIR, system_prompt

RAW_DIR = Path(os.getenv("RAW_DATA", Path.home() / "repos/home_directory/training"))
OUT_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_DIR.parent.parent / "data")))

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

# Longest keyword first so "nvim" is matched before "vim" (vim is a substring
# of nvim) and "starship" before "ship", etc. Frozen at import for stable order.
_PATH_HINTS_ORDERED = sorted(PATH_HINTS.items(), key=lambda kv: len(kv[0]), reverse=True)


def infer_path(prompt: str, completion: str) -> str:
	text = (prompt + " " + completion).lower()
	for keyword, path in _PATH_HINTS_ORDERED:
		if keyword in text:
			return path
	return "~/.zshrc"


def to_tool_call_conversation(row: dict) -> dict:
	prompt = row["prompt"]
	completion = row["completion"]
	path = infer_path(prompt, completion)

	return {
		"messages": [
			{"role": "system", "content": system_prompt()},
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
			{"role": "tool", "tool_call_id": "call_0", "content": f"Written to {path}"},
			{"role": "assistant", "content": f"Done. Written to `{path}`."},
		]
	}


def load_jsonl(path: Path) -> list[dict]:
	return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
	# trailing newline keeps file diffs and `wc -l` sane
	path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def main() -> None:
	OUT_DIR.mkdir(parents=True, exist_ok=True)

	train_rows: list[dict] = []
	for fname in ["train.jsonl", "dotfiles_training.jsonl"]:
		src = RAW_DIR / fname
		if src.exists():
			rows = load_jsonl(src)
			train_rows.extend(rows)
			print(f"  loaded {len(rows):>3} rows from {fname}")

	train_converted = [to_tool_call_conversation(r) for r in train_rows]
	write_jsonl(OUT_DIR / "train.jsonl", train_converted)
	print(f"train.jsonl -> {len(train_converted)} examples")

	valid_src = RAW_DIR / "valid.jsonl"
	if valid_src.exists():
		valid_converted = [to_tool_call_conversation(r) for r in load_jsonl(valid_src)]
		write_jsonl(OUT_DIR / "valid.jsonl", valid_converted)
		print(f"valid.jsonl -> {len(valid_converted)} examples")


if __name__ == "__main__":
	main()
