"""Tool schema loading, execution, and parsing of model-emitted tool calls.

The parser uses balanced-brace scanning instead of a non-greedy regex so that
tool-call arguments containing nested JSON (or a literal `}` inside file
content) are no longer truncated.

Assistant/tool messages are built in the exact shape the training data uses
(scripts/prepare.py): structured `tool_calls` on the assistant turn and
`tool_call_id` on the tool-result turn. The previous loop sent `{"name": ...}`
on tool results, which the Qwen3 chat template does not associate with the
preceding call — that mismatch is fixed here.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .config import TOOLS_PATH

# Delimiters the Qwen3 chat template wraps tool calls in (the literal tags are
# assembled via chr() so the source stays free of magic bytes). We capture the
# body between them and then extract complete JSON objects with a brace scan.
_OPEN = chr(60) + "tool_call" + chr(62)
_CLOSE = chr(60) + "/tool_call" + chr(62)
_TOOL_CALL_RE = re.compile(re.escape(_OPEN) + r"(.*?)" + re.escape(_CLOSE), re.DOTALL)


def load_tools() -> list[dict]:
	"""Load the OpenAI-style tool schema used for chat-template rendering."""
	return json.loads(TOOLS_PATH.read_text())


def execute_tool(name: str, args: dict[str, Any]) -> str:
	"""Run a tool by name. All paths support `~` expansion. Never raises —
	errors come back as strings so the model can react to them."""
	try:
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
				result = subprocess.run(
					args["cmd"], shell=True, capture_output=True,
					text=True, timeout=30,
				)
				out = (result.stdout + result.stderr).strip()
				return out or "(no output)"

			case "check_command_exists":
				r = subprocess.run(
					f"command -v {args['command']}", shell=True, capture_output=True,
				)
				return "exists" if r.returncode == 0 else "not found"

			case _:
				return f"Unknown tool: {name}"
	except subprocess.TimeoutExpired:
		return "Error: command timed out after 30s"
	except KeyError as e:
		return f"Error: missing argument {e}"
	except Exception as e:  # noqa: BLE001 — surface every failure to the model
		return f"Error: {type(e).__name__}: {e}"


def _extract_json_objects(s: str) -> list[dict]:
	"""Pull every top-level `{...}` object out of `s` using a brace scan that
	respects string literals and escapes. Nested objects stay intact."""
	objs: list[dict] = []
	i, n = 0, len(s)
	while i < n:
		if s[i] != "{":
			i += 1
			continue
		depth = 0
		start = i
		in_str = False
		esc = False
		while i < n:
			c = s[i]
			if in_str:
				if esc:
					esc = False
				elif c == "\\":
					esc = True
				elif c == '"':
					in_str = False
			else:
				if c == '"':
					in_str = True
				elif c == "{":
					depth += 1
				elif c == "}":
					depth -= 1
					if depth == 0:
						try:
							objs.append(json.loads(s[start : i + 1]))
						except json.JSONDecodeError:
							pass
						i += 1
						break
			i += 1
	return objs


def parse_tool_calls(text: str) -> list[dict]:
	"""Parse tool calls the model emitted. Each dict has at least `name` and
	`arguments` (or `args`). Returns [] when the model produced a plain reply."""
	calls: list[dict] = []
	for m in _TOOL_CALL_RE.finditer(text):
		for obj in _extract_json_objects(m.group(1)):
			if "name" in obj:
				calls.append(obj)
	return calls


def build_assistant_tool_message(calls: list[dict]) -> dict:
	"""Reconstruct an assistant turn carrying structured tool_calls, mirroring
	the training format (content empty, arguments as a JSON string)."""
	tool_calls = []
	for i, call in enumerate(calls):
		cid = call.get("id", f"call_{i}")
		name = call["name"]
		args = call.get("arguments", call.get("args", {}))
		args_str = args if isinstance(args, str) else json.dumps(args)
		tool_calls.append({
			"id": cid,
			"type": "function",
			"function": {"name": name, "arguments": args_str},
		})
	return {"role": "assistant", "content": "", "tool_calls": tool_calls}


def build_tool_result_message(call_id: str, result: str) -> dict:
	"""Tool-result turn. Uses `tool_call_id` (not `name`) so the chat template
	binds it to the preceding assistant tool_call."""
	return {"role": "tool", "tool_call_id": call_id, "content": result}
