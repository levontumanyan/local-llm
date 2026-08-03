"""Tests for dotfiles_agent.tools — the tool-call parser, executor, and
message builders. These are the bug-fix hotspots:

- parse_tool_calls: balanced-brace scan (regression for nested } in content)
- build_tool_result_message: must use tool_call_id, not name
- execute_tool: never raises; every failure returns an error string
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dotfiles_agent.tools import (
	_build_open,
	_build_close,
	build_assistant_tool_message,
	build_tool_result_message,
	execute_tool,
	load_tools,
	parse_tool_calls,
)

# The literal tags are assembled via chr() to keep them out of the source (and
# out of this test file) so model-output parsing in the harness isn't confused.
OPEN = _build_open()
CLOSE = _build_close()


def _wrap(body: str) -> str:
	return OPEN + body + CLOSE


# ─────────────────────────────────────────────────────────────────────────────
# load_tools
# ─────────────────────────────────────────────────────────────────────────────
class TestLoadTools:
	def test_returns_list_of_function_schemas(self):
		tools = load_tools()
		assert isinstance(tools, list)
		assert len(tools) >= 7
		for t in tools:
			assert t["type"] == "function"
			assert "name" in t["function"]
			assert "parameters" in t["function"]

	def test_expected_tool_names_present(self):
		names = {t["function"]["name"] for t in load_tools()}
		assert names >= {
			"read_file", "write_file", "edit_file", "append_to_file",
			"list_files", "run_command", "check_command_exists",
		}


# ─────────────────────────────────────────────────────────────────────────────
# parse_tool_calls — the regression-critical parser
# ─────────────────────────────────────────────────────────────────────────────
class TestParseToolCalls:
	def test_single_call(self):
		s = _wrap(json.dumps({"name": "read_file", "arguments": {"path": "~/.zshrc"}}))
		calls = parse_tool_calls(s)
		assert len(calls) == 1
		assert calls[0]["name"] == "read_file"
		assert calls[0]["arguments"]["path"] == "~/.zshrc"

	def test_plain_reply_no_calls(self):
		assert parse_tool_calls("just a normal answer with no tags") == []

	def test_malformed_json_dropped_not_raised(self):
		# bad JSON inside the tags must be skipped, never raise
		assert parse_tool_calls(_wrap("{not valid json}")) == []

	def test_multiple_calls_in_one_block(self):
		body = json.dumps({"name": "read_file", "arguments": {"path": "a"}}) + \
		       json.dumps({"name": "read_file", "arguments": {"path": "b"}})
		calls = parse_tool_calls(_wrap(body))
		assert len(calls) == 2
		assert calls[0]["arguments"]["path"] == "a"
		assert calls[1]["arguments"]["path"] == "b"

	def test_nested_brace_in_string_not_truncated(self):
		"""Regression: the old non-greedy regex stopped at the first }, truncating
		any tool call whose arguments contained a nested object or a literal }."""
		args = {"path": "~/.zshrc", "content": "x = {a: 1}\nexport FOO=bar}"}
		s = _wrap(json.dumps({"name": "write_file", "arguments": args}))
		calls = parse_tool_calls(s)
		assert len(calls) == 1
		assert calls[0]["arguments"]["content"] == args["content"]

	def test_nested_json_object_in_arguments(self):
		"""Arguments value is itself a JSON object with nested braces."""
		args = {"path": "~/.config.json", "content": json.dumps({"key": {"nested": True}})}
		s = _wrap(json.dumps({"name": "write_file", "arguments": args}))
		calls = parse_tool_calls(s)
		assert len(calls) == 1
		assert calls[0]["arguments"]["content"] == args["content"]

	def test_escaped_quote_in_string(self):
		"""A \" inside a string value must not confuse the brace scanner."""
		args = {"path": "~/.zshrc", "content": 'alias x="hello \"world\""'}
		s = _wrap(json.dumps({"name": "write_file", "arguments": args}))
		calls = parse_tool_calls(s)
		assert len(calls) == 1
		assert calls[0]["arguments"]["content"] == args["content"]

	def test_args_alias_accepted(self):
		"""Some models emit 'args' instead of 'arguments' — both are accepted."""
		s = _wrap(json.dumps({"name": "read_file", "args": {"path": "~/.zshrc"}}))
		calls = parse_tool_calls(s)
		assert len(calls) == 1
		assert calls[0]["name"] == "read_file"

	def test_text_around_tags_ignored(self):
		s = "Here is what I'll do:\n" + _wrap(json.dumps({"name": "read_file", "arguments": {"path": "a"}})) + "\nDone."
		calls = parse_tool_calls(s)
		assert len(calls) == 1

	def test_multiple_separate_blocks(self):
		s = _wrap(json.dumps({"name": "read_file", "arguments": {"path": "a"}})) + \
		    "thinking..." + \
		    _wrap(json.dumps({"name": "read_file", "arguments": {"path": "b"}}))
		calls = parse_tool_calls(s)
		assert len(calls) == 2


# ─────────────────────────────────────────────────────────────────────────────
# build_assistant_tool_message / build_tool_result_message
# ─────────────────────────────────────────────────────────────────────────────
class TestMessageBuilders:
	def test_assistant_message_shape_matches_training(self):
		"""Mirrors scripts/prepare.py: content empty, structured tool_calls."""
		calls = [{"name": "write_file", "arguments": {"path": "a", "content": "b"}}]
		msg = build_assistant_tool_message(calls)
		assert msg["role"] == "assistant"
		assert msg["content"] == ""
		assert len(msg["tool_calls"]) == 1
		tc = msg["tool_calls"][0]
		assert tc["id"] == "call_0"
		assert tc["type"] == "function"
		assert tc["function"]["name"] == "write_file"
		# arguments serialized to a JSON string (not left as a dict)
		assert isinstance(tc["function"]["arguments"], str)
		assert json.loads(tc["function"]["arguments"]) == {"path": "a", "content": "b"}

	def test_assistant_message_preserves_existing_id(self):
		calls = [{"id": "abc-123", "name": "read_file", "arguments": {"path": "x"}}]
		msg = build_assistant_tool_message(calls)
		assert msg["tool_calls"][0]["id"] == "abc-123"

	def test_assistant_message_args_as_string_passthrough(self):
		"""If arguments is already a JSON string, it's passed through unchanged."""
		calls = [{"name": "read_file", "arguments": '{"path": "x"}'}]
		msg = build_assistant_tool_message(calls)
		assert msg["tool_calls"][0]["function"]["arguments"] == '{"path": "x"}'

	def test_tool_result_uses_tool_call_id(self):
		"""Regression: the old loop sent {'role':'tool','name':...} which the
		Qwen3 chat template does not bind to the preceding call."""
		msg = build_tool_result_message("call_0", "ok")
		assert msg["role"] == "tool"
		assert msg["tool_call_id"] == "call_0"
		assert msg["content"] == "ok"
		# must NOT have a 'name' key — the template keys off tool_call_id
		assert "name" not in msg

	def test_ids_round_trip_between_calls(self):
		"""The id on the assistant tool_call must be the same id on the tool result
		so the chat template can pair them."""
		calls = [{"name": "read_file", "arguments": {"path": "x"}}]
		assistant = build_assistant_tool_message(calls)
		cid = assistant["tool_calls"][0]["id"]
		result = build_tool_result_message(cid, "file contents")
		assert result["tool_call_id"] == assistant["tool_calls"][0]["id"]


# ─────────────────────────────────────────────────────────────────────────────
# execute_tool — every branch, never raises
# ─────────────────────────────────────────────────────────────────────────────
class TestExecuteTool:
	def test_unknown_tool(self):
		assert execute_tool("nope", {}) == "Unknown tool: nope"

	def test_missing_argument(self):
		assert "Error: missing argument" in execute_tool("read_file", {})

	def test_read_file_missing(self, tmp_path):
		r = execute_tool("read_file", {"path": str(tmp_path / "nope.txt")})
		assert "not found" in r

	def test_write_then_read(self, tmp_path):
		f = tmp_path / "out.txt"
		r = execute_tool("write_file", {"path": str(f), "content": "hello"})
		assert "Written 5 bytes" in r
		assert f.read_text() == "hello"
		assert execute_tool("read_file", {"path": str(f)}) == "hello"

	def test_write_creates_parent_dirs(self, tmp_path):
		f = tmp_path / "a" / "b" / "c.txt"
		execute_tool("write_file", {"path": str(f), "content": "x"})
		assert f.read_text() == "x"

	def test_edit_file(self, tmp_path):
		f = tmp_path / "f.txt"
		f.write_text("foo bar baz")
		assert execute_tool("edit_file", {"path": str(f), "old": "bar", "new": "QUX"}) == "Edited successfully"
		assert f.read_text() == "foo QUX baz"

	def test_edit_file_only_first_match(self, tmp_path):
		f = tmp_path / "f.txt"
		f.write_text("a a a")
		execute_tool("edit_file", {"path": str(f), "old": "a", "new": "X"})
		assert f.read_text() == "X a a"

	def test_edit_file_string_not_found(self, tmp_path):
		f = tmp_path / "f.txt"
		f.write_text("hello")
		r = execute_tool("edit_file", {"path": str(f), "old": "zzz", "new": "x"})
		assert "not found" in r

	def test_append_to_file(self, tmp_path):
		f = tmp_path / "f.txt"
		f.write_text("line1")
		execute_tool("append_to_file", {"path": str(f), "content": "line2"})
		assert f.read_text() == "line1\nline2"

	def test_list_files(self, tmp_path):
		(tmp_path / "b.txt").write_text("")
		(tmp_path / "a.txt").write_text("")
		r = execute_tool("list_files", {"dir": str(tmp_path)})
		assert r == "a.txt\nb.txt"

	def test_run_command_success(self):
		assert execute_tool("run_command", {"cmd": "echo hi"}) == "hi"

	def test_run_command_no_output(self):
		assert execute_tool("run_command", {"cmd": "true"}) == "(no output)"

	def test_run_command_stderr_captured(self):
		r = execute_tool("run_command", {"cmd": "echo err 1>&2"})
		assert "err" in r

	def test_check_command_exists_true(self):
		assert execute_tool("check_command_exists", {"command": "echo"}) == "exists"

	def test_check_command_exists_false(self):
		assert execute_tool("check_command_exists", {"command": "this_does_not_exist_xyz"}) == "not found"

	def test_path_expansion(self, tmp_path, monkeypatch):
		"""~ in a path expands to $HOME."""
		monkeypatch.setenv("HOME", str(tmp_path))
		execute_tool("write_file", {"path": "~/test_expansion.txt", "content": "x"})
		assert (tmp_path / "test_expansion.txt").read_text() == "x"
