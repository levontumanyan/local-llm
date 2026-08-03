"""Tests for scripts/prepare.py — the training-data converter.

The critical invariant: to_tool_call_conversation() must emit messages in the
exact shape that build_assistant_tool_message() / build_tool_result_message()
produce at inference time. If these drift, the fine-tuned adapter sees a
different distribution than the live agent produces.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# scripts/ isn't a package — add it to sys.path so prepare.py is importable.
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import prepare  # type: ignore  # noqa: E402
from dotfiles_agent.config import system_prompt  # noqa: E402


class TestInferPath:
	def test_zsh_keyword(self):
		assert prepare.infer_path("set up zsh", "") == "~/.zshrc"

	def test_git_keyword_in_completion(self):
		assert prepare.infer_path("", "configure git user") == "~/.gitconfig"

	def test_case_insensitive(self):
		assert prepare.infer_path("ZSH config", "") == "~/.zshrc"

	def test_first_match_wins(self):
		# both zsh and bash present — zsh comes first in PATH_HINTS (dict order)
		p = prepare.infer_path("zsh and bash", "")
		assert p in {"~/.zshrc", "~/.bashrc"}

	def test_default_when_no_keyword(self):
		assert prepare.infer_path("something random", "no keywords here") == "~/.zshrc"

	def test_all_path_hints_covered(self):
		"""Every keyword in PATH_HINTS should resolve to its own path."""
		for keyword, expected in prepare.PATH_HINTS.items():
			assert prepare.infer_path(keyword, "") == expected


class TestToToolCallConversation:
	def _row(self, prompt="do a thing", completion="alias x=y"):
		return {"prompt": prompt, "completion": completion}

	def test_message_order_and_roles(self):
		row = self._row()
		conv = prepare.to_tool_call_conversation(row)
		roles = [m["role"] for m in conv["messages"]]
		assert roles == ["system", "user", "assistant", "tool", "assistant"]

	def test_system_prompt_matches_inference(self):
		"""The system prompt in training data must match what main.py sends."""
		conv = prepare.to_tool_call_conversation(self._row())
		sys_msg = conv["messages"][0]
		assert sys_msg["content"] == system_prompt()

	def test_assistant_tool_call_shape_matches_inference(self):
		"""The assistant turn's tool_calls must have the same keys as
		build_assistant_tool_message() produces."""
		conv = prepare.to_tool_call_conversation(self._row())
		assistant = conv["messages"][2]
		assert assistant["content"] == ""
		assert assistant["tool_calls"][0]["id"] == "call_0"
		assert assistant["tool_calls"][0]["type"] == "function"
		fn = assistant["tool_calls"][0]["function"]
		assert fn["name"] == "write_file"
		# arguments is a JSON string, not a dict — matches build_assistant_tool_message
		assert isinstance(fn["arguments"], str)
		parsed = json.loads(fn["arguments"])
		assert set(parsed.keys()) == {"path", "content"}
		assert parsed["content"] == "alias x=y"

	def test_tool_result_uses_tool_call_id(self):
		"""Regression: tool results must key off tool_call_id, not name."""
		conv = prepare.to_tool_call_conversation(self._row())
		tool_msg = conv["messages"][3]
		assert tool_msg["role"] == "tool"
		assert tool_msg["tool_call_id"] == "call_0"
		assert "name" not in tool_msg

	def test_tool_result_id_matches_assistant_call_id(self):
		conv = prepare.to_tool_call_conversation(self._row())
		assistant_id = conv["messages"][2]["tool_calls"][0]["id"]
		tool_id = conv["messages"][3]["tool_call_id"]
		assert assistant_id == tool_id

	def test_completion_round_trips_through_arguments(self):
		"""Whatever the completion was, it survives as the write_file content."""
		completion = "export FOO=bar\n# {nested braces} in here"
		conv = prepare.to_tool_call_conversation({"prompt": "x", "completion": completion})
		args = json.loads(conv["messages"][2]["tool_calls"][0]["function"]["arguments"])
		assert args["content"] == completion

	def test_path_inferred_into_arguments(self):
		conv = prepare.to_tool_call_conversation({"prompt": "git config", "completion": "x"})
		args = json.loads(conv["messages"][2]["tool_calls"][0]["function"]["arguments"])
		assert args["path"] == "~/.gitconfig"


class TestJsonlRoundTrip:
	def test_write_then_load(self, tmp_path):
		rows = [{"a": 1}, {"b": {"nested": True}}, {"c": "with } brace"}]
		f = tmp_path / "out.jsonl"
		prepare.write_jsonl(f, rows)
		loaded = prepare.load_jsonl(f)
		assert loaded == rows

	def test_write_jsonl_has_trailing_newline(self, tmp_path):
		f = tmp_path / "out.jsonl"
		prepare.write_jsonl(f, [{"a": 1}])
		assert f.read_text().endswith("\n")

	def test_load_skips_blank_lines(self, tmp_path):
		f = tmp_path / "out.jsonl"
		f.write_text('{"a":1}\n\n  \n{"b":2}\n')
		assert prepare.load_jsonl(f) == [{"a": 1}, {"b": 2}]

	def test_load_empty_file(self, tmp_path):
		f = tmp_path / "empty.jsonl"
		f.write_text("")
		assert prepare.load_jsonl(f) == []
