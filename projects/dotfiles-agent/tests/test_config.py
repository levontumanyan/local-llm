"""Tests for dotfiles_agent.config — paths, env parsing, system prompt loading."""

from __future__ import annotations

import importlib
from pathlib import Path

import dotfiles_agent.config as config


def test_project_paths_exist():
	"""TOOLS_PATH and PROMPT_PATH must point at real files committed in the repo."""
	assert config.TOOLS_PATH.is_file(), f"missing: {config.TOOLS_PATH}"
	assert config.PROMPT_PATH.is_file(), f"missing: {config.PROMPT_PATH}"
	assert config.PROJECT_DIR.name == "dotfiles-agent"


def test_system_prompt_loaded_from_file():
	"""system_prompt() reads prompts/system.md, not the hard-coded fallback."""
	prompt = config.system_prompt()
	fallback = config._FALLBACK_PROMPT
	assert prompt != fallback, "system_prompt() returned the fallback, not the file"
	assert "dotfiles assistant" in prompt.lower()
	# cached — same object on second call
	assert config.system_prompt() is prompt


def test_system_prompt_falls_back_when_file_missing(tmp_path, monkeypatch):
	"""If the prompt file is gone, the fallback string is used (not an error)."""
	monkeypatch.setattr(config, "PROMPT_PATH", tmp_path / "nonexistent.md")
	# lru_cache must be cleared so the cached value from the real file is dropped
	config.system_prompt.cache_clear()
	try:
		assert config.system_prompt() == config._FALLBACK_PROMPT
	finally:
		config.system_prompt.cache_clear()


def test_max_tool_iters_env_override(monkeypatch):
	"""MAX_TOOL_ITERS is parsed from the env at import time."""
	monkeypatch.setenv("MAX_TOOL_ITERS", "5")
	importlib.reload(config)
	try:
		assert config.MAX_TOOL_ITERS == 5
	finally:
		monkeypatch.delenv("MAX_TOOL_ITERS", raising=False)
		importlib.reload(config)


def test_max_tokens_env_override(monkeypatch):
	monkeypatch.setenv("MAX_TOKENS", "1234")
	importlib.reload(config)
	try:
		assert config.MAX_TOKENS == 1234
	finally:
		monkeypatch.delenv("MAX_TOKENS", raising=False)
		importlib.reload(config)
