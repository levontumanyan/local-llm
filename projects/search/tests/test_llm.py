"""Tests for the deduplication / anti-repetition logic in llm.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from search_agent.llm import _deduplicate_sentences


def test_no_repetition_unchanged():
	text = "Spain won the 2026 World Cup. They defeated Argentina 1-0."
	assert _deduplicate_sentences(text) == text


def test_repeated_sentence_gets_truncated():
	text = "Spain won. Spain won. Spain won."
	result = _deduplicate_sentences(text)
	assert result == "Spain won."


def test_repetition_mid_text():
	text = "Spain won. They defeated Argentina. They defeated Argentina. Great match."
	result = _deduplicate_sentences(text)
	assert "Great match" not in result
	assert result.count("They defeated Argentina") == 1


def test_empty_string():
	assert _deduplicate_sentences("") == ""


def test_single_sentence_unchanged():
	assert _deduplicate_sentences("Spain won.") == "Spain won."
