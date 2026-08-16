"""Tests for the search query enrichment logic."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import _enrich_query, _FOLLOWUP_SIGNALS


def test_no_history_returns_query_unchanged():
	assert _enrich_query("who won the world cup", []) == "who won the world cup"


def test_short_query_gets_enriched():
	history = [{"question": "who won the 2022 world cup", "answer": "Argentina"}]
	result = _enrich_query("who scored the most", history)
	assert "who scored the most" in result
	assert "who won the 2022 world cup" in result


def test_long_unambiguous_query_not_enriched():
	history = [{"question": "who won the 2022 world cup", "answer": "Argentina"}]
	query = "what are the most popular programming languages in 2024 according to stackoverflow"
	result = _enrich_query(query, history)
	assert result == query  # long query, no followup signals → unchanged


def test_pronoun_triggers_enrichment():
	history = [{"question": "who won the 2022 world cup", "answer": "Argentina"}]
	result = _enrich_query("what was their record in the tournament", history)
	assert "who won the 2022 world cup" in result  # "their" is a followup signal


def test_demonstrative_triggers_enrichment():
	history = [{"question": "tell me about the Eiffel Tower", "answer": "It is in Paris"}]
	result = _enrich_query("how tall is it", history)
	assert "tell me about the Eiffel Tower" in result


def test_followup_signals_set_is_not_empty():
	assert len(_FOLLOWUP_SIGNALS) > 0
	assert "they" in _FOLLOWUP_SIGNALS
	assert "it" in _FOLLOWUP_SIGNALS
