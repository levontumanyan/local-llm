"""Tests for the parser backends — no real HTTP calls, mocked via pytest-monkeypatch."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from search_agent.parser import fetch_page_markdown, TRUNCATE_CHARS


def _make_response(status: int, text: str) -> MagicMock:
	resp = MagicMock()
	resp.status_code = status
	resp.text = text
	return resp


class TestJinaParser:
	def test_returns_markdown_on_200(self):
		with patch("search_agent.parser.httpx.get") as mock_get:
			mock_get.return_value = _make_response(200, "# Title\nContent here")
			result = fetch_page_markdown("https://example.com", parser="jina")
		assert "Content here" in result

	def test_returns_error_on_non_200(self):
		with patch("search_agent.parser.httpx.get") as mock_get:
			mock_get.return_value = _make_response(404, "Not Found")
			result = fetch_page_markdown("https://example.com", parser="jina")
		# Non-200 → _fetch_with_jina returns "" → empty-content guard fires → error string
		assert result.startswith("Error fetching content from")

	def test_truncates_long_content(self):
		long_text = "x" * (TRUNCATE_CHARS + 500)
		with patch("search_agent.parser.httpx.get") as mock_get:
			mock_get.return_value = _make_response(200, long_text)
			result = fetch_page_markdown("https://example.com", parser="jina")
		assert len(result) <= TRUNCATE_CHARS + len("\n...[content truncated]")
		assert result.endswith("...[content truncated]")

	def test_returns_error_string_on_exception(self):
		with patch("search_agent.parser.httpx.get", side_effect=Exception("timeout")):
			result = fetch_page_markdown("https://example.com", parser="jina")
		assert result.startswith("Error fetching content from")


class TestTrafilaturaParser:
	def test_returns_extracted_text(self):
		article_html = "<html><body><p>Argentina won the World Cup.</p></body></html>"
		with patch("search_agent.parser.httpx.get") as mock_get, \
			patch("search_agent.parser.trafilatura.extract", return_value="Argentina won the World Cup."):
			mock_get.return_value = _make_response(200, article_html)
			result = fetch_page_markdown("https://example.com", parser="trafilatura")
		assert "Argentina won" in result

	def test_returns_error_string_on_exception(self):
		with patch("search_agent.parser.httpx.get", side_effect=Exception("connection error")):
			result = fetch_page_markdown("https://example.com", parser="trafilatura")
		assert result.startswith("Error fetching content from")


def test_unknown_parser_returns_error():
	result = fetch_page_markdown("https://example.com", parser="unknown_parser")
	assert "Unknown parser" in result
