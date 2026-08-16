"""Entrypoint wrapper for the tiny-search CLI.

This module lives at the repo root so uv can register it as a project script
without needing PYTHONPATH tricks. It adds projects/tiny-search to sys.path
so the search_agent package resolves correctly regardless of where uv is
invoked from.
"""
import sys
from pathlib import Path

# Ensure search_agent is importable regardless of cwd
_SEARCH_ROOT = Path(__file__).parent / "projects" / "search"
if str(_SEARCH_ROOT) not in sys.path:
	sys.path.insert(0, str(_SEARCH_ROOT))

from main import main  # noqa: E402


if __name__ == "__main__":
	main()
