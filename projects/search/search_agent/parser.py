import httpx
import html2text
import trafilatura
from concurrent.futures import ThreadPoolExecutor
from readability import Document

TRUNCATE_CHARS = 6000
PARSERS = ["jina", "trafilatura", "readability"]


# ─── Backend: Jina Reader ──────────────────────────────────────────────────────

def _fetch_with_jina(url: str, timeout: int = 30) -> str:
	"""Fetch page as clean Markdown via Jina Reader hosted API (r.jina.ai)."""
	jina_url = f"https://r.jina.ai/{url}"
	headers = {
		"Accept": "text/plain",
		"X-Return-Format": "markdown",
		"X-Remove-Selector": "nav, header, footer, aside",
	}
	response = httpx.get(jina_url, headers=headers, timeout=timeout, follow_redirects=True)
	if response.status_code == 200:
		return response.text
	return ""


# ─── Backend: trafilatura ──────────────────────────────────────────────────────

def _fetch_with_trafilatura(url: str, timeout: int = 30) -> str:
	"""Fetch page via httpx (with timeout) then extract article body via trafilatura."""
	response = httpx.get(url, timeout=timeout, follow_redirects=True, headers={
		"User-Agent": "Mozilla/5.0 (compatible; tiny-search/1.0)"
	})
	if response.status_code != 200:
		return ""
	text = trafilatura.extract(
		response.text,
		output_format="markdown",
		include_tables=False,
		include_comments=False,
		no_fallback=False,
	)
	return text or ""


# ─── Backend: readability-lxml + html2text ────────────────────────────────────

def _fetch_with_readability(url: str, timeout: int = 30) -> str:
	"""Fetch page, apply Mozilla Readability extraction, convert to Markdown (local)."""
	response = httpx.get(url, timeout=timeout, follow_redirects=True, headers={
		"User-Agent": "Mozilla/5.0 (compatible; tiny-search/1.0)"
	})
	if response.status_code != 200:
		return ""
	doc = Document(response.text)
	article_html = doc.summary()
	converter = html2text.HTML2Text()
	converter.ignore_links = True
	converter.ignore_images = True
	converter.body_width = 0
	return converter.handle(article_html)


# ─── Public API ───────────────────────────────────────────────────────────────

def fetch_page_markdown(url: str, parser: str = "jina", timeout: int = 30) -> str:
	"""Fetch and parse a URL using the specified parser backend.

	Args:
		url: The URL to fetch.
		parser: One of 'jina', 'trafilatura', 'readability'.
		timeout: Request timeout in seconds.

	Returns:
		Clean Markdown string, truncated to TRUNCATE_CHARS. Returns error string on failure.
	"""
	try:
		if parser == "jina":
			text = _fetch_with_jina(url, timeout)
		elif parser == "trafilatura":
			text = _fetch_with_trafilatura(url, timeout)
		elif parser == "readability":
			text = _fetch_with_readability(url, timeout)
		else:
			return f"Unknown parser: {parser}"

		if not text or not text.strip():
			return f"Error fetching content from {url}: empty response"

		if len(text) > TRUNCATE_CHARS:
			text = text[:TRUNCATE_CHARS] + "\n...[content truncated]"
		return text

	except Exception as e:
		return f"Error fetching content from {url}: {e}"


def fetch_pages_concurrently(
	urls: list[str],
	parser: str = "jina",
	max_workers: int = 4,
) -> list[str]:
	"""Fetch multiple URLs concurrently using the specified parser backend."""
	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		results = list(executor.map(lambda u: fetch_page_markdown(u, parser=parser), urls))
	return results
