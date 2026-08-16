from ddgs import DDGS


def search_web(query: str, max_results: int = 3) -> list[dict[str, str]]:
	"""Search DuckDuckGo using ddgs and return a list of result dictionaries containing title, href, body."""
	results = []
	with DDGS() as ddgs:
		ddg_results = ddgs.text(query, max_results=max_results)
		for r in ddg_results:
			results.append({
				"title": r.get("title", ""),
				"url": r.get("href", ""),
				"snippet": r.get("body", "")
			})
	return results
