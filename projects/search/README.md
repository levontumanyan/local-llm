# Tiny Search Agent

Local web search, page parsing, and text explanation pipeline powered by tiny sub-500M LLMs (`Qwen2.5-0.5B-Instruct` or `SmolLM2-360M-Instruct`) on Apple Silicon via MLX.

# TODO List & Roadmap

## Search Providers
- [x] DuckDuckGo (`duckduckgo_search`) integration (Zero-config, no API key required)
- [ ] Tavily API search provider (`TAVILY_API_KEY`)
- [ ] Brave Search API provider (`BRAVE_API_KEY`)
- [ ] SearXNG local instance provider (`SEARXNG_URL`)

## Web Parsing & Content Extraction
- [x] Jina Reader API (`r.jina.ai`) for instant Markdown conversion
- [ ] Local HTML-to-Markdown fallback parser (`BeautifulSoup4` + `html2text`)
- [ ] Summarization context truncation to fit 4k window limit

## Models & Inference
- [x] `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (Default)
- [ ] `mlx-community/SmolLM2-360M-Instruct-4bit` alternative model flag
- [ ] Interactive CLI mode & streaming response support

# Quick Start

```sh
# Run a query directly from the terminal
uv run python projects/tiny-search/main.py --query "What are the latest features in Python 3.13?"
```
