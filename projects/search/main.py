import argparse
from concurrent.futures import ThreadPoolExecutor
import sys
import time
from search_agent.search import search_web
from search_agent.parser import fetch_pages_concurrently, PARSERS
from search_agent.llm import TinyLLM

DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

MODEL_ALIASES: dict[str, str] = {
	"small":  "mlx-community/Qwen2.5-0.5B-Instruct-4bit",   # ~350MB
	"medium": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",   # ~900MB  ← default
	"large":  "mlx-community/Qwen2.5-3B-Instruct-4bit",     # ~1.8GB
}

_FOLLOWUP_SIGNALS = {"it", "that", "they", "them", "he", "she", "his", "her", "their", "this", "those", "there"}


def _enrich_query(query: str, conversation_history: list[dict]) -> str:
	"""Append topic context from the last turn when the query looks like a follow-up.

	A query is treated as a follow-up if it is short (<=6 words) or starts with
	a pronoun/demonstrative that implies prior context.
	"""
	if not conversation_history:
		return query

	words = query.lower().split()
	is_followup = len(words) <= 6 or bool(_FOLLOWUP_SIGNALS & set(words))

	if is_followup:
		last_question = conversation_history[-1]["question"]
		return f"{query} {last_question}"

	return query


def run_query(
	query: str,
	llm: TinyLLM,
	parser: str,
	num_results: int,
	verbose: bool,
	conversation_history: list[dict],
) -> str:
	"""Run a single search query and return the answer."""
	with ThreadPoolExecutor(max_workers=4) as executor:
		model_future = executor.submit(llm.load_model)

		search_query = _enrich_query(query, conversation_history)
		print(f"🔍 Searching web for: '{search_query}'...")
		t_search = time.perf_counter()
		results = search_web(search_query, max_results=num_results)
		t_search_end = time.perf_counter()

		if not results:
			print("No search results found.")
			return ""

		urls = [res["url"] for res in results]
		print(f"📄 Fetching & parsing {len(urls)} pages via [{parser}]...")
		t_parse = time.perf_counter()
		pages_content = fetch_pages_concurrently(urls, parser=parser)
		t_parse_end = time.perf_counter()

		context_blocks = []
		for res, content in zip(results, pages_content):
			if content and not content.startswith("Error fetching"):
				context_blocks.append(f"### Source: {res['title']}\nURL: {res['url']}\n\n{content}")

		model_future.result()

	if not context_blocks:
		print("\n❌ Could not retrieve any page content from search results.")
		return ""

	combined_context = "\n\n".join(context_blocks)

	if verbose:
		print("\n" + "─" * 50)
		print("CONTEXT SENT TO LLM:")
		print("─" * 50)
		print(combined_context)
		print("─" * 50 + "\n")

	print("\n🤖 Synthesizing answer...")
	t_llm = time.perf_counter()
	answer = llm.synthesize_answer(query, combined_context, conversation_history)
	t_llm_end = time.perf_counter()

	t_total = t_llm_end - t_search

	print(f"\n⏱  search {t_search_end - t_search:.2f}s  |  parse {t_parse_end - t_parse:.2f}s  |  llm {t_llm_end - t_llm:.2f}s  |  total {t_total:.2f}s")

	return answer


def interactive_loop(llm: TinyLLM, parser: str, num_results: int, verbose: bool):
	"""Run an interactive REPL — model stays loaded, conversation history accumulates."""
	conversation_history: list[dict] = []

	print("\n💬 Interactive search mode. Type your question or 'exit' to quit.\n")

	while True:
		try:
			query = input("You: ").strip()
		except (EOFError, KeyboardInterrupt):
			print("\n\nExiting.")
			break

		if not query:
			continue
		if query.lower() in ("exit", "quit", "q"):
			print("Exiting.")
			break

		answer = run_query(query, llm, parser, num_results, verbose, conversation_history)

		if answer:
			print("\n" + "=" * 50)
			print(answer)
			print("=" * 50 + "\n")
			conversation_history.append({"question": query, "answer": answer})


def main():
	parser = argparse.ArgumentParser(description="Local web search & text understanding agent")
	parser.add_argument("--query", "-q", type=str, help="Single query mode. Omit for interactive REPL.")
	parser.add_argument("--interactive", "-i", action="store_true", help="Force interactive REPL mode")
	parser.add_argument("--num-results", "-n", type=int, default=3, help="Number of search results to fetch & parse")
	parser.add_argument("--model", "-m", type=str, default="medium",
		help="Model shorthand (small | medium | large) or full HF repo path. Default: medium (~900MB)")
	parser.add_argument("--parser", "-p", type=str, default="trafilatura", choices=PARSERS,
		help="Web page parser backend: jina (hosted API), trafilatura (local), readability (local Mozilla algo)")
	parser.add_argument("--verbose", "-v", action="store_true", help="Print full context being sent to the LLM")
	args = parser.parse_args()

	model_name = MODEL_ALIASES.get(args.model, args.model)
	llm = TinyLLM(model_name=model_name)

	# Interactive mode: no --query given, or --interactive flag
	if args.interactive or not args.query:
		print(f"⚡ Loading model [{args.model}] on first query...")
		interactive_loop(llm, args.parser, args.num_results, args.verbose)
		return

	# Single-shot mode
	print(f"⚡ Starting background MLX model load & parallel web search... [parser: {args.parser}]")
	answer = run_query(args.query, llm, args.parser, args.num_results, args.verbose, [])

	if answer:
		print("\n" + "=" * 50)
		print("ANSWER:")
		print("=" * 50)
		print(answer)
		print("=" * 50 + "\n")


if __name__ == "__main__":
	main()
