import re
from mlx_lm import load, generate

DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


def _deduplicate_sentences(text: str) -> str:
	"""Truncate output at the first repeated sentence — fixes 0.5B/1.5B model repetition loops."""
	sentences = re.split(r'(?<=[.!?])\s+', text.strip())
	seen = []
	for sentence in sentences:
		normalized = sentence.strip().lower()
		if normalized in seen:
			break
		seen.append(normalized)
	return " ".join(sentences[:len(seen)]).strip()


class TinyLLM:
	def __init__(self, model_name: str = DEFAULT_MODEL):
		self.model_name = model_name
		self.model = None
		self.tokenizer = None

	def load_model(self):
		if self.model is None:
			print(f"Loading model: {self.model_name}...")
			self.model, self.tokenizer = load(self.model_name)

	def synthesize_answer(
		self,
		query: str,
		context: str,
		conversation_history: list[dict] | None = None,
	) -> str:
		self.load_model()

		# Build conversation history block if present
		history_block = ""
		if conversation_history:
			lines = []
			for turn in conversation_history:
				lines.append(f"Q: {turn['question']}\nA: {turn['answer']}")
			history_block = (
				"Previous conversation:\n"
				+ "\n\n".join(lines)
				+ "\n\n"
			)

		prompt = (
			f"<|im_start|>system\n"
			f"You are a precise search synthesis assistant. Your ONLY job is to answer questions "
			f"based strictly on the provided context below. Follow these rules:\n"
			f"1. ONLY use facts explicitly stated in the context. Do NOT add outside knowledge.\n"
			f"2. If the context does not contain enough information to answer the question, say: "
			f"'The search results do not contain a clear answer to this question.'\n"
			f"3. If the event is in the future or the result is not yet known, say so explicitly.\n"
			f"4. Never guess, hallucinate, or fill in missing facts.\n"
			f"5. Be concise — answer in 2-4 sentences maximum.\n"
			f"<|im_end|>\n"
			f"<|im_start|>user\n"
			f"{history_block}"
			f"Question: {query}\n\n"
			f"Context from web search:\n{context}\n"
			f"<|im_end|>\n"
			f"<|im_start|>assistant\n"
		)

		response = generate(
			self.model,
			self.tokenizer,
			prompt=prompt,
			max_tokens=300,
			verbose=False
		)

		return _deduplicate_sentences(response.strip())
