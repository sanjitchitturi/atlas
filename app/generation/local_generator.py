"""
Local (no-API-key) extractive generator.

Strategy:
1. Build a context string from retrieved chunks.
2. Score each sentence by keyword overlap with the question.
3. Return the top-scoring sentences as the answer.
   If no sentence passes the threshold, return a "don't know" message.
"""
from __future__ import annotations
import re
from app.generation.base import BaseGenerator
from app.models.schemas import DocumentChunk


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


class LocalGenerator(BaseGenerator):
    MIN_OVERLAP = 1          # minimum shared tokens to consider a sentence
    MAX_SENTENCES = 5        # how many sentences to include in the answer

    def generate(self, question: str, chunks: list[DocumentChunk]) -> str:
        if not chunks:
            return "I could not find any relevant information in the knowledge base."

        q_tokens = _tokenize(question)
        scored: list[tuple[float, str]] = []

        for chunk in chunks:
            sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 20:
                    continue
                s_tokens = _tokenize(sent)
                overlap = len(q_tokens & s_tokens)
                if overlap >= self.MIN_OVERLAP:
                    scored.append((overlap, sent))

        if not scored:
            return (
                "Based on the retrieved documents, I cannot find a specific answer "
                "to your question. Please try rephrasing or upload more relevant documents."
            )

        scored.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s for _, s in scored[: self.MAX_SENTENCES]]

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for s in top_sentences:
            if s not in seen:
                seen.add(s)
                unique.append(s)

        return " ".join(unique)
