"""
Evaluation metrics for the RAG pipeline.

Metrics implemented (heuristic, no paid APIs needed):

1. context_relevance   – fraction of retrieved chunks that share tokens with the question
2. answer_faithfulness – fraction of answer sentences that are grounded in the context
3. retrieval_hit       – 1 if the expected source appears in retrieved sources, else 0
"""
from __future__ import annotations
import re


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w{3,}\b", text.lower()))


def context_relevance(question: str, chunk_texts: list[str]) -> float:
    """
    Returns [0, 1]: proportion of chunks that share at least one content
    token with the question.
    """
    if not chunk_texts:
        return 0.0
    q_tokens = _tokenize(question)
    hits = sum(
        1 for c in chunk_texts if len(_tokenize(c) & q_tokens) >= 1
    )
    return round(hits / len(chunk_texts), 4)


def answer_faithfulness(answer: str, chunk_texts: list[str]) -> float:
    """
    Returns [0, 1]: proportion of answer sentences whose tokens overlap
    significantly with at least one retrieved chunk.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 10]
    if not sentences:
        return 0.0

    context_tokens = _tokenize(" ".join(chunk_texts))
    faithful = 0
    for sent in sentences:
        s_tokens = _tokenize(sent)
        overlap = len(s_tokens & context_tokens)
        if overlap >= max(1, len(s_tokens) * 0.3):   # 30% overlap threshold
            faithful += 1

    return round(faithful / len(sentences), 4)


def retrieval_hit(expected_source: str, retrieved_sources: list[str]) -> int:
    """
    Binary: 1 if expected_source filename appears in any retrieved source, else 0.
    """
    expected_lower = expected_source.lower()
    for src in retrieved_sources:
        if expected_lower in src.lower():
            return 1
    return 0
