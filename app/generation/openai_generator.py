"""OpenAI-based answer generator (requires OPENAI_API_KEY)."""
from __future__ import annotations
import json
import requests
from app.generation.base import BaseGenerator
from app.models.schemas import DocumentChunk
from app.core.config import settings
from app.core.logging import logger

_SYSTEM = (
    "You are Atlas, a precise engineering knowledge assistant. "
    "Answer the user's question using ONLY the context provided below. "
    "If the context does not contain enough information, say you don't know. "
    "Be concise and factual. Do not speculate."
)


def _build_context(chunks: list[DocumentChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[Source {i}: {c.source}, page {c.page}]\n{c.text}")
    return "\n\n---\n\n".join(parts)


class OpenAIGenerator(BaseGenerator):
    def generate(self, question: str, chunks: list[DocumentChunk]) -> str:
        context = _build_context(chunks)
        user_msg = f"Context:\n{context}\n\nQuestion: {question}"
        payload = {
            "model": settings.openai_model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        }
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.error(f"OpenAI generation failed: {exc}")
            raise
