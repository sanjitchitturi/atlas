"""Abstract base class for answer generators."""
from __future__ import annotations
from abc import ABC, abstractmethod
from app.models.schemas import DocumentChunk


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, question: str, chunks: list[DocumentChunk]) -> str:
        """Return an answer grounded in *chunks*."""
