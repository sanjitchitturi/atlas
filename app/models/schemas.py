"""Pydantic schemas for API request/response and internal data models."""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    page: int = 0
    chunk_index: int = 0
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestedDocument(BaseModel):
    doc_id: str
    source: str
    num_chunks: int
    status: str = "ok"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class SourceReference(BaseModel):
    source: str
    chunk_id: str
    chunk_index: int
    page: int
    snippet: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceReference]
    mode: str


class IngestResponse(BaseModel):
    documents: list[IngestedDocument]
    total_chunks: int
    index_size: int


class HealthResponse(BaseModel):
    status: str
    index_size: int
    embedding_model: str
    generation_mode: str
