from __future__ import annotations

from typing import TypedDict

from backend.core.models import (
    AnswerWithCitations,
    CacheKey,
    ComponentIdentification,
    DocumentationCandidate,
    RetrievedChunk,
)


class AgentState(TypedDict, total=False):
    image_bytes: bytes
    mime_type: str
    question: str
    identification: ComponentIdentification | None
    cache_key: CacheKey | None
    documentation_candidates: list[DocumentationCandidate]
    retrieved_chunks: list[RetrievedChunk]
    answer: AnswerWithCitations | None
    error: str | None
    current_node: str
    cache_hit: bool
    needs_refetch: bool
    refined_query: str | None
    fetch_attempts: int
    reused_identification: bool
