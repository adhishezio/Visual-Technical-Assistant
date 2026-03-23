from __future__ import annotations

from datetime import datetime, timezone
from typing import cast
from unittest.mock import patch

from backend.agent.graph import VisualTechnicalAssistantAgent
from backend.agent.nodes import GeneratedAnswer, TechnicalAssistantNodes, build_safe_answer
from backend.agent.state import AgentState
from backend.core.models import (
    AnswerWithCitations,
    CacheKey,
    ComponentIdentification,
    DocumentChunk,
    DocumentationCandidate,
    DocumentMetadata,
    DocumentType,
    RetrievedChunk,
)
from backend.services.fetcher import DocumentFetcher
from backend.services.search import DocumentationSearchService
from backend.vector_store.base import VectorStore


class FakeSearchService:
    def search(self, identification, refined_query=None):
        del identification
        del refined_query
        return [
            DocumentationCandidate(
                url="https://example.com/bad",
                title="Broken page",
                document_type=DocumentType.DATASHEET,
                score=0.8,
            ),
            DocumentationCandidate(
                url="https://example.com/good",
                title="Good page",
                document_type=DocumentType.DATASHEET,
                score=0.7,
            ),
        ]


class FakeFetcher:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def fetch(self, candidate, identification, cache_key):
        del identification
        self.calls.append(candidate.url)
        if candidate.url.endswith("/bad"):
            raise RuntimeError("parser failure")

        metadata = DocumentMetadata(
            source_url=candidate.url,
            source_title=candidate.title,
            manufacturer="ABB",
            model_number="S 204 M B 40 UC",
            part_number="2CD274061R0405",
            document_type=DocumentType.DATASHEET,
            revision=None,
            page_map={0: 1},
            retrieved_at=datetime(2026, 3, 23, 12, 0, tzinfo=timezone.utc),
            content_hash="abc123",
            cache_key=cache_key,
        )
        return [
            DocumentChunk(
                chunk_text="Rated current 40A.",
                chunk_index=0,
                metadata=metadata,
                page_number=1,
                section_title="Electrical data",
            )
        ]


class FakeVectorStore:
    def __init__(self, exists: bool = False) -> None:
        self.added_chunks: list[list[DocumentChunk]] = []
        self.exists = exists

    def add_documents(self, chunks, metadata) -> None:
        del metadata
        self.added_chunks.append(chunks)

    def similarity_search(self, query, k, filter_key=None):
        del query
        del k
        del filter_key
        return []

    def key_exists(self, cache_key: CacheKey) -> bool:
        del cache_key
        return self.exists

    def delete_by_key(self, cache_key: CacheKey) -> None:
        del cache_key

    def health_check(self) -> bool:
        return True


def build_retrieved_chunk() -> RetrievedChunk:
    cache_key = CacheKey.from_parts("ABB", "S 204 M B 40 UC", "2CD274061R0405")
    metadata = DocumentMetadata(
        source_url="https://example.com/specs.pdf",
        source_title="ABB breaker specs",
        manufacturer="ABB",
        model_number="S 204 M B 40 UC",
        part_number="2CD274061R0405",
        document_type=DocumentType.DATASHEET,
        revision=None,
        page_map={0: 1},
        retrieved_at=datetime(2026, 3, 23, 12, 0, tzinfo=timezone.utc),
        content_hash="xyz789",
        cache_key=cache_key,
    )
    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_text="Rated current 40A. Rated insulation voltage 440V.",
            chunk_index=0,
            metadata=metadata,
            page_number=1,
            section_title="Electrical data",
        ),
        similarity_score=0.94,
    )


def test_prime_cache_skips_bad_candidate_and_ingests_next_result(test_settings) -> None:
    fetcher = FakeFetcher()
    vector_store = FakeVectorStore()
    agent = VisualTechnicalAssistantAgent(
        settings=test_settings,
        search_service=cast(DocumentationSearchService, FakeSearchService()),
        fetcher=cast(DocumentFetcher, fetcher),
        vector_store=cast(VectorStore, vector_store),
    )
    identification = ComponentIdentification(
        manufacturer="ABB",
        model_number="S 204 M B 40 UC",
        part_number="2CD274061R0405",
        component_type="Miniature Circuit Breaker",
        confidence_score=1.0,
        raw_ocr_text="ABB S 204 M B 40 UC 2CD274061R0405",
        should_attempt_document_lookup=True,
    )

    primed = agent.prime_cache(identification)

    assert primed is True
    assert fetcher.calls == ["https://example.com/bad", "https://example.com/good"]
    assert len(vector_store.added_chunks) == 1
    assert vector_store.added_chunks[0][0].chunk_text == "Rated current 40A."


def test_prime_cache_returns_immediately_when_lookup_not_allowed(test_settings) -> None:
    agent = VisualTechnicalAssistantAgent(
        settings=test_settings,
        search_service=cast(DocumentationSearchService, FakeSearchService()),
        fetcher=cast(DocumentFetcher, FakeFetcher()),
        vector_store=cast(VectorStore, FakeVectorStore()),
    )
    identification = ComponentIdentification(
        manufacturer="ABB",
        model_number="S 204 M B 40 UC",
        part_number="2CD274061R0405",
        component_type="Miniature Circuit Breaker",
        confidence_score=0.3,
        raw_ocr_text="ABB",
        should_attempt_document_lookup=False,
    )

    primed = agent.prime_cache(identification)

    assert primed is False


def test_prime_cache_skips_search_when_component_is_already_cached(test_settings) -> None:
    class ExplodingSearchService:
        def search(self, identification, refined_query=None):
            del identification
            del refined_query
            raise AssertionError("Search should not run when cache already exists.")

    agent = VisualTechnicalAssistantAgent(
        settings=test_settings,
        search_service=cast(DocumentationSearchService, ExplodingSearchService()),
        fetcher=cast(DocumentFetcher, FakeFetcher()),
        vector_store=cast(VectorStore, FakeVectorStore(exists=True)),
    )
    identification = ComponentIdentification(
        manufacturer="ABB",
        model_number="S 204 M B 40 UC",
        part_number="2CD274061R0405",
        component_type="Miniature Circuit Breaker",
        confidence_score=1.0,
        raw_ocr_text="ABB S 204 M B 40 UC 2CD274061R0405",
        should_attempt_document_lookup=True,
    )

    primed = agent.prime_cache(identification)

    assert primed is True


def test_generate_answer_falls_back_to_extractive_when_llm_returns_no_citations(
    test_settings,
) -> None:
    retrieved_chunk = build_retrieved_chunk()
    nodes = TechnicalAssistantNodes(
        settings=test_settings,
        search_service=cast(DocumentationSearchService, FakeSearchService()),
        fetcher=cast(DocumentFetcher, FakeFetcher()),
        vector_store=cast(VectorStore, FakeVectorStore()),
    )
    state: AgentState = {
        "image_bytes": b"",
        "mime_type": "image/jpeg",
        "question": "What is the rated current for this component?",
        "identification": None,
        "reused_identification": False,
        "answer_from_identification": False,
        "retrieved_chunks": [retrieved_chunk],
        "documentation_candidates": [],
        "fetch_attempts": 0,
    }

    with patch(
        "backend.agent.nodes.generate_structured_content",
        return_value=GeneratedAnswer(
            answer_text="The current rating is available in the datasheet.",
            confidence=0.9,
            citation_indexes=[],
        ),
    ):
        result = nodes.generate_answer(state)

    answer = result["answer"]
    assert answer is not None
    assert answer.has_citations is True
    assert "40A" in answer.answer_text


def test_generate_answer_uses_identification_fast_path_for_model_questions(
    test_settings,
) -> None:
    nodes = TechnicalAssistantNodes(
        settings=test_settings,
        search_service=cast(DocumentationSearchService, FakeSearchService()),
        fetcher=cast(DocumentFetcher, FakeFetcher()),
        vector_store=cast(VectorStore, FakeVectorStore()),
    )
    identification = ComponentIdentification(
        manufacturer="ABB",
        model_number="S202",
        part_number="2CDS252001R0404",
        component_type="Miniature circuit breaker",
        confidence_score=0.86,
        raw_ocr_text="ABB S202 2CDS252001R0404",
        should_attempt_document_lookup=False,
    )
    state: AgentState = {
        "image_bytes": b"",
        "mime_type": "image/jpeg",
        "question": "What is the model number?",
        "identification": identification,
        "reused_identification": True,
        "answer_from_identification": False,
        "retrieved_chunks": [],
        "documentation_candidates": [],
        "fetch_attempts": 0,
    }

    result = nodes.generate_answer(state)

    answer = result["answer"]
    assert answer is not None
    assert answer.has_citations is False
    assert "model number appears to be S202" in answer.answer_text
    assert result["answer_from_identification"] is True


def test_validate_citations_preserves_identification_fast_path_answer(
    test_settings,
) -> None:
    nodes = TechnicalAssistantNodes(
        settings=test_settings,
        search_service=cast(DocumentationSearchService, FakeSearchService()),
        fetcher=cast(DocumentFetcher, FakeFetcher()),
        vector_store=cast(VectorStore, FakeVectorStore()),
    )
    uncited_answer = AnswerWithCitations(
        answer_text="From the image identification, the manufacturer appears to be ABB.",
        citations=[],
        confidence=0.78,
    )
    state: AgentState = {
        "answer": uncited_answer,
        "answer_from_identification": True,
    }

    result = nodes.validate_citations(state)

    validated = result["answer"]
    assert validated is not None
    assert validated.answer_text == uncited_answer.answer_text
    assert validated.has_citations is False


def test_validate_citations_rejects_uncited_non_identification_answers(
    test_settings,
) -> None:
    nodes = TechnicalAssistantNodes(
        settings=test_settings,
        search_service=cast(DocumentationSearchService, FakeSearchService()),
        fetcher=cast(DocumentFetcher, FakeFetcher()),
        vector_store=cast(VectorStore, FakeVectorStore()),
    )
    state: AgentState = {
        "answer": AnswerWithCitations(
            answer_text="Unsupported uncited answer",
            citations=[],
            confidence=0.66,
        ),
        "answer_from_identification": False,
    }

    result = nodes.validate_citations(state)

    validated = result["answer"]
    assert validated is not None
    assert validated.answer_text == build_safe_answer().answer_text
    assert validated.has_citations is False
