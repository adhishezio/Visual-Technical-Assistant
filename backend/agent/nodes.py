from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field

from backend.agent.prompts import (
    SAFE_NO_CITATIONS_MESSAGE,
    build_answer_generation_prompt,
    build_chunk_grading_prompt,
)
from backend.agent.state import AgentState
from backend.core.config import Settings, get_settings
from backend.core.models import AnswerWithCitations, ComponentIdentification, RetrievedChunk
from backend.services.fetcher import DocumentFetcher
from backend.services.gemini import GeminiServiceError, generate_structured_content
from backend.services.search import DocumentationSearchService
from backend.services.vision import VisionIdentificationService
from backend.vector_store import get_vector_store
from backend.vector_store.base import VectorStore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ChunkGrade(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sufficient: bool
    confidence: float = 0.0
    refined_query: str | None = None
    reasoning: str = ""


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_text: str
    confidence: float = 0.0
    citation_indexes: list[int] = Field(default_factory=list)



def build_safe_answer() -> AnswerWithCitations:
    return AnswerWithCitations(
        answer_text=SAFE_NO_CITATIONS_MESSAGE,
        citations=[],
        confidence=0.0,
    )



def enforce_cited_answer(answer: AnswerWithCitations | None) -> AnswerWithCitations:
    if answer and answer.has_citations:
        return answer
    return build_safe_answer()


class TechnicalAssistantNodes:
    def __init__(
        self,
        settings: Settings | None = None,
        vision_service: VisionIdentificationService | None = None,
        search_service: DocumentationSearchService | None = None,
        fetcher: DocumentFetcher | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.vision_service = vision_service or VisionIdentificationService(self.settings)
        self.search_service = search_service or DocumentationSearchService(self.settings)
        self.fetcher = fetcher or DocumentFetcher(self.settings)
        self.vector_store = vector_store or get_vector_store(settings=self.settings)

    def identify_component(self, state: AgentState) -> AgentState:
        identification = self.vision_service.identify_component(
            image_bytes=state["image_bytes"],
            mime_type=state.get("mime_type") or "image/jpeg",
        )
        logger.info(
            "[identify] manufacturer=%r model=%r part=%r type=%r confidence=%.2f lookup=%s error=%r",
            identification.manufacturer,
            identification.model_number,
            identification.part_number,
            identification.component_type,
            identification.confidence_score,
            identification.should_attempt_document_lookup,
            identification.error_details,
        )
        return {
            "identification": identification,
            "current_node": "identify_component",
            "error": identification.error_details,
        }

    def build_cache_key(self, state: AgentState) -> AgentState:
        identification = state.get("identification")
        if identification is None:
            logger.info("[build_cache_key] missing_identification=True")
            return {
                "cache_key": None,
                "error": "Identification failed before cache key generation.",
                "current_node": "build_cache_key",
            }
        cache_key = identification.to_cache_key()
        logger.info("[build_cache_key] key=%s", cache_key.value)
        return {
            "cache_key": cache_key,
            "current_node": "build_cache_key",
        }

    def check_cache(self, state: AgentState) -> AgentState:
        identification = state.get("identification")
        cache_key = state.get("cache_key")
        try:
            cache_hit = bool(
                identification
                and cache_key
                and identification.should_attempt_document_lookup
                and self.vector_store.key_exists(cache_key)
            )
        except Exception as exc:
            logger.info(
                "[check_cache] key=%s exists=False error=%r",
                cache_key.value if cache_key else None,
                str(exc),
            )
            return {
                "cache_hit": False,
                "error": str(exc),
                "current_node": "check_cache",
            }
        logger.info(
            "[check_cache] key=%s exists=%s lookup=%s",
            cache_key.value if cache_key else None,
            cache_hit,
            identification.should_attempt_document_lookup if identification else False,
        )
        return {
            "cache_hit": cache_hit,
            "current_node": "check_cache",
        }

    def fetch_documentation(self, state: AgentState) -> AgentState:
        identification = state.get("identification")
        cache_key = state.get("cache_key")
        fetch_attempts = state.get("fetch_attempts", 0) + 1
        if identification is None or cache_key is None:
            logger.info(
                "[fetch] attempt=%s missing_identification=%s missing_cache_key=%s",
                fetch_attempts,
                identification is None,
                cache_key is None,
            )
            return {
                "documentation_candidates": [],
                "fetch_attempts": fetch_attempts,
                "error": "Cannot fetch documentation without identification and cache key.",
                "current_node": "fetch_documentation",
            }

        try:
            documentation_candidates = self.search_service.search(
                identification=identification,
                refined_query=state.get("refined_query"),
            )
        except Exception as exc:
            logger.info(
                "[search] attempt=%s refined_query=%r results=0 error=%r",
                fetch_attempts,
                state.get("refined_query"),
                str(exc),
            )
            return {
                "documentation_candidates": [],
                "fetch_attempts": fetch_attempts,
                "error": str(exc),
                "current_node": "fetch_documentation",
            }

        logger.info(
            "[search] attempt=%s refined_query=%r results=%s urls=%s",
            fetch_attempts,
            state.get("refined_query"),
            len(documentation_candidates),
            [candidate.url for candidate in documentation_candidates[:5]],
        )

        fetched_any = False
        last_error: str | None = None
        fetched_count = 0
        failed_count = 0
        total_chunks = 0
        for candidate in documentation_candidates:
            try:
                chunks = self.fetcher.fetch(candidate, identification, cache_key)
            except Exception as exc:
                last_error = str(exc)
                failed_count += 1
                continue
            if not chunks:
                continue
            self.vector_store.add_documents(chunks=chunks, metadata=chunks[0].metadata)
            fetched_any = True
            fetched_count += 1
            total_chunks += len(chunks)

        logger.info(
            "[fetch] attempt=%s fetched=%s failed=%s chunks=%s error=%r",
            fetch_attempts,
            fetched_count,
            failed_count,
            total_chunks,
            None if fetched_any else (last_error or "No documentation could be ingested."),
        )

        return {
            "documentation_candidates": documentation_candidates,
            "fetch_attempts": fetch_attempts,
            "error": None if fetched_any else (last_error or "No documentation could be ingested."),
            "current_node": "fetch_documentation",
        }

    def retrieve_chunks(self, state: AgentState) -> AgentState:
        cache_key = state.get("cache_key")
        if cache_key is None:
            logger.info("[retrieve] key=None chunks=0")
            return {
                "retrieved_chunks": [],
                "current_node": "retrieve_chunks",
            }
        try:
            retrieved_chunks = self.vector_store.similarity_search(
                query=state["question"],
                k=self.settings.similarity_search_k,
                filter_key=cache_key,
            )
        except Exception as exc:
            logger.info(
                "[retrieve] key=%s chunks=0 error=%r",
                cache_key.value,
                str(exc),
            )
            return {
                "retrieved_chunks": [],
                "error": str(exc),
                "current_node": "retrieve_chunks",
            }
        logger.info(
            "[retrieve] key=%s chunks=%s top_sources=%s",
            cache_key.value,
            len(retrieved_chunks),
            [retrieved.chunk.metadata.source_url for retrieved in retrieved_chunks[:3]],
        )
        return {
            "retrieved_chunks": retrieved_chunks,
            "current_node": "retrieve_chunks",
        }

    def grade_chunks(self, state: AgentState) -> AgentState:
        identification = state.get("identification")
        retrieved_chunks = state.get("retrieved_chunks", [])
        fetch_attempts = state.get("fetch_attempts", 0)
        if not retrieved_chunks:
            needs_refetch = fetch_attempts < self.settings.max_fetch_attempts
            fallback_query = (
                self._fallback_refined_query(identification, state["question"])
                if needs_refetch
                else None
            )
            logger.info(
                "[grade] chunks=0 passed=0 needs_refetch=%s reason=%r refined_query=%r",
                needs_refetch,
                "no retrieved chunks",
                fallback_query,
            )
            return {
                "needs_refetch": needs_refetch,
                "refined_query": fallback_query,
                "current_node": "grade_chunks",
            }

        try:
            grade = generate_structured_content(
                prompt=build_chunk_grading_prompt(
                    identification=identification,
                    question=state["question"],
                    retrieved_chunks=retrieved_chunks,
                ),
                response_schema=ChunkGrade,
                settings=self.settings,
            )
        except GeminiServiceError:
            grade = self._heuristic_grade(identification, state["question"], retrieved_chunks)

        needs_refetch = (not grade.sufficient) and (
            fetch_attempts < self.settings.max_fetch_attempts
        )
        refined_query = (
            grade.refined_query
            if needs_refetch and grade.refined_query
            else self._fallback_refined_query(identification, state["question"])
            if needs_refetch
            else None
        )
        logger.info(
            "[grade] chunks=%s passed=%s needs_refetch=%s confidence=%.2f reason=%r refined_query=%r",
            len(retrieved_chunks),
            int(bool(grade.sufficient)),
            needs_refetch,
            grade.confidence,
            grade.reasoning,
            refined_query,
        )
        return {
            "needs_refetch": needs_refetch,
            "refined_query": refined_query,
            "current_node": "grade_chunks",
        }

    def generate_answer(self, state: AgentState) -> AgentState:
        retrieved_chunks = state.get("retrieved_chunks", [])
        if not retrieved_chunks:
            logger.info("[generate] chunks=0 has_citations=False")
            return {
                "answer": build_safe_answer(),
                "current_node": "generate_answer",
            }

        try:
            generated = generate_structured_content(
                prompt=build_answer_generation_prompt(
                    question=state["question"],
                    retrieved_chunks=retrieved_chunks,
                ),
                response_schema=GeneratedAnswer,
                settings=self.settings,
            )
        except GeminiServiceError:
            logger.info(
                "[generate] chunks=%s has_citations=False error=%r",
                len(retrieved_chunks),
                "generation failed",
            )
            return {
                "answer": build_safe_answer(),
                "current_node": "generate_answer",
            }

        citations = [
            retrieved_chunks[index]
            for index in generated.citation_indexes
            if 0 <= index < len(retrieved_chunks)
        ]
        answer = AnswerWithCitations(
            answer_text=generated.answer_text,
            citations=citations,
            confidence=generated.confidence,
        )
        logger.info(
            "[generate] chunks=%s citation_indexes=%s has_citations=%s confidence=%.2f",
            len(retrieved_chunks),
            generated.citation_indexes,
            answer.has_citations,
            answer.confidence,
        )
        return {
            "answer": answer,
            "current_node": "generate_answer",
        }

    def validate_citations(self, state: AgentState) -> AgentState:
        validated = enforce_cited_answer(state.get("answer"))
        logger.info(
            "[validate] has_citations=%s confidence=%.2f",
            validated.has_citations,
            validated.confidence,
        )
        return {
            "answer": validated,
            "current_node": "validate_citations",
        }

    @staticmethod
    def route_after_cache_check(state: AgentState) -> str:
        identification = state.get("identification")
        if identification is None or not identification.should_attempt_document_lookup:
            return "generate_answer"
        if state.get("cache_hit"):
            return "retrieve_chunks"
        return "fetch_documentation"

    @staticmethod
    def route_after_grade(state: AgentState) -> str:
        return "fetch_documentation" if state.get("needs_refetch") else "generate_answer"

    @staticmethod
    def _fallback_refined_query(
        identification: ComponentIdentification | None,
        question: str,
    ) -> str | None:
        if identification is None:
            return None
        key_token = (
            identification.part_number
            or identification.model_number
            or identification.component_type
            or "component"
        )
        manufacturer = identification.manufacturer or ""
        return f"{manufacturer} {key_token} official manual {question} pdf".strip()

    @staticmethod
    def _heuristic_grade(
        identification: ComponentIdentification | None,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> ChunkGrade:
        del identification
        del question
        return ChunkGrade(sufficient=bool(retrieved_chunks), confidence=0.5)
