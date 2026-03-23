from __future__ import annotations

import logging
import re

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
        identification = state.get("identification")
        if cache_key is None:
            logger.info("[retrieve] key=None chunks=0")
            return {
                "retrieved_chunks": [],
                "current_node": "retrieve_chunks",
            }
        retrieval_query = self._build_retrieval_query(
            state["question"],
            identification,
        )
        try:
            retrieved_chunks = self.vector_store.similarity_search(
                query=retrieval_query,
                k=self.settings.similarity_search_k,
                filter_key=cache_key,
            )
        except Exception as exc:
            logger.info(
                "[retrieve] key=%s query=%r chunks=0 error=%r",
                cache_key.value,
                retrieval_query,
                str(exc),
            )
            return {
                "retrieved_chunks": [],
                "error": str(exc),
                "current_node": "retrieve_chunks",
            }
        logger.info(
            "[retrieve] key=%s query=%r chunks=%s top_sources=%s",
            cache_key.value,
            retrieval_query,
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

        if state.get("reused_identification"):
            grade = self._heuristic_grade(identification, state["question"], retrieved_chunks)
        else:
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
                grade = self._heuristic_grade(
                    identification,
                    state["question"],
                    retrieved_chunks,
                )

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

        extractive_answer = self._build_extractive_answer(
            state["question"],
            retrieved_chunks,
        )
        if self._should_use_extractive_fast_path(state["question"], extractive_answer):
            logger.info(
                "[generate] chunks=%s mode=extractive_fast has_citations=%s",
                len(retrieved_chunks),
                extractive_answer.has_citations,
            )
            return {
                "answer": extractive_answer,
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
            fallback_answer = self._build_extractive_answer(
                state["question"],
                retrieved_chunks,
            )
            logger.info(
                "[generate] chunks=%s has_citations=%s error=%r",
                len(retrieved_chunks),
                fallback_answer.has_citations,
                "generation failed",
            )
            return {
                "answer": fallback_answer,
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
        if not answer.has_citations and extractive_answer.has_citations:
            logger.info(
                "[generate] chunks=%s mode=extractive_fallback has_citations=%s",
                len(retrieved_chunks),
                extractive_answer.has_citations,
            )
            return {
                "answer": extractive_answer,
                "current_node": "generate_answer",
            }
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
        question_tokens = _question_tokens(question)
        needs_electrical_value = bool(
            {"voltage", "current", "power", "watt", "amp", "input", "operating"}
            & question_tokens
        )
        best_overlap = 0
        found_measurement = False

        for retrieved in retrieved_chunks:
            lowered = retrieved.chunk.chunk_text.lower()
            overlap = sum(token in lowered for token in question_tokens)
            best_overlap = max(best_overlap, overlap)
            if re.search(
                r"\b\d+(?:\.\d+)?\s?(?:v|vac|vdc|a|ma|w|kw|hz)\b",
                lowered,
            ):
                found_measurement = True

        if needs_electrical_value:
            sufficient = best_overlap >= 1 and found_measurement
            return ChunkGrade(
                sufficient=sufficient,
                confidence=0.75 if sufficient else 0.2,
                reasoning=(
                    "retrieved chunks mention the requested electrical property"
                    if sufficient
                    else "retrieved chunks do not clearly mention the requested electrical specification"
                ),
            )

        sufficient = best_overlap >= 2
        return ChunkGrade(
            sufficient=sufficient,
            confidence=0.6 if sufficient else 0.25,
            reasoning=(
                "retrieved chunks overlap with the question terms"
                if sufficient
                else "retrieved chunks have low lexical overlap with the question"
            ),
        )

    @staticmethod
    def _build_retrieval_query(
        question: str,
        identification: ComponentIdentification | None,
    ) -> str:
        if identification is None:
            return question

        tokens = [
            question.strip(),
            identification.manufacturer or "",
            identification.model_number or "",
            identification.part_number or "",
            identification.component_type or "",
        ]
        return " ".join(token for token in tokens if token).strip()

    @staticmethod
    def _build_extractive_answer(
        question: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> AnswerWithCitations:
        if not retrieved_chunks:
            return build_safe_answer()

        question_tokens = {
            token for token in _question_tokens(question)
        }
        needs_electrical_value = bool(
            {"voltage", "current", "power", "watt", "amp", "input", "operating"}
            & question_tokens
        )
        best_sentence = ""
        best_chunk: RetrievedChunk | None = None
        best_score = -1

        for retrieved in retrieved_chunks:
            sentences = re.split(r"(?<=[.!?])\s+|\n+", retrieved.chunk.chunk_text)
            for sentence in sentences:
                cleaned = " ".join(sentence.split()).strip()
                if len(cleaned) < 20:
                    continue
                lowered = cleaned.lower()
                overlap = sum(token in lowered for token in question_tokens)
                has_measurement = bool(
                    needs_electrical_value
                    and re.search(
                        r"\b\d+(?:\.\d+)?\s?(?:v|vac|vdc|a|ma|w|kw|hz)\b",
                        lowered,
                    )
                )
                value_bonus = 2 if has_measurement else 0
                score = overlap + value_bonus
                if score > best_score:
                    best_score = score
                    best_sentence = cleaned
                    best_chunk = retrieved

        if best_chunk is None or best_score <= 0:
            return build_safe_answer()

        if needs_electrical_value:
            if not re.search(
                r"\b\d+(?:\.\d+)?\s?(?:v|vac|vdc|a|ma|w|kw|hz)\b",
                best_sentence.lower(),
            ):
                return build_safe_answer()
            formatted = _extract_electrical_answer(
                best_chunk.chunk.chunk_text,
                question,
            )
            if formatted:
                best_sentence = formatted

        return AnswerWithCitations(
            answer_text=best_sentence,
            citations=[best_chunk],
            confidence=0.42,
        )

    @staticmethod
    def _should_use_extractive_fast_path(
        question: str,
        extractive_answer: AnswerWithCitations,
    ) -> bool:
        if not extractive_answer.has_citations:
            return False

        question_tokens = _question_tokens(question)
        return bool(
            question_tokens
            & {
                "voltage",
                "current",
                "power",
                "watt",
                "amp",
                "frequency",
                "rating",
                "rated",
                "standard",
                "standards",
                "certification",
                "certifications",
                "temperature",
            }
        )


def _question_tokens(question: str) -> set[str]:
    stopwords = {
        "what",
        "which",
        "this",
        "that",
        "with",
        "from",
        "your",
        "have",
        "into",
        "about",
        "component",
        "device",
        "please",
        "could",
        "would",
        "there",
        "their",
        "input",
        "operating",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", question.lower())
        if len(token) > 2 and token not in stopwords
    }


def _extract_electrical_answer(chunk_text: str, question: str) -> str | None:
    question_lower = question.lower()
    if "voltage" in question_lower:
        patterns = [
            r"(?:input voltage|operating voltage|power supply|device input)[^.;\n]{0,48}\b\d+(?:\.\d+)?(?:\s?-\s?\d+(?:\.\d+)?)?\s?V(?:\s?(?:AC|DC))?\b",
        ]
    elif "current" in question_lower or "amp" in question_lower:
        patterns = [
            r"(?:current|power supply|device input)[^.;\n]{0,48}\b\d+(?:\.\d+)?\s?(?:A|mA)\b",
        ]
    elif "power" in question_lower or "watt" in question_lower:
        patterns = [
            r"(?:power consumption|power supply)[^.;\n]{0,48}\b\d+(?:\.\d+)?\s?(?:W|kW)\b",
        ]
    else:
        patterns = [
            r"(?:power supply|device input|input voltage|operating voltage)[^.;\n]{0,90}\b\d+(?:\.\d+)?\s?(?:v|vac|vdc)\b[^.;\n]{0,30}",
            r"(?:power consumption|current|frequency)[^.;\n]{0,70}\b\d+(?:\.\d+)?\s?(?:a|ma|w|kw|hz)\b[^.;\n]{0,20}",
        ]
    for pattern in patterns:
        for match in re.finditer(pattern, chunk_text, flags=re.IGNORECASE):
            cleaned = " ".join(match.group(0).split()).strip(" -")
            cleaned = re.sub(
                r"\b(\d{3})(\d{3})\s+V\b",
                r"\1-\2 V",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(
                r"\b(\d{3})(\d{3})\s+V\s+(AC|DC)\b",
                r"\1-\2 V \3",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(
                r"\b(\d{2})(\d{2})\s+Hz\b",
                r"\1-\2 Hz",
                cleaned,
                flags=re.IGNORECASE,
            )
            if cleaned:
                return f"{cleaned}."
    return None
