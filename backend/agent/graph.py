from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from backend.agent.nodes import TechnicalAssistantNodes, build_safe_answer
from backend.agent.state import AgentState
from backend.core.config import Settings, get_settings
from backend.core.models import AnswerWithCitations, ComponentIdentification
from backend.services.fetcher import DocumentFetcher
from backend.services.search import DocumentationSearchService
from backend.services.vision import VisionIdentificationService
from backend.vector_store.base import VectorStore

logger = logging.getLogger(__name__)


class VisualTechnicalAssistantAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        vision_service: VisionIdentificationService | None = None,
        search_service: DocumentationSearchService | None = None,
        fetcher: DocumentFetcher | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.nodes = TechnicalAssistantNodes(
            settings=self.settings,
            vision_service=vision_service,
            search_service=search_service,
            fetcher=fetcher,
            vector_store=vector_store,
        )
        self.graph = build_graph(self.nodes)

    def run(
        self,
        image_bytes: bytes,
        question: str,
        mime_type: str = "image/jpeg",
        identification=None,
    ) -> AnswerWithCitations:
        initial_state: AgentState = {
            "image_bytes": image_bytes,
            "mime_type": mime_type,
            "question": question,
            "identification": identification,
            "reused_identification": identification is not None,
            "retrieved_chunks": [],
            "documentation_candidates": [],
            "fetch_attempts": 0,
        }
        final_state = self.graph.invoke(initial_state)
        return final_state.get("answer") or build_safe_answer()

    def prime_cache(self, identification: ComponentIdentification) -> bool:
        if not identification.should_attempt_document_lookup:
            return False

        cache_key = identification.to_cache_key()
        if self.nodes.vector_store.key_exists(cache_key):
            return True

        documentation_candidates = self.nodes.search_service.search(identification)
        fetched_documents = 0

        for candidate in documentation_candidates:
            try:
                chunks = self.nodes.fetcher.fetch(candidate, identification, cache_key)
            except Exception as exc:
                logger.info(
                    "[prime_cache] skipped url=%s reason=%r",
                    candidate.url,
                    str(exc),
                )
                continue
            if not chunks:
                continue
            self.nodes.vector_store.add_documents(chunks=chunks, metadata=chunks[0].metadata)
            fetched_documents += 1
            if fetched_documents >= 3:
                break

        logger.info(
            "[prime_cache] key=%s fetched_documents=%s",
            cache_key.value,
            fetched_documents,
        )
        return fetched_documents > 0



def build_graph(nodes: TechnicalAssistantNodes):
    graph = StateGraph(AgentState)
    graph.add_node("identify_component", nodes.identify_component)
    graph.add_node("build_cache_key", nodes.build_cache_key)
    graph.add_node("check_cache", nodes.check_cache)
    graph.add_node("fetch_documentation", nodes.fetch_documentation)
    graph.add_node("retrieve_chunks", nodes.retrieve_chunks)
    graph.add_node("grade_chunks", nodes.grade_chunks)
    graph.add_node("generate_answer", nodes.generate_answer)
    graph.add_node("validate_citations", nodes.validate_citations)

    graph.add_conditional_edges(
        START,
        lambda state: "build_cache_key" if state.get("identification") else "identify_component",
        {
            "identify_component": "identify_component",
            "build_cache_key": "build_cache_key",
        },
    )
    graph.add_edge("identify_component", "build_cache_key")
    graph.add_edge("build_cache_key", "check_cache")
    graph.add_conditional_edges(
        "check_cache",
        nodes.route_after_cache_check,
        {
            "fetch_documentation": "fetch_documentation",
            "retrieve_chunks": "retrieve_chunks",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("fetch_documentation", "retrieve_chunks")
    graph.add_edge("retrieve_chunks", "grade_chunks")
    graph.add_conditional_edges(
        "grade_chunks",
        nodes.route_after_grade,
        {
            "fetch_documentation": "fetch_documentation",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("generate_answer", "validate_citations")
    graph.add_edge("validate_citations", END)
    return graph.compile()
