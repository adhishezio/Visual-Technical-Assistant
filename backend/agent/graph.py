from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from backend.agent.nodes import TechnicalAssistantNodes, build_safe_answer
from backend.agent.state import AgentState
from backend.core.config import Settings, get_settings
from backend.core.models import AnswerWithCitations
from backend.services.fetcher import DocumentFetcher
from backend.services.search import DocumentationSearchService
from backend.services.vision import VisionIdentificationService
from backend.vector_store.base import VectorStore


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
    ) -> AnswerWithCitations:
        initial_state: AgentState = {
            "image_bytes": image_bytes,
            "mime_type": mime_type,
            "question": question,
            "retrieved_chunks": [],
            "documentation_candidates": [],
            "fetch_attempts": 0,
        }
        final_state = self.graph.invoke(initial_state)
        return final_state.get("answer") or build_safe_answer()



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

    graph.add_edge(START, "identify_component")
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
