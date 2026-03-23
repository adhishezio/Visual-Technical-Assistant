from __future__ import annotations

from typing import Any, TypedDict, cast

from backend.core.config import EmbeddingProvider
from backend.services.embedder import (
    EMBEDDING_BATCH_LIMIT,
    RETRIEVAL_DOCUMENT_TASK,
    RETRIEVAL_QUERY_TASK,
    VertexTextEmbedder,
)


class _EmbeddingRequest(TypedDict):
    text: str
    task: str


class _ModelCall(TypedDict):
    texts: list[_EmbeddingRequest]
    auto_truncate: bool
    output_dimensionality: int | None


class _FakeEmbedding:
    def __init__(self, values: list[float]) -> None:
        self.values = values


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[_ModelCall] = []

    def get_embeddings(
        self,
        texts: list[Any],
        *,
        auto_truncate: bool = True,
        output_dimensionality: int | None = None,
    ) -> list[_FakeEmbedding]:
        typed_texts = [cast(_EmbeddingRequest, text) for text in texts]
        self.calls.append(
            {
                "texts": typed_texts,
                "auto_truncate": auto_truncate,
                "output_dimensionality": output_dimensionality,
            }
        )
        return [_FakeEmbedding([float(index), 1.0, 0.0]) for index, _ in enumerate(texts)]


def test_vertex_text_embedder_uses_retrieval_task_types(test_settings) -> None:
    settings = test_settings.model_copy(
        update={
            "embedding_provider": EmbeddingProvider.VERTEX,
            "embedding_model": "text-embedding-005",
            "embedding_dimension": 3,
            "vertex_project_id": "jobhunt-490400",
            "vertex_ai_location": "us-central1",
        }
    )
    model = _FakeModel()
    embedder = VertexTextEmbedder(
        settings=settings,
        model=cast(Any, model),
        text_input_factory=lambda text, task: {"text": text, "task": task},
    )

    document_vectors = embedder.embed_documents(
        ["Electrical specifications section", "Battery details"]
    )
    query_vector = embedder.embed_query("What is the input voltage?")

    assert document_vectors == [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    assert query_vector == [0.0, 1.0, 0.0]
    assert [item["task"] for item in model.calls[0]["texts"]] == [
        RETRIEVAL_DOCUMENT_TASK,
        RETRIEVAL_DOCUMENT_TASK,
    ]
    assert [item["task"] for item in model.calls[1]["texts"]] == [RETRIEVAL_QUERY_TASK]
    assert model.calls[0]["output_dimensionality"] == 3


def test_vertex_text_embedder_batches_requests_at_250(test_settings) -> None:
    settings = test_settings.model_copy(
        update={
            "embedding_provider": EmbeddingProvider.VERTEX,
            "embedding_model": "text-embedding-005",
            "embedding_dimension": 3,
            "vertex_project_id": "jobhunt-490400",
            "vertex_ai_location": "us-central1",
        }
    )
    model = _FakeModel()
    embedder = VertexTextEmbedder(
        settings=settings,
        model=cast(Any, model),
        text_input_factory=lambda text, task: {"text": text, "task": task},
    )

    texts = [f"Document chunk {index}" for index in range(EMBEDDING_BATCH_LIMIT * 2 + 1)]
    vectors = embedder.embed_documents(texts)

    assert len(vectors) == len(texts)
    assert [len(call["texts"]) for call in model.calls] == [250, 250, 1]
    assert all(
        item["task"] == RETRIEVAL_DOCUMENT_TASK
        for call in model.calls
        for item in call["texts"]
    )

