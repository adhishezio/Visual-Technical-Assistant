from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Callable, Protocol, Sequence, cast

from backend.core.config import EmbeddingProvider, Settings, get_settings

EMBEDDING_BATCH_LIMIT = 250
RETRIEVAL_DOCUMENT_TASK = "RETRIEVAL_DOCUMENT"
RETRIEVAL_QUERY_TASK = "RETRIEVAL_QUERY"


class EmbedderServiceError(RuntimeError):
    """Raised when embeddings cannot be generated."""


class TextEmbedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple documents."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""


class _EmbeddingLike(Protocol):
    values: Sequence[float]


class _VertexEmbeddingModelLike(Protocol):
    def get_embeddings(
        self,
        texts: list[Any],
        *,
        auto_truncate: bool = True,
        output_dimensionality: int | None = None,
    ) -> list[_EmbeddingLike]:
        ...


class HashingTextEmbedder:
    """Deterministic local embedder suitable for tests and fallback use."""

    def __init__(self, dimension: int = 768) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = re.findall(r"[a-z0-9-]+", text.lower()) or [text.lower()]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], byteorder="big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class VertexTextEmbedder:
    """Vertex-backed text embedder with task-type-aware batching."""

    def __init__(
        self,
        settings: Settings | None = None,
        model: _VertexEmbeddingModelLike | None = None,
        text_input_factory: Callable[[str, str], object] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model = model or self._load_model()
        self.text_input_factory = text_input_factory or _build_text_embedding_input

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed_texts(texts=texts, task_type=RETRIEVAL_DOCUMENT_TASK)

    def embed_query(self, text: str) -> list[float]:
        vectors = self._embed_texts(texts=[text], task_type=RETRIEVAL_QUERY_TASK)
        if not vectors:
            return [0.0] * self.settings.embedding_dimension
        return vectors[0]

    def _embed_texts(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        indexed_texts = [(index, text.strip()) for index, text in enumerate(texts)]
        non_empty = [(index, text) for index, text in indexed_texts if text]
        if not non_empty:
            return [[0.0] * self.settings.embedding_dimension for _ in texts]

        resolved_vectors = [[0.0] * self.settings.embedding_dimension for _ in texts]
        for batch_start in range(0, len(non_empty), EMBEDDING_BATCH_LIMIT):
            batch = non_empty[batch_start : batch_start + EMBEDDING_BATCH_LIMIT]
            inputs = [
                self.text_input_factory(text, task_type)
                for _, text in batch
            ]
            try:
                response = self.model.get_embeddings(
                    inputs,
                    auto_truncate=True,
                    output_dimensionality=self.settings.embedding_dimension,
                )
            except Exception as exc:
                raise EmbedderServiceError(f"Embedding request failed: {exc}") from exc

            vectors = [list(map(float, embedding.values)) for embedding in response]
            if len(vectors) != len(batch):
                raise EmbedderServiceError(
                    "Embedding response length did not match the number of requested texts."
                )
            for (original_index, _), vector in zip(batch, vectors):
                resolved_vectors[original_index] = vector

        return resolved_vectors

    def _load_model(self) -> _VertexEmbeddingModelLike:
        try:
            import vertexai
            from vertexai.language_models import TextEmbeddingModel
        except ImportError as exc:
            raise EmbedderServiceError(
                "google-cloud-aiplatform is required to generate Vertex embeddings."
            ) from exc

        if not self.settings.vertex_project_id:
            raise EmbedderServiceError(
                "VERTEX_PROJECT_ID is required for Vertex embedding requests."
            )
        if not self.settings.vertex_ai_location:
            raise EmbedderServiceError(
                "VERTEX_AI_LOCATION is required for Vertex embedding requests."
            )

        vertexai.init(
            project=self.settings.vertex_project_id,
            location=self.settings.vertex_ai_location,
        )
        return cast(
            _VertexEmbeddingModelLike,
            TextEmbeddingModel.from_pretrained(self.settings.embedding_model),
        )


def _build_text_embedding_input(text: str, task_type: str) -> object:
    try:
        from vertexai.language_models import TextEmbeddingInput
    except ImportError as exc:
        raise EmbedderServiceError(
            "google-cloud-aiplatform is required to generate Vertex embeddings."
        ) from exc
    return TextEmbeddingInput(text, task_type)


def get_embedder(settings: Settings | None = None) -> TextEmbedder:
    resolved_settings = settings or get_settings()

    if resolved_settings.embedding_provider is EmbeddingProvider.HASHING:
        return HashingTextEmbedder(dimension=resolved_settings.embedding_dimension)
    if resolved_settings.embedding_provider is EmbeddingProvider.VERTEX:
        return VertexTextEmbedder(settings=resolved_settings)

    raise ValueError(f"Unsupported embedding provider: {resolved_settings.embedding_provider}")
