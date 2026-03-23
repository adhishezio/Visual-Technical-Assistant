from __future__ import annotations

from backend.core.config import Settings, VectorStoreProvider, get_settings
from backend.services.embedder import TextEmbedder, get_embedder
from backend.vector_store.base import VectorStore
from backend.vector_store.chroma import ChromaVectorStore
from backend.vector_store.vertex import VertexVectorStore



def get_vector_store(
    settings: Settings | None = None,
    embedder: TextEmbedder | None = None,
) -> VectorStore:
    resolved_settings = settings or get_settings()
    resolved_embedder = embedder or get_embedder(resolved_settings)

    if resolved_settings.vector_store is VectorStoreProvider.CHROMA:
        return ChromaVectorStore(settings=resolved_settings, embedder=resolved_embedder)
    if resolved_settings.vector_store is VectorStoreProvider.VERTEX:
        return VertexVectorStore(settings=resolved_settings, embedder=resolved_embedder)

    raise ValueError(f"Unsupported vector store provider: {resolved_settings.vector_store}")
