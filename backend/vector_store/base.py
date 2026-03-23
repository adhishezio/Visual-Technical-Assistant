from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from backend.core.models import CacheKey, DocumentChunk, DocumentMetadata, RetrievedChunk


class VectorStore(ABC):
    @abstractmethod
    def add_documents(
        self,
        chunks: Sequence[DocumentChunk],
        metadata: DocumentMetadata,
    ) -> None:
        """Add document chunks to the store."""

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int,
        filter_key: CacheKey | None = None,
    ) -> list[RetrievedChunk]:
        """Return the most similar chunks for a query."""

    @abstractmethod
    def key_exists(self, cache_key: CacheKey) -> bool:
        """Return whether a component cache key already exists."""

    @abstractmethod
    def delete_by_key(self, cache_key: CacheKey) -> None:
        """Delete all chunks associated with a cache key."""

    @abstractmethod
    def health_check(self) -> bool:
        """Check whether the vector store is available."""
